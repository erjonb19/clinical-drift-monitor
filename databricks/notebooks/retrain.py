# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: the retraining loop, champion against challenger (one GPU job)
# MAGIC Under `results/phase3/retrain/spec.json`. Refuses unless the monthly drift table holds an
# MAGIC alert for a month with Buenos Aires images. Fine-tunes a challenger from each champion
# MAGIC seed on the clients' training split plus the labelled Buenos Aires images the monthly job
# MAGIC scored (their labels have arrived), then compares the two ensembles on Buenos Aires patients
# MAGIC the challenger never saw and on the clients' validation split, and applies the promotion
# MAGIC rule fixed in the spec. The decision and results are written before any checkpoint.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/retrain",
    "deadline_minutes": "45",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")], check=True
)

# COMMAND ----------

import copy
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyspark.sql import functions as F
from torch.utils.data import DataLoader

from cdm.data import eval_transform, train_transform
from cdm.ood import extract
from cdm.serving import balanced_accuracy
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash
from cdm.vlm import clusters, paired_bootstrap

START = time.time()
DEADLINE = START + 60 * float(dbutils.widgets.get("deadline_minutes"))
CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
INDEX = {c: i for i, c in enumerate(CLASSES)}

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
for rel, digest in json.loads(dbutils.widgets.get("files_sha256")).items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if (OUT / "decision.json").exists():
    raise RuntimeError(f"{OUT / 'decision.json'} exists: this retraining already ran")

# The trigger: an alert in a month with Buenos Aires images.
monthly = spark.table("workspace.cdm.gold_drift_monthly").toPandas()
monthly["ba"] = monthly["summary"].map(
    lambda s: json.loads(s).get("by_source", {}).get("buenos_aires", 0)
)
trigger = monthly[(monthly["alert"]) & (monthly["ba"] > 0)]
if trigger.empty:
    raise RuntimeError("no alert in a month with Buenos Aires images: nothing to retrain for")
print("trigger months:", sorted(trigger["month"]), flush=True)

# COMMAND ----------

silver = (
    spark.read.table("workspace.cdm.silver_images")
    .select(
        "isic_id", "source", "role", "client", "split", "label", "lesion_id", "patient_id", "image"
    )
    .toPandas()
)
assert not silver["isic_id"].duplicated().any()
scored = set(
    spark.table("workspace.cdm.gold_prediction_audit")
    .where(F.col("month").between(5, 9))
    .toPandas()["isic_id"]
)
ba = silver[(silver["source"] == "buenos_aires") & silver["label"].notna()]
ba_train = ba[ba["isic_id"].isin(scored)]


def person(r: pd.Series) -> str:
    """The patient, or the lesion where no patient is recorded."""
    return (
        r["patient_id"]
        if isinstance(r["patient_id"], str) and r["patient_id"]
        else f"lesion:{r['lesion_id']}"
    )


seen = {person(r) for _, r in ba_train.iterrows()}
ba_held = ba[~ba["isic_id"].isin(scored) & ~ba.apply(lambda r: person(r) in seen, axis=1)]
clients = silver[(silver["role"] == "client") & silver["label"].notna()]
train_rows, val_rows = clients[clients["split"] == "train"], clients[clients["split"] == "val"]
sizes = {
    "client_train": len(train_rows),
    "ba_train": len(ba_train),
    "ba_held_out": len(ba_held),
    "client_val": len(val_rows),
}
print(sizes, flush=True)

cfg = TrainConfig(arch="b0", image_size=224, view="center", epochs=3, lr=1e-4, num_workers=4)


def dataset(frame: pd.DataFrame, augment: bool) -> EncodedImages:
    transform = (
        train_transform(cfg.image_size, cfg.rotate, cfg.color_jitter, cfg.view, cfg.color_constancy)
        if augment
        else eval_transform(cfg.image_size)
    )
    return EncodedImages(list(frame["image"]), frame["label"].map(INDEX).tolist(), transform)


train_set = dataset(pd.concat([train_rows, ba_train], ignore_index=True), True)
val_set, held_set = dataset(val_rows, False), dataset(ba_held, False)
device = torch.device("cuda")
manifest = {m["seed"]: m for m in json.loads(Path(dbutils.widgets.get("models_file")).read_text())}


def softmax_on(model: torch.nn.Module, data: EncodedImages) -> np.ndarray:
    with torch.no_grad():
        _, logits, _ = extract(model, DataLoader(data, batch_size=128, num_workers=4), device)
    z = np.exp(logits - logits.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


probs = {"champion": {"held": [], "val": []}, "challenger": {"held": [], "val": []}}
challengers = {}
for seed in (0, 1, 2):
    if time.time() > DEADLINE:
        raise RuntimeError("deadline reached: stopping inside the GPU cap")
    champ = build_model(pretrained=False, arch="b0")
    champ.load_state_dict(torch.load(manifest[seed]["checkpoint"], map_location="cpu"))
    assert weights_hash(champ) == manifest[seed]["weights_sha256"]
    champ = champ.to(device).eval()
    probs["champion"]["held"].append(softmax_on(champ, held_set))
    probs["champion"]["val"].append(softmax_on(champ, val_set))
    generator = seed_everything(seed, cfg.num_threads, gpu=True)
    model, history = train(copy.deepcopy(champ), train_set, val_set, cfg, seed, device, generator)
    model.eval()
    probs["challenger"]["held"].append(softmax_on(model, held_set))
    probs["challenger"]["val"].append(softmax_on(model, val_set))
    challengers[seed] = (model, history)
    print(
        f"seed {seed}: trained, best epoch val {max(h['val_bacc'] for h in history):.4f}",
        flush=True,
    )

# COMMAND ----------

y_held = np.asarray(held_set.labels, dtype=np.int64)
y_val = np.asarray(val_set.labels, dtype=np.int64)
ens = {
    k: {s: np.log(np.clip(np.mean(v[s], axis=0), 1e-12, 1)) for s in ("held", "val")}
    for k, v in probs.items()
}
units = clusters(list(ba_held["patient_id"]), list(ba_held["lesion_id"]), list(ba_held["isic_id"]))
paired = paired_bootstrap(ens["challenger"]["held"], ens["champion"]["held"], y_held, units, n=1000)
metrics = {
    k: {"held_out_balanced_accuracy": balanced_accuracy(y_held, ens[k]["held"].argmax(1)),
        "client_val_balanced_accuracy": balanced_accuracy(y_val, ens[k]["val"].argmax(1))}
    for k in ens
}  # fmt: skip
gain = (
    metrics["challenger"]["held_out_balanced_accuracy"]
    - metrics["champion"]["held_out_balanced_accuracy"]
)
val_change = (
    metrics["challenger"]["client_val_balanced_accuracy"]
    - metrics["champion"]["client_val_balanced_accuracy"]
)
promoted = bool(gain >= 0.02 and paired["balanced_accuracy"]["low"] > 0 and val_change >= -0.01)
decision = {
    "spec": "results/phase3/retrain/spec.json",
    "code_version": CODE, "wheel_sha256": WHEEL_SHA, "ran_on": f"Databricks serverless GPU (A10), job run {JOB_RUN}",
    "trigger_months": [int(m) for m in sorted(trigger["month"])], "sizes": sizes,
    "champion": metrics["champion"], "challenger": metrics["challenger"],
    "held_out_gain_points": 100 * gain, "client_val_change_points": 100 * val_change,
    "paired_bootstrap_held_out": paired,
    "rule": "gain >= 2.0 points, paired lower bound > 0, client validation change >= -1.0 point",
    "promoted": promoted,
    "next_step": "rebuild a serving bundle for the challenger before any registry change" if promoted else "champion kept; no registry change",
    "training_history": {f"seed{s}": h for s, (_, h) in challengers.items()},
    "seconds": round(time.time() - START),
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
}  # fmt: skip
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
spark.createDataFrame(pd.DataFrame([{
    "decided_utc": decision["completed_utc"], "job_run": JOB_RUN, "promoted": promoted,
    "held_out_gain_points": float(100 * gain), "client_val_change_points": float(100 * val_change),
    "record": json.dumps(decision, default=float)}])).write.mode("append").saveAsTable("workspace.cdm.gold_model_decisions")  # fmt: skip
if promoted:
    ck = Path(f"/Volumes/workspace/cdm/raw/checkpoints/phase3/challenger-{JOB_RUN}")
    ck.mkdir(parents=True, exist_ok=True)
    for seed, (model, _) in challengers.items():
        torch.save(model.state_dict(), f"/tmp/c{seed}.pt")
        (ck / f"seed{seed}.pt").write_bytes(Path(f"/tmp/c{seed}.pt").read_bytes())
print(
    json.dumps(
        {
            k: decision[k]
            for k in ("sizes", "champion", "challenger", "held_out_gain_points", "promoted")
        },
        default=float,
    )
)
