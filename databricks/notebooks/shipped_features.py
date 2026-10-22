# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase E follow-up: the shipped model's own features (one GPU job)
# MAGIC Penultimate features (1,280 per image) from each of the three shipped EfficientNet-B0 v2
# MAGIC checkpoints, for every image of the train, val, test, Buenos Aires and PAD-UFES-20 sets,
# MAGIC in exactly the row order of the Phase B BiomedCLIP embeddings (`meta.csv`), so the two
# MAGIC feature sets line up row for row. Checkpoints are checked by weight hash. Before anything
# MAGIC is saved, each seed's softmax on the test split must match its committed analysis-look
# MAGIC probabilities (`predictions.csv.gz`) row for row: mean absolute gap in the top probability
# MAGIC at most 0.02 (GPU against CPU). No metric is computed here. A deadline stops the job well
# MAGIC inside the owner's 2-hour GPU cap.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "embeddings": "/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings",
    "out": "/Volumes/workspace/cdm/raw/results/research/e_fusion/shipped_features",
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

import hashlib
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.ood import extract
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

START = time.time()
DEADLINE = START + 60 * float(dbutils.widgets.get("deadline_minutes"))
CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
SETS = ("train", "val", "test", "buenos_aires", "pad_ufes")
SEEDS = (0, 1, 2)

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if (OUT / "summary.json").exists():
    raise RuntimeError(f"{OUT / 'summary.json'} exists: the features were already extracted")

meta = pd.read_csv(Path(dbutils.widgets.get("embeddings")) / "meta.csv", dtype=str)
images = (
    spark.read.table("workspace.cdm.silver_images")
    .select("source", "isic_id", "image")
    .toPandas()
    .set_index(["source", "isic_id"])["image"]
)
device = torch.device("cuda")
manifest = {m["seed"]: m for m in json.loads(Path(dbutils.widgets.get("models_file")).read_text())}
pred = pd.read_csv(REPO / "results/phase2/analysis_look/predictions.csv.gz").set_index("isic_id")

# COMMAND ----------

features: dict[str, np.ndarray] = {}
gaps: dict[str, float] = {}
for seed in SEEDS:
    model = build_model(pretrained=False, arch="b0")
    model.load_state_dict(torch.load(manifest[seed]["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == manifest[seed]["weights_sha256"], (
        f"seed {seed} checkpoint mismatch"
    )
    model = model.to(device).eval()
    for name in SETS:
        if time.time() > DEADLINE:
            raise RuntimeError("deadline reached: stopping inside the GPU cap")
        rows = meta[meta["set"] == name]
        blobs = [images.loc[(s, i)] for s, i in zip(rows["source"], rows["isic_id"], strict=True)]
        data = EncodedImages(blobs, [0] * len(blobs), eval_transform(224))
        loader = DataLoader(data, batch_size=128, num_workers=4, pin_memory=True)
        with torch.no_grad():
            feats, logits, _ = extract(model, loader, device)
        features[f"seed{seed}__{name}"] = feats.astype(np.float32)
        if name == "test":
            top = torch.softmax(torch.from_numpy(logits), dim=1).numpy().max(axis=1)
            committed = pred.loc[
                list(rows["isic_id"]), [f"v2-final-seed{seed}__{c}" for c in CLASSES]
            ]
            gap = float(np.mean(np.abs(top - committed.to_numpy().max(axis=1))))
            if gap > 0.02:
                raise RuntimeError(
                    f"seed {seed}: test top-probability gap {gap:.4f}: rows do not line up"
                )
            gaps[f"seed{seed}"] = gap
        print(f"seed {seed} {name}: {feats.shape}", flush=True)
    del model
    torch.cuda.empty_cache()

# COMMAND ----------

# Volumes do not support the seeks a zip writer needs: write locally, check, then copy.
local = Path(tempfile.gettempdir()) / "features.npz"
np.savez(local, **features)
with np.load(local) as z:
    assert sorted(z.files) == sorted(features), "npz did not read back"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "features.npz").write_bytes(local.read_bytes())
summary = {
    "code_version": CODE,
    "wheel_sha256": WHEEL_SHA,
    "job_run_id": JOB_RUN,
    "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "device": torch.cuda.get_device_name(0),
    "models": {f"seed{s}": manifest[s]["weights_sha256"] for s in SEEDS},
    "shape": {k: list(v.shape) for k, v in features.items()},
    "row_order": "the Phase B embeddings' meta.csv, set by set",
    "test_top_probability_gap": gaps,
    "seconds_total": round(time.time() - START),
    "sha256": {"features.npz": hashlib.sha256((OUT / "features.npz").read_bytes()).hexdigest()},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("test_top_probability_gap", "seconds_total")}))
