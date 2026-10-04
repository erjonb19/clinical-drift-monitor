# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 triage operating point (triage framing, not a clinical claim), CPU only
# MAGIC Two separate runs, so the threshold is committed to git before any test image is scored.
# MAGIC `stage=threshold`: on v2 validation only, the highest melanoma-probability threshold that
# MAGIC refers at least 95% of validation melanomas, for the 3-seed ensemble (the shipped model,
# MAGIC uncalibrated) and for each single model. `stage=test`: refuses unless the thresholds passed
# MAGIC in equal the saved ones; then, once per model (lock: started, then completed), melanoma
# MAGIC sensitivity, specificity and referral rate on the clients' test split, per client, and on
# MAGIC Buenos Aires and PAD-UFES-20 (labelled images), with bootstrap 95% intervals by lesion.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "stage": "threshold",
    "thresholds": "{}",
    "target_sensitivity": "0.95",
    "bootstrap": "1000",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")], check=True
)

# COMMAND ----------

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.eval import softmax
from cdm.ood import extract
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash
from cdm.triage import bootstrap_triage, threshold_for_sensitivity, triage_metrics

CONFIG = json.loads(dbutils.widgets.get("config"))
STAGE = dbutils.widgets.get("stage")
CODE = dbutils.widgets.get("code_version")
TARGET = float(dbutils.widgets.get("target_sensitivity"))
N_BOOT = int(dbutils.widgets.get("bootstrap"))
MODELS = sorted(json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
                key=lambda m: m["seed"])  # fmt: skip
assert [m["seed"] for m in MODELS] == [0, 1, 2], MODELS
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/triage")
LOCKS = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2/triage_looks")
OUT.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)
MEL = CLASSES.index("mel")
INDEX = {c: i for i, c in enumerate(CLASSES)}
device = torch.device("cpu")
KEYS = ["v2-ensemble-final-seeds0-1-2", *(f"v2-final-seed{m['seed']}" for m in MODELS)]

# COMMAND ----------

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("label IS NOT NULL")
    .select("isic_id", "source", "role", "client", "split", "label", "lesion_id", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
rows["lesion"] = rows["lesion_id"].fillna(rows["isic_id"])
models = []
for m in MODELS:
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"{m} checkpoint mismatch"
    models.append(model.eval())


def p_mel(frame) -> dict[str, np.ndarray]:
    """Melanoma probability per model key (ensemble = mean of the 3 seeds' probabilities)."""
    data = EncodedImages(list(frame["image"]), frame["label"].map(INDEX).tolist(),
                         eval_transform(CONFIG.get("image_size", 224)))  # fmt: skip
    probs = []
    for model in models:
        _, logits, _ = extract(model, DataLoader(data, batch_size=64, num_workers=0), device)
        probs.append(softmax(logits))
    out = {f"v2-final-seed{m['seed']}": p[:, MEL] for m, p in zip(MODELS, probs, strict=True)}
    out["v2-ensemble-final-seeds0-1-2"] = np.mean(probs, axis=0)[:, MEL]
    return out


# COMMAND ----------

if STAGE == "threshold":
    val = rows[(rows["role"] == "client") & (rows["split"] == "val")].reset_index(drop=True)
    is_mel = (val["label"] == "mel").to_numpy()
    probs = p_mel(val)
    result = {
        "stage": "threshold", "code_version": CODE, "target_sensitivity": TARGET,
        "set_on": "v2 validation (all four clients)",
        "validation_images": len(val), "validation_melanoma_images": int(is_mel.sum()),
        "validation_melanoma_lesions": int(val.loc[is_mel, "lesion"].nunique()),
        "thresholds": {k: threshold_for_sensitivity(probs[k], is_mel, TARGET) for k in KEYS},
    }  # fmt: skip
    result["validation_at_threshold"] = {
        k: triage_metrics(probs[k], is_mel, result["thresholds"][k]) for k in KEYS
    }
    (OUT / "threshold.json").write_text(json.dumps(result, indent=2))
    print("TRIAGE_THRESHOLD " + json.dumps(result), flush=True)
    dbutils.notebook.exit(json.dumps(result))

# COMMAND ----------

# Test stage: the thresholds must be exactly the committed ones.
saved = json.loads((OUT / "threshold.json").read_text())
given = json.loads(dbutils.widgets.get("thresholds"))
assert given == saved["thresholds"], f"thresholds {given} differ from saved {saved['thresholds']}"
for k in KEYS:
    lock = LOCKS / f"{k}.json"
    if lock.exists():
        raise RuntimeError(f"{k} already had its triage look: {lock.read_text()}")
looks = {}
for k in KEYS:
    looks[k] = {"status": "started", "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "code_version": CODE}  # fmt: skip
    (LOCKS / f"{k}.json").write_text(json.dumps(looks[k]))

scored = rows[
    ((rows["role"] == "client") & (rows["split"] == "test"))
    | rows["source"].isin(["buenos_aires", "pad_ufes"])
].reset_index(drop=True)
is_test = ((scored["role"] == "client") & (scored["split"] == "test")).to_numpy()
groups = {"client_test_pooled": is_test}
for c in sorted(scored.loc[is_test, "client"].unique()):
    groups[f"client_test_{c}"] = is_test & (scored["client"] == c).to_numpy()
groups["external_buenos_aires"] = (scored["source"] == "buenos_aires").to_numpy()
groups["external_pad_ufes"] = (scored["source"] == "pad_ufes").to_numpy()
probs = p_mel(scored)
is_mel = (scored["label"] == "mel").to_numpy()
lesions = scored["lesion"].tolist()
result = {"stage": "test", "code_version": CODE, "target_sensitivity": TARGET,
          "thresholds": saved["thresholds"],
          "threshold_set_on": {k: saved[k] for k in ("set_on", "validation_images",
                                                   "validation_melanoma_images",
                                                   "validation_melanoma_lesions")},
          "framing": "triage operating point, not a clinical claim", "models": {}}  # fmt: skip
for k in KEYS:
    t = saved["thresholds"][k]
    result["models"][k] = {
        name: {**triage_metrics(probs[k][mask], is_mel[mask], t),
               "bootstrap_95": bootstrap_triage(probs[k][mask], is_mel[mask],
                                                [lesions[i] for i in np.flatnonzero(mask)], t,
                                                n=N_BOOT)}
        for name, mask in groups.items()
    }  # fmt: skip
(OUT / "test.json").write_text(json.dumps(result, indent=2))
assert json.loads((OUT / "test.json").read_text())["models"].keys() == set(KEYS)
for k in KEYS:  # completed only once the result is saved and reads back
    (LOCKS / f"{k}.json").write_text(json.dumps({**looks[k], "status": "completed",
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds")}))  # fmt: skip
print("TRIAGE_TEST " + json.dumps(result), flush=True)
dbutils.notebook.exit(json.dumps(result))
