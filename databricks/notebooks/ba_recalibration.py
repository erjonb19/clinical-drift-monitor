# Databricks notebook source
# MAGIC %md
# MAGIC # Simulated local recalibration at Buenos Aires (CPU), PCCP evidence only
# MAGIC Re-scores Buenos Aires' labelled images with the shipped ensemble exactly as the triage
# MAGIC test run did, stops unless that reproduces the triage run's Buenos Aires numbers at the
# MAGIC committed threshold, then runs 20 seeded 50/50 lesion splits: the 95%-sensitivity
# MAGIC threshold is set on one half and reported on the other, next to the shipped threshold.
# MAGIC Not a change to the shipped model or threshold. Spec:
# MAGIC `results/phase2/ba_recalibration/spec.json`.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "triage": "/Volumes/workspace/cdm/raw/results/phase2/triage",
    "splits": "20",
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
from cdm.triage import local_recalibration, triage_metrics

CONFIG = json.loads(dbutils.widgets.get("config"))
TRIAGE = Path(dbutils.widgets.get("triage"))
KEY = "v2-ensemble-final-seeds0-1-2"
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/ba_recalibration")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = sorted(json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
                key=lambda m: m["seed"])  # fmt: skip
MEL, INDEX = CLASSES.index("mel"), {c: i for i, c in enumerate(CLASSES)}
threshold = json.loads((TRIAGE / "threshold.json").read_text())["thresholds"][KEY]
expected = json.loads((TRIAGE / "test.json").read_text())["models"][KEY]["external_buenos_aires"]

# Same rows, order and preprocessing as the triage notebook's scored frame (Buenos Aires part).
ba = (
    spark.read.table("workspace.cdm.silver_images")
    .where("label IS NOT NULL AND source = 'buenos_aires'")
    .select("isic_id", "label", "lesion_id", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
ba["lesion"] = ba["lesion_id"].fillna(ba["isic_id"])
data = EncodedImages(list(ba["image"]), ba["label"].map(INDEX).tolist(),
                     eval_transform(CONFIG.get("image_size", 224)))  # fmt: skip
probs = []
for m in MODELS:
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], m
    _, logits, _ = extract(model.eval(), DataLoader(data, batch_size=64, num_workers=0),
                           torch.device("cpu"))  # fmt: skip
    probs.append(softmax(logits))
p_mel = np.mean(probs, axis=0)[:, MEL]
is_mel = (ba["label"] == "mel").to_numpy()

got = triage_metrics(p_mel, is_mel, threshold)
check = {k: [got[k], expected[k]] for k in ("images", "melanoma_images", "sensitivity",
                                             "specificity", "referral_rate")}  # fmt: skip
reproduced = all(abs(a - b) <= 1e-12 for a, b in check.values())
(OUT / "reproduction_check.json").write_text(
    json.dumps({"reproduced": reproduced, "values": check})
)
if not reproduced:
    raise RuntimeError(
        f"re-scoring did not reproduce the triage run's Buenos Aires numbers: {check}"
    )

splits = local_recalibration(p_mel, is_mel, ba["lesion"].tolist(), threshold,
                             seeds=range(int(dbutils.widgets.get("splits"))),
                             n_boot=int(dbutils.widgets.get("bootstrap")))  # fmt: skip


def spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


summary = {
    which: {k: spread([s[which][k] for s in splits]) for k in ("sensitivity", "specificity",
                                                             "referral_rate")}
    for which in ("local", "shipped")
}  # fmt: skip
summary["local_threshold"] = spread([s["local_threshold"] for s in splits])
result = {"code_version": dbutils.widgets.get("code_version"), "shipped_threshold": threshold,
          "reproduction_check": check, "images": len(ba), "melanoma_images": int(is_mel.sum()),
          "lesions": int(ba["lesion"].nunique()), "splits": splits, "over_splits": summary,
          "framing": "PCCP evidence for site-level recalibration; not a change to the shipped "
                     "model or threshold; triage framing, not a clinical claim"}  # fmt: skip
(OUT / "results.json").write_text(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps({"reproduced": reproduced, "over_splits": summary}))
