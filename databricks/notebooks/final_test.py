# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: the single look at test (selected configuration, 3 seeds)
# MAGIC Scores the four clients' pooled test split once, for each seed's checkpoint (refusing a
# MAGIC weight-hash mismatch) and for the 3-seed ensemble (mean of class probabilities):
# MAGIC per-client and Barcelona bordered/non-bordered scores, mean and range over seeds,
# MAGIC bootstrap 95% intervals by lesion, and calibration (ECE, NLL) before and after a
# MAGIC temperature fitted on validation. Lesion-level averaging is reported as a separately
# MAGIC labelled supplementary row: it was not selected by the tuning ladder.
# MAGIC
# MAGIC A marker file in the volume records that test was looked at; a second run stops
# MAGIC unless `allow_second_look=true`, which is then recorded in the output.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models": "[]",
    "bootstrap": "1000",
    "allow_second_look": "false",
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
from cdm.eval import classification_metrics, summarize
from cdm.final import bootstrap_by_lesion, calibration, ensemble_logits
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.posthoc import lesion_average
from cdm.report import score_rows, support
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
MODELS = json.loads(dbutils.widgets.get("models"))
N_BOOT = int(dbutils.widgets.get("bootstrap"))
SECOND_LOOK = dbutils.widgets.get("allow_second_look").lower() == "true"
MARKER = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2/test_looked_at.json")
KEYS = ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity", "accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 0  # serverless CPU has too little shared memory for DataLoader workers

if MARKER.exists() and not SECOND_LOOK:
    raise RuntimeError(f"test was already scored: {MARKER.read_text()}")
result: dict[str, object] = {
    "code_version": dbutils.widgets.get("code_version"),
    "config": CONFIG,
    "models": MODELS,
    "device": str(device),
    "previous_look": MARKER.read_text() if MARKER.exists() else None,
}

# COMMAND ----------

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
result["support"] = support(rows)
index = {c: i for i, c in enumerate(CLASSES)}
frames = {s: rows[rows["split"] == s].reset_index(drop=True) for s in ("val", "test")}
lesions = frames["test"]["lesion_id"].fillna(frames["test"]["isic_id"]).tolist()
result["test_images"], result["test_lesions"] = len(frames["test"]), len(set(lesions))


def logits_for(model: torch.nn.Module, split: str) -> tuple[np.ndarray, np.ndarray]:
    frame = frames[split]
    data = EncodedImages(
        list(frame["image"]),
        frame["label"].map(index).tolist(),
        eval_transform(CONFIG.get("image_size", 224), CONFIG.get("color_constancy", False)),
    )
    _, logits, labels = extract(model, DataLoader(data, batch_size=64, num_workers=WORKERS), device)
    return logits, labels


# Validation logits first (temperature fitting needs them); the marker is written just
# before the first test image is scored.
val_logits, test_logits = [], []
models = []
for m in MODELS:
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"seed {m['seed']} checkpoint mismatch"
    models.append(model.to(device).eval())
    logits, val_labels = logits_for(models[-1], "val")
    val_logits.append(logits)

MARKER.parent.mkdir(parents=True, exist_ok=True)
MARKER.write_text(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"),
                              "code_version": result["code_version"]}))  # fmt: skip
for model in models:
    logits, test_labels = logits_for(model, "test")
    test_logits.append(logits)

# COMMAND ----------


def report(val: np.ndarray, test: np.ndarray) -> dict[str, object]:
    return {
        "val_pooled": {k: classification_metrics(val_labels, val)[k] for k in KEYS},
        "scores": score_rows(frames["test"], test, test_labels)["test"],
        "bootstrap_95": bootstrap_by_lesion(test, test_labels, lesions, n=N_BOOT),
        "calibration": calibration(val, val_labels, test, test_labels),
    }


per_seed = []
for m, val, test in zip(MODELS, val_logits, test_logits, strict=True):
    per_seed.append({"seed": m["seed"], **report(val, test)})
result["single_model_per_seed"] = per_seed


def pooled(entry: dict[str, object], key: str) -> float:
    return entry["scores"]["pooled"][key]


result["single_model_over_seeds"] = {
    **{k: summarize([pooled(e, k) for e in per_seed]) for k in KEYS},
    **{
        f"calibration_{k}": summarize([e["calibration"][k] for e in per_seed])
        for k in ("ece_before", "ece_after", "nll_before", "nll_after")
    },
}
result["ensemble_3_seeds"] = report(ensemble_logits(val_logits), ensemble_logits(test_logits))

# Supplementary, not selected: lesion averaging failed the keep rule on validation.
avg_seeds = [lesion_average(t, lesions) for t in test_logits]
avg_ens = lesion_average(ensemble_logits(test_logits), lesions)
result["supplementary_lesion_average_not_selected"] = {
    "note": "Not the selected configuration: lesion averaging missed the keep rule on "
    "validation (+0.83 points balanced accuracy). Shown for reference only.",
    "single_model_per_seed": [
        {"seed": m["seed"], **{k: classification_metrics(test_labels, a)[k] for k in KEYS}}
        for m, a in zip(MODELS, avg_seeds, strict=True)
    ],
    "single_model_over_seeds": {
        k: summarize([classification_metrics(test_labels, a)[k] for a in avg_seeds]) for k in KEYS
    },
    "ensemble_3_seeds": {
        **{k: classification_metrics(test_labels, avg_ens)[k] for k in KEYS},
        "bootstrap_95": bootstrap_by_lesion(avg_ens, test_labels, lesions, n=N_BOOT),
    },
}

print("FINAL_TEST_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
