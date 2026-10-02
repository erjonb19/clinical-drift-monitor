# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 tuning ladder, rungs 4 to 6: post-training changes (no training)
# MAGIC Loads the reference checkpoint (refusing a weight-hash mismatch), scores **validation
# MAGIC only**, and tests three single changes against the reference's plain predictions with
# MAGIC `cdm.ladder.passes`: 4 test-time flip averaging, 5 per-class thresholds (cross-fitted
# MAGIC on lesion halves of validation), 6 lesion-level averaging. If two or more are kept, the
# MAGIC kept changes are combined (flips, then lesion averaging, then thresholds) and the
# MAGIC combination is kept only if it passes against the best single change. Runs on CPU.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "batch": "posthoc",
    "seed": "0",
    "reference": "{}",
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
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.images import corner_brightness
from cdm.ladder import combination_needed, deltas, passes
from cdm.ood import extract, logits_with_flips
from cdm.posthoc import cross_fit_offsets, fit_offsets, lesion_average
from cdm.report import score_rows
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

SEED = int(dbutils.widgets.get("seed"))
REFERENCE = json.loads(dbutils.widgets.get("reference"))
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = REFERENCE["config"]
model = build_model(pretrained=False, arch=cfg.get("arch", "b0"))
model.load_state_dict(torch.load(REFERENCE["checkpoint"], map_location="cpu"))
assert weights_hash(model) == REFERENCE["weights_sha256"], "reference checkpoint mismatch"
model.to(device).eval()
result: dict[str, object] = {
    "batch": dbutils.widgets.get("batch"),
    "code_version": dbutils.widgets.get("code_version"),
    "seed": SEED,
    "reference": REFERENCE,
    "device": str(device),
}

# COMMAND ----------

val = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split = 'val' AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
val["bordered"] = [corner_brightness(b) < 20 for b in val["image"]]
# An image without a lesion ID is its own lesion.
lesions = val["lesion_id"].fillna(val["isic_id"]).tolist()
result["val_images"], result["val_lesions"] = len(val), len(set(lesions))
data = EncodedImages(
    list(val["image"]),
    val["label"].map({c: i for i, c in enumerate(CLASSES)}).tolist(),
    eval_transform(cfg.get("image_size", 224), cfg.get("color_constancy", False)),
)
loader = DataLoader(data, batch_size=64, num_workers=WORKERS)
_, plain, labels = extract(model, loader, device)
flipped, _ = logits_with_flips(model, loader, device)

# COMMAND ----------

KEYS = ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity")


def make(changes: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    """Logits for a set of rungs, applied as flips, then lesion averaging, then thresholds."""
    extra: dict[str, object] = {}
    logits = flipped if "4-tta" in changes else plain
    if "6-lesion-average" in changes:
        logits = lesion_average(logits, lesions)
    if "5-thresholds" in changes:
        cross, halves = cross_fit_offsets(logits, labels, lesions, seed=SEED)
        extra["offsets_by_half"] = halves
        # For the single look at test: offsets fitted on all of validation.
        extra["offsets_all_val"] = [float(v) for v in fit_offsets(logits, labels)]
        logits = cross
    return logits, extra


def scored(name: str, changes: list[str]) -> dict[str, object]:
    logits, extra = make(changes)
    scores = score_rows(val, logits, labels)
    metrics = {k: scores["val"]["pooled"][k] for k in KEYS}
    print(json.dumps({"rung": name, "metrics": metrics}), flush=True)
    return {"rung": name, "changes": changes, "metrics": metrics, "scores": scores, **extra}


def judge(out: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    kept = passes(out["metrics"], reference["metrics"])
    return {"vs": reference["rung"], "deltas": deltas(out["metrics"], reference["metrics"]),
            "accuracy_rule": kept, "kept": kept}  # fmt: skip


# The reference is judged as recomputed here, on this hardware, so every rung differs from
# it only by the change; the recomputed numbers are checked against the training run's.
base = scored(REFERENCE["name"], [])
result["reference_recomputed"] = {
    "metrics": base["metrics"],
    "scores": base["scores"],
    "points_from_recorded": deltas(base["metrics"], REFERENCE["metrics"]),
}
assert all(abs(v) < 0.5 for v in result["reference_recomputed"]["points_from_recorded"].values()), (
    "recomputed reference differs from the recorded run by 0.5 points or more"
)

singles = []
for rung in ("4-tta", "5-thresholds", "6-lesion-average"):
    out = scored(rung, [rung])
    out["verdict"] = judge(out, base)
    singles.append(out)
result["singles"] = singles
kept = [s for s in singles if s["verdict"]["kept"]]
result["kept_singles"] = [s["rung"] for s in kept]
winner = {"name": REFERENCE["name"], "changes": [], "metrics": base["metrics"]}
if kept:
    best = max(kept, key=lambda s: s["metrics"]["balanced_accuracy"])
    winner = {"name": best["rung"], "changes": best["changes"], "metrics": best["metrics"]}
if combination_needed([s["rung"] for s in kept]):
    combo = scored("combined", [s["rung"] for s in kept])
    combo["verdict"] = judge(combo, best)
    result["combined"] = combo
    if combo["verdict"]["kept"]:
        winner = {"name": "combined", "changes": combo["changes"], "metrics": combo["metrics"]}
result["winner"] = winner

print("BATCH_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
