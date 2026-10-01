# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: re-evaluate a saved checkpoint
# MAGIC No training. Loads a checkpoint, refuses to continue unless its weight hash matches,
# MAGIC and reports validation and test scores (pooled, per client, and HAM10000 over both
# MAGIC institutions), confusion matrices, and per-client data checks: class mix, original
# MAGIC resolution, corner brightness (image framing) and the ISIC diagnoses behind each label.

# COMMAND ----------

dbutils.widgets.text("wheel", "")
dbutils.widgets.text("checkpoint", "")
dbutils.widgets.text("expected_weights", "")
dbutils.widgets.text("code_version", "")

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
from cdm.eval import classification_metrics, confusion
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.silver import PngDataset
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
model = build_model(pretrained=False)
model.load_state_dict(torch.load(dbutils.widgets.get("checkpoint"), map_location="cpu"))
model.to(device).eval()
got = weights_hash(model)
expected = dbutils.widgets.get("expected_weights")
assert got == expected, f"checkpoint weights {got}, expected {expected}"
result: dict[str, object] = {
    "code_version": dbutils.widgets.get("code_version"),
    "checkpoint": dbutils.widgets.get("checkpoint"),
    "weights_sha256": got,
}

# COMMAND ----------

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client'")
    .select("isic_id", "client", "split", "label", "diagnosis_3", "width", "height", "png")
    .orderBy("client", "isic_id")
    .toPandas()
)
labelled = rows[rows["split"].isin(["val", "test"]) & rows["label"].notna()].reset_index(drop=True)
index = {c: i for i, c in enumerate(CLASSES)}
dataset = PngDataset(list(labelled["png"]), labelled["label"].map(index).tolist(), eval_transform())
_, logits, labels = extract(model, DataLoader(dataset, batch_size=128, num_workers=WORKERS), device)


def scores(mask: np.ndarray) -> dict[str, object]:
    return {**classification_metrics(labels[mask], logits[mask]),
            "confusion": confusion(labels[mask], logits[mask])}  # fmt: skip


evaluation: dict[str, object] = {}
for split in ("val", "test"):
    in_split = (labelled["split"] == split).to_numpy()
    groups = {"pooled": in_split,
              "ham10000_both_institutions": in_split & labelled["client"].str.startswith("ham_").to_numpy()}  # fmt: skip
    for client in sorted(labelled["client"].unique()):
        groups[client] = in_split & (labelled["client"] == client).to_numpy()
    evaluation[split] = {name: scores(mask) for name, mask in groups.items()}
result["evaluation"] = evaluation
result["class_order"] = list(CLASSES)

# COMMAND ----------

trainable = rows[rows["label"].notna() & rows["split"].isin(["train", "val", "test"])]
result["class_mix"] = {
    f"{c}/{s}": g["label"].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()
    for (c, s), g in trainable.groupby(["client", "split"])
}
result["original_resolution"] = {
    c: {"width": g["width"].quantile([0, 0.5, 1]).astype(int).tolist(),
        "height": g["height"].quantile([0, 0.5, 1]).astype(int).tolist()}
    for c, g in rows.groupby("client")
}  # fmt: skip
rows["corner"] = [corner_brightness(p) for p in rows["png"]]
result["corner_brightness"] = {
    c: {
        "median": float(g["corner"].median()),
        "share_below_20": round(float((g["corner"] < 20).mean()), 4),
    }
    for c, g in rows.groupby("client")
}
result["diagnoses_behind_labels"] = {
    c: {
        f"{lab} <- {d3}": int(n)
        for (lab, d3), n in g.groupby(["label", "diagnosis_3"], dropna=False).size().items()
    }
    for c, g in trainable.groupby("client")
}

print("EVAL_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
