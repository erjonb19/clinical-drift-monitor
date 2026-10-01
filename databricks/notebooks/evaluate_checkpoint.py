# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: score a saved checkpoint (no training)
# MAGIC Loads a checkpoint, refuses to continue unless its weight hash matches, and scores the
# MAGIC requested splits (validation only by default) with the shared Phase 2 report: pooled,
# MAGIC per client, HAM10000 both institutions, Barcelona bordered and non-bordered, confusion
# MAGIC matrices, macro AUROC and melanoma sensitivity. With `tta=true`, logits are averaged over
# MAGIC flips (tuning rung 4). Also reports per-class support and, for every site, how many
# MAGIC images are bordered (dark corners), including the held-out sites.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "rung": "",
    "checkpoint": "",
    "expected_weights": "",
    "arch": "b0",
    "image_size": "224",
    "tta": "false",
    "splits": "val",
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

import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.images import corner_brightness
from cdm.ood import extract, logits_with_flips
from cdm.report import score_rows, support
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
SIZE = int(dbutils.widgets.get("image_size"))
TTA = dbutils.widgets.get("tta").lower() == "true"
SPLITS = [s.strip() for s in dbutils.widgets.get("splits").split(",")]
model = build_model(pretrained=False, arch=dbutils.widgets.get("arch"))
model.load_state_dict(torch.load(dbutils.widgets.get("checkpoint"), map_location="cpu"))
model.to(device).eval()
got, expected = weights_hash(model), dbutils.widgets.get("expected_weights")
assert got == expected, f"checkpoint weights {got}, expected {expected}"
result: dict[str, object] = {
    "rung": dbutils.widgets.get("rung"),
    "code_version": dbutils.widgets.get("code_version"),
    "checkpoint": dbutils.widgets.get("checkpoint"),
    "weights_sha256": got,
    "arch": dbutils.widgets.get("arch"),
    "image_size": SIZE,
    "tta": TTA,
    "scored_splits": SPLITS,
}

# COMMAND ----------

table = spark.read.table("workspace.cdm.silver_images")
image_col = "image" if "image" in table.columns else "png"  # silver used "png" before rung 1a
everything = (
    table.select("isic_id", "source", "role", "client", "split", "label", "lesion_id", image_col)
    .withColumnRenamed(image_col, "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
everything["bordered"] = [corner_brightness(b) < 20 for b in everything["image"]]
result["bordered_images_by_source"] = {
    s: {"images": len(g), "bordered": int(g["bordered"].sum()),
        "share": round(float(g["bordered"].mean()), 4)}
    for s, g in everything.groupby("source")
}  # fmt: skip

clients = everything[
    (everything["role"] == "client")
    & everything["split"].isin(["train", "val", "test"])
    & everything["label"].notna()
].reset_index(drop=True)
result["support"] = support(clients)

scored = clients[clients["split"].isin(SPLITS)].reset_index(drop=True)
index = {c: i for i, c in enumerate(CLASSES)}
data = EncodedImages(
    list(scored["image"]), scored["label"].map(index).tolist(), eval_transform(SIZE)
)
loader = DataLoader(data, batch_size=128, num_workers=WORKERS)
if TTA:
    logits, labels = logits_with_flips(model, loader, device)
else:
    _, logits, labels = extract(model, loader, device)
result["scores"] = score_rows(scored, logits, labels)

print("EVAL_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
