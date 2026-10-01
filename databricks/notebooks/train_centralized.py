# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: centralized training on serverless GPU (one tuning rung per run)
# MAGIC Trains on the four clients' pooled training splits from `workspace.cdm.silver_images`
# MAGIC with the settings in the widgets, scores **validation only** (pooled, per client,
# MAGIC HAM10000 both institutions, Barcelona bordered and non-bordered), logs to MLflow and
# MAGIC saves the best-validation weights. Test is scored only with `score_test=true`, which
# MAGIC the tuning ladder uses once, at the end. Returns the result as the exit value (JSON).

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "rung": "",
    "seed": "0",
    "epochs": "12",
    "arch": "b0",
    "image_size": "224",
    "rotate": "false",
    "color_jitter": "false",
    "warmup_epochs": "0",
    "label_smoothing": "0.0",
    "determinism_check": "false",
    "score_test": "false",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

# Install the pinned cdm wheel. Widget substitution ($wheel) does not work in %pip here
# (run 911992146364689 passed "$wheel" to pip literally), so install from Python.
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")], check=True
)

# COMMAND ----------

import json
import os
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.report import score_rows, support
from cdm.silver import EncodedImages, split_datasets
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash


def flag(name: str) -> bool:
    return dbutils.widgets.get(name).lower() == "true"


RUNG = dbutils.widgets.get("rung")
SEED = int(dbutils.widgets.get("seed"))
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
ARCH = dbutils.widgets.get("arch")
CFG = TrainConfig(
    epochs=int(dbutils.widgets.get("epochs")),
    batch_size=32 if ARCH == "b3" else 64,
    num_workers=WORKERS,
    arch=ARCH,
    image_size=int(dbutils.widgets.get("image_size")),
    rotate=flag("rotate"),
    color_jitter=flag("color_jitter"),
    warmup_epochs=int(dbutils.widgets.get("warmup_epochs")),
    label_smoothing=float(dbutils.widgets.get("label_smoothing")),
)
device = torch.device("cuda")
user = spark.sql("SELECT current_user()").first()[0]
result: dict[str, object] = {
    "rung": RUNG,
    "code_version": dbutils.widgets.get("code_version"),
    "seed": SEED,
    "config": {k: getattr(CFG, k) for k in CFG.__dataclass_fields__},
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
}

# COMMAND ----------

start = time.perf_counter()
table = spark.read.table("workspace.cdm.silver_images")
image_col = "image" if "image" in table.columns else "png"  # silver used "png" before rung 1a
rows = (
    table.where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", image_col)
    .withColumnRenamed(image_col, "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
sets = split_datasets(rows, CFG)
result["load_seconds"] = round(time.perf_counter() - start)
result["images"] = {s: len(d) for s, d in sets.items()}
result["support"] = support(rows)
print(json.dumps({k: result[k] for k in ("rung", "config", "images")}))

# COMMAND ----------


def run(cfg: TrainConfig, seed: int) -> tuple[torch.nn.Module, list[dict[str, float]], float]:
    generator = seed_everything(seed, cfg.num_threads, gpu=True)
    model = build_model(pretrained=True, arch=cfg.arch)
    t0 = time.perf_counter()
    model, history = train(model, sets["train"], sets["val"], cfg, seed, device, generator)
    return model, history, time.perf_counter() - t0


if flag("determinism_check"):
    one = TrainConfig(**{**result["config"], "epochs": 1})
    a, _, _ = run(one, SEED)
    b, _, _ = run(one, SEED)
    result["determinism_one_epoch"] = {"identical": weights_hash(a) == weights_hash(b)}

# COMMAND ----------

scored_splits = ["val", "test"] if flag("score_test") else ["val"]
mlflow.set_experiment(f"/Users/{user}/cdm-phase2")
with mlflow.start_run(run_name=f"{RUNG}-seed{SEED}") as mlrun:
    mlflow.log_params({"rung": RUNG, "seed": SEED, **result["config"]})
    model, history, seconds = run(CFG, SEED)
    for h in history:
        mlflow.log_metrics(
            {"train_loss": h["train_loss"], "val_bacc": h["val_bacc"]}, step=int(h["epoch"])
        )

    # One evaluation dataset built from exactly the scored rows: predictions line up with rows.
    scored = rows[rows["split"].isin(scored_splits)].reset_index(drop=True)
    index = {c: i for i, c in enumerate(CLASSES)}
    data = EncodedImages(
        list(scored["image"]), scored["label"].map(index).tolist(), eval_transform(CFG.image_size)
    )
    _, logits, labels = extract(
        model, DataLoader(data, batch_size=128, num_workers=WORKERS), device
    )
    scores = score_rows(scored, logits, labels)
    pooled_val = scores["val"]["pooled"]
    mlflow.log_metrics(
        {
            "val_bacc_pooled": pooled_val["balanced_accuracy"],
            "val_macro_auroc_pooled": pooled_val["macro_auroc"],
            "val_melanoma_sensitivity_pooled": pooled_val["melanoma_sensitivity"],
        }
    )

    folder = Path(f"/Volumes/workspace/cdm/raw/checkpoints/phase2/{RUNG}-seed{SEED}")
    folder.mkdir(parents=True, exist_ok=True)
    local = Path("/tmp/model.pt")
    torch.save(model.state_dict(), local)
    (folder / "model.pt").write_bytes(local.read_bytes())
    result.update(
        {
            "mlflow_run_id": mlrun.info.run_id,
            "train_seconds": round(seconds),
            "train_images_per_s": round(len(sets["train"]) * CFG.epochs / seconds),
            "best_epoch": int(max(history, key=lambda h: h["val_bacc"])["epoch"]),
            "history": history,
            "scored_splits": scored_splits,
            "scores": scores,
            "checkpoint": str(folder / "model.pt"),
            "weights_sha256": weights_hash(model),
        }
    )

print("TRAIN_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
