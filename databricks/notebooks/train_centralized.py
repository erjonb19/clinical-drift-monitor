# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: centralized training on serverless GPU
# MAGIC Trains EfficientNet-B0 on the four clients' pooled training splits from
# MAGIC `workspace.cdm.silver_images`, evaluates on the pooled test set (all four clients' test
# MAGIC splits) and per client, logs to MLflow and saves the weights to the volume. With
# MAGIC `determinism_check`, it first trains one epoch twice with the same seed and compares
# MAGIC the weights. The result is returned as the notebook's exit value (JSON).

# COMMAND ----------

dbutils.widgets.text("wheel", "")
dbutils.widgets.text("seed", "0")
dbutils.widgets.text("epochs", "12")
dbutils.widgets.text("determinism_check", "true")
dbutils.widgets.text("code_version", "")

# COMMAND ----------

# Install the pinned cdm wheel. Widget substitution ($wheel) does not work in %pip here
# (the first run passed "$wheel" to pip literally), so install from Python.
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")],
    check=True,
)

# COMMAND ----------

import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.eval import classification_metrics
from cdm.ood import extract
from cdm.silver import split_datasets
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash

SEED = int(dbutils.widgets.get("seed"))
EPOCHS = int(dbutils.widgets.get("epochs"))
CHECK = dbutils.widgets.get("determinism_check").lower() == "true"
CODE = dbutils.widgets.get("code_version")
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
device = torch.device("cuda")
user = spark.sql("SELECT current_user()").first()[0]
result: dict[str, object] = {
    "code_version": CODE,
    "seed": SEED,
    "epochs": EPOCHS,
    "gpu": torch.cuda.get_device_name(0),
    "cpu_count": os.cpu_count(),
    "num_workers": WORKERS,
    "torch": torch.__version__,
}

# COMMAND ----------

start = time.perf_counter()
rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "png")
    .orderBy("client", "isic_id")
    .toPandas()
)
sets = split_datasets(rows)
result["load_seconds"] = round(time.perf_counter() - start)
result["images"] = {s: len(d) for s, d in sets.items()}
result["images_by_client_and_split"] = rows.groupby(["client", "split"]).size().to_dict()
result["images_by_client_and_split"] = {
    f"{c}/{s}": int(n) for (c, s), n in result["images_by_client_and_split"].items()
}
print(json.dumps(result))

# COMMAND ----------


def run(epochs: int, seed: int) -> tuple[torch.nn.Module, list[dict[str, float]], float]:
    cfg = TrainConfig(epochs=epochs, num_workers=WORKERS)
    generator = seed_everything(seed, cfg.num_threads, gpu=True)
    model = build_model(pretrained=True)
    t0 = time.perf_counter()
    model, history = train(model, sets["train"], sets["val"], cfg, seed, device, generator)
    return model, history, time.perf_counter() - t0


if CHECK:
    a, _, seconds = run(1, SEED)
    b, _, _ = run(1, SEED)
    result["determinism_one_epoch"] = {
        "same_seed_identical_weights": weights_hash(a) == weights_hash(b),
        "weights": [weights_hash(a), weights_hash(b)],
        "seconds_per_epoch": round(seconds),
    }
    print(json.dumps(result["determinism_one_epoch"]))

# COMMAND ----------

mlflow.set_experiment(f"/Users/{user}/cdm-phase2")
with mlflow.start_run(run_name=f"centralized-seed{SEED}") as mlrun:
    mlflow.log_params(
        {
            "mode": "centralized",
            "seed": SEED,
            "epochs": EPOCHS,
            "code_version": CODE,
            "train_images": len(sets["train"]),
        }
    )
    model, history, seconds = run(EPOCHS, SEED)
    for h in history:
        mlflow.log_metrics(
            {"train_loss": h["train_loss"], "val_bacc": h["val_bacc"]}, step=int(h["epoch"])
        )

    test = rows[rows["split"] == "test"].reset_index(drop=True)
    loader = DataLoader(sets["test"], batch_size=128, num_workers=WORKERS)
    _, logits, labels = extract(model, loader, device)
    pooled = classification_metrics(labels, logits)
    per_client = {}
    for client, idx in test.groupby("client").groups.items():
        idx = np.asarray(list(idx))
        per_client[client] = classification_metrics(labels[idx], logits[idx])["balanced_accuracy"]
    mlflow.log_metrics(
        {
            "test_bacc_pooled": pooled["balanced_accuracy"],
            **{f"test_bacc_{c}": v for c, v in per_client.items()},
        }
    )

    folder = Path(f"/Volumes/workspace/cdm/raw/checkpoints/phase2/centralized-seed{SEED}")
    folder.mkdir(parents=True, exist_ok=True)
    local = Path("/tmp/model.pt")
    torch.save(model.state_dict(), local)
    (folder / "model.pt").write_bytes(local.read_bytes())
    result.update(
        {
            "mlflow_run_id": mlrun.info.run_id,
            "train_seconds": round(seconds),
            "train_images_per_s": round(len(sets["train"]) * EPOCHS / seconds),
            "best_epoch": int(max(history, key=lambda h: h["val_bacc"])["epoch"]),
            "history": history,
            "test_pooled": pooled,
            "test_bacc_by_client": per_client,
            "checkpoint": str(folder / "model.pt"),
            "weights_sha256": weights_hash(model),
        }
    )

print("TRAIN_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
