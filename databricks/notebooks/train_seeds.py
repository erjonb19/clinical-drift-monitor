# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: train the selected configuration with more seeds (one GPU job)
# MAGIC Trains the tuning ladder's winning configuration once per seed, sequentially on one
# MAGIC GPU, scores **validation only**, and saves each checkpoint to
# MAGIC `checkpoints/phase2/final-seed{N}`. Seed 0 is the ladder's own run and is not retrained.
# MAGIC These are the extra seeds for the 3-seed mean, range and ensemble; nothing is judged.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "seeds": "1,2",
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
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.images import corner_brightness
from cdm.ood import extract
from cdm.report import score_rows
from cdm.silver import EncodedImages, split_datasets
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
SEEDS = [int(s) for s in dbutils.widgets.get("seeds").split(",")]
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
device = torch.device("cuda")
cfg = TrainConfig(**{"num_workers": WORKERS, **CONFIG})
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/cdm-phase2")

rows = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split IN ('train', 'val', 'test') AND label IS NOT NULL")
    .select("isic_id", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
rows["bordered"] = [corner_brightness(b) < 20 for b in rows["image"]]
sets = split_datasets(rows, cfg)
val = rows[rows["split"] == "val"].reset_index(drop=True)
result: dict[str, object] = {
    "code_version": dbutils.widgets.get("code_version"),
    "config": CONFIG,
    "seeds": SEEDS,
    "gpu": torch.cuda.get_device_name(0),
    "runs": [],
}

# COMMAND ----------

for seed in SEEDS:
    with mlflow.start_run(run_name=f"final-seed{seed}") as run:
        mlflow.log_params({"rung": "final", "seed": seed, **CONFIG})
        generator = seed_everything(seed, cfg.num_threads, gpu=True)
        model = build_model(pretrained=True, arch=cfg.arch)
        t0 = time.perf_counter()
        model, history = train(model, sets["train"], sets["val"], cfg, seed, device, generator)
        seconds = time.perf_counter() - t0
        data = EncodedImages(
            list(val["image"]),
            val["label"].map({c: i for i, c in enumerate(CLASSES)}).tolist(),
            eval_transform(cfg.image_size, cfg.color_constancy),
        )
        _, logits, labels = extract(
            model, DataLoader(data, batch_size=128, num_workers=WORKERS), device
        )
        scores = score_rows(val, logits, labels)
        pooled = scores["val"]["pooled"]
        mlflow.log_metrics({"val_bacc_pooled": pooled["balanced_accuracy"]})
        folder = Path(f"/Volumes/workspace/cdm/raw/checkpoints/phase2/final-seed{seed}")
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "/tmp/model.pt")
        (folder / "model.pt").write_bytes(Path("/tmp/model.pt").read_bytes())
        out = {
            "seed": seed,
            "metrics": {k: pooled[k] for k in ("balanced_accuracy", "macro_auroc",
                                                "melanoma_sensitivity")},
            "scores": scores,
            "best_epoch": int(max(history, key=lambda h: h["val_bacc"])["epoch"]),
            "history": history,
            "train_seconds": round(seconds),
            "checkpoint": str(folder / "model.pt"),
            "weights_sha256": weights_hash(model),
            "mlflow_run_id": run.info.run_id,
        }  # fmt: skip
    result["runs"].append(out)
    print(json.dumps({"seed": seed, "metrics": out["metrics"]}), flush=True)

print("SEEDS_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
