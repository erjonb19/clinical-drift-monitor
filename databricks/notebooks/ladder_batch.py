# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 tuning ladder: one batch of single changes, then their combination
# MAGIC One job on one GPU (fewer waits for GPU capacity). Each candidate is one change applied
# MAGIC to the reference configuration, scored on **validation only** and judged by
# MAGIC `cdm.ladder.passes` against the reference. If two or more are kept, one model with all
# MAGIC kept changes is trained and kept only if it passes against the best single change. With
# MAGIC `drift=true`, held-out drift AUROCs are computed for the reference checkpoint and every
# MAGIC model, and colour-constancy changes must also pass `cdm.ladder.drift_preserved`.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "batch": "",
    "seed": "0",
    "reference": "{}",
    "candidates": "[]",
    "drift": "false",
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
from cdm.eval import auroc
from cdm.images import corner_brightness
from cdm.ladder import combination_needed, deltas, drift_preserved, merge, passes
from cdm.ood import MahalanobisDetector, energy_score, extract
from cdm.report import score_rows, support
from cdm.silver import EncodedImages, split_datasets
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, seed_everything, train, weights_hash

BATCH = dbutils.widgets.get("batch")
SEED = int(dbutils.widgets.get("seed"))
REFERENCE = json.loads(dbutils.widgets.get("reference"))
CANDIDATES = json.loads(dbutils.widgets.get("candidates"))
DRIFT = dbutils.widgets.get("drift").lower() == "true"
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
device = torch.device("cuda")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/cdm-phase2")
result: dict[str, object] = {
    "batch": BATCH,
    "code_version": dbutils.widgets.get("code_version"),
    "seed": SEED,
    "reference": REFERENCE,
    "candidates": CANDIDATES,
    "drift": DRIFT,
    "gpu": torch.cuda.get_device_name(0),
}

# COMMAND ----------

table = spark.read.table("workspace.cdm.silver_images")
everything = (
    table.select("isic_id", "source", "role", "client", "split", "label", "lesion_id", "image")
    .orderBy("client", "isic_id")
    .toPandas()
)
everything["bordered"] = [corner_brightness(b) < 20 for b in everything["image"]]
rows = everything[
    (everything["role"] == "client")
    & everything["split"].isin(["train", "val", "test"])
    & everything["label"].notna()
].reset_index(drop=True)
held_out = everything[everything["role"] == "held_out"].reset_index(drop=True)
val = rows[rows["split"] == "val"].reset_index(drop=True)
INDEX = {c: i for i, c in enumerate(CLASSES)}
result["support"] = support(rows)


def config(overrides: dict[str, object]) -> TrainConfig:
    batch = 32 if overrides.get("arch") == "b3" else 64
    return TrainConfig(**{"num_workers": WORKERS, "batch_size": batch, **overrides})


def predict(model: torch.nn.Module, frame, cfg: TrainConfig):
    labels = frame["label"].map(INDEX).fillna(-1).astype(int).tolist()
    data = EncodedImages(
        list(frame["image"]), labels, eval_transform(cfg.image_size, cfg.color_constancy)
    )
    return extract(model, DataLoader(data, batch_size=128, num_workers=WORKERS), device)


def drift_aurocs(model: torch.nn.Module, cfg: TrainConfig) -> dict[str, float]:
    """AUROC separating validation images from each held-out site, energy and Mahalanobis
    (PCA 256, fitted on training features only)."""
    train_feats, _, train_labels = predict(model, rows[rows["split"] == "train"], cfg)
    val_feats, val_logits, _ = predict(model, val, cfg)
    maha = MahalanobisDetector().fit(train_feats, train_labels)
    out = {}
    for site in sorted(held_out["source"].unique()):
        feats, logits, _ = predict(model, held_out[held_out["source"] == site], cfg)
        out[f"{site}/energy"] = auroc(energy_score(val_logits), energy_score(logits))
        out[f"{site}/mahalanobis"] = auroc(maha.score(val_feats), maha.score(feats))
    return out


def fit_and_score(name: str, overrides: dict[str, object]) -> dict[str, object]:
    cfg = config(overrides)
    sets = split_datasets(rows, cfg)
    with mlflow.start_run(run_name=f"{BATCH}-{name}-seed{SEED}") as run:
        mlflow.log_params({"batch": BATCH, "rung": name, "seed": SEED, **overrides})
        generator = seed_everything(SEED, cfg.num_threads, gpu=True)
        model = build_model(pretrained=True, arch=cfg.arch)
        t0 = time.perf_counter()
        model, history = train(model, sets["train"], sets["val"], cfg, SEED, device, generator)
        seconds = time.perf_counter() - t0
        _, logits, labels = predict(model, val, cfg)
        scores = score_rows(val, logits, labels)
        pooled = scores["val"]["pooled"]
        metrics = {
            k: pooled[k] for k in ("balanced_accuracy", "macro_auroc", "melanoma_sensitivity")
        }
        mlflow.log_metrics({f"val_{k}_pooled": v for k, v in metrics.items()})
        folder = Path(f"/Volumes/workspace/cdm/raw/checkpoints/phase2/{BATCH}-{name}-seed{SEED}")
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "/tmp/model.pt")
        (folder / "model.pt").write_bytes(Path("/tmp/model.pt").read_bytes())
        out = {
            "rung": name,
            "config": overrides,
            "metrics": metrics,
            "scores": scores,
            "best_epoch": int(max(history, key=lambda h: h["val_bacc"])["epoch"]),
            "history": history,
            "train_seconds": round(seconds),
            "checkpoint": str(folder / "model.pt"),
            "weights_sha256": weights_hash(model),
            "mlflow_run_id": run.info.run_id,
        }
        if DRIFT:
            out["drift"] = drift_aurocs(model, cfg)
    print(json.dumps({"rung": name, "metrics": metrics}), flush=True)
    return out


def judge(out: dict[str, object], reference: dict[str, object]) -> dict[str, object]:
    verdict = {"vs": reference["name"], "deltas": deltas(out["metrics"], reference["metrics"]),
               "accuracy_rule": passes(out["metrics"], reference["metrics"])}  # fmt: skip
    needs_drift = bool(out["config"].get("color_constancy")) and not reference["config"].get(
        "color_constancy"
    )
    if needs_drift:
        verdict["drift_rule"] = drift_preserved(out["drift"], reference["drift"])
        verdict["kept"] = verdict["accuracy_rule"] and verdict["drift_rule"]["preserved"]
    else:
        verdict["kept"] = verdict["accuracy_rule"]
    return verdict


# COMMAND ----------

if DRIFT:  # the reference's own drift, from its saved checkpoint
    ref_cfg = config(REFERENCE["config"])
    ref_model = build_model(pretrained=False, arch=ref_cfg.arch)
    ref_model.load_state_dict(torch.load(REFERENCE["checkpoint"], map_location="cpu"))
    assert weights_hash(ref_model) == REFERENCE["weights_sha256"], "reference checkpoint mismatch"
    REFERENCE["drift"] = drift_aurocs(ref_model.to(device).eval(), ref_cfg)

singles = []
for cand in CANDIDATES:
    out = fit_and_score(cand["rung"], merge(REFERENCE["config"], [cand["changes"]]))
    out["changes"] = cand["changes"]
    out["verdict"] = judge(out, REFERENCE)
    singles.append(out)
result["singles"] = singles

kept = [s for s in singles if s["verdict"]["kept"]]
result["kept_singles"] = [s["rung"] for s in kept]
winner = {"name": REFERENCE["name"], "config": REFERENCE["config"], "metrics": REFERENCE["metrics"]}
if kept:
    best = max(kept, key=lambda s: s["metrics"]["balanced_accuracy"])
    winner = {"name": best["rung"], "config": best["config"], "metrics": best["metrics"],
              "checkpoint": best["checkpoint"], "weights_sha256": best["weights_sha256"]}  # fmt: skip
    if DRIFT:
        winner["drift"] = best["drift"]
if combination_needed([s["rung"] for s in kept]):
    combined_cfg = merge(REFERENCE["config"], [s["changes"] for s in kept])
    combo = fit_and_score("combined", combined_cfg)
    combo["changes"] = [s["rung"] for s in kept]
    combo["verdict"] = judge(
        combo, {**winner, "drift": winner.get("drift", REFERENCE.get("drift"))}
    )
    result["combined"] = combo
    if combo["verdict"]["kept"]:
        winner = {"name": "combined", "config": combo["config"], "metrics": combo["metrics"],
                  "checkpoint": combo["checkpoint"], "weights_sha256": combo["weights_sha256"]}  # fmt: skip
result["winner"] = winner

print("BATCH_RESULT " + json.dumps(result))
dbutils.notebook.exit(json.dumps(result))
