# Databricks notebook source
# MAGIC %md
# MAGIC # Batch-level drift: Mahalanobis validation reference (CPU), spec amendment 2
# MAGIC For each v2 centralized model, refit Mahalanobis (the drift run's method-D PCA size) on
# MAGIC CPU training features and score validation. Two checks, both recorded per model before
# MAGIC its validation scores are saved, and either failure stops the job (tolerances are not
# MAGIC loosened): (1) the refit's method-D PCA size is within 1 component of the drift run's;
# MAGIC (2) for 300 benchmark evaluation images (150 PathMNIST, 150 CIFAR-10, seeded positions),
# MAGIC the CPU scores match the drift run's saved GPU scores: Spearman >= 0.999 and at least
# MAGIC 99% within 1% relative difference. Scores only; no metric is computed on those images.
# MAGIC Finished models (check passed, scores saved) are skipped on a rerun.
# MAGIC Spec: `results/phase2/batch_drift/spec.json`.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "drift_results": "/Volumes/workspace/cdm/raw/results/phase2/drift/v2",
    "benchmarks": "/Volumes/workspace/cdm/raw/benchmarks/benchmarks.zip",
    "benchmarks_sha256": "",
    "pca_size_tolerance": "1",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", dbutils.widgets.get("wheel")], check=True
)

# COMMAND ----------

import csv
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.ood import MahalanobisDetector, extract, pca_size_for_variance
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
CODE = dbutils.widgets.get("code_version")
DRIFT = Path(dbutils.widgets.get("drift_results"))
K_TOL = int(dbutils.widgets.get("pca_size_tolerance"))
RHO_MIN, WITHIN, SHARE_MIN, N_EACH = 0.999, 0.01, 0.99, 150  # spec amendment 2
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/batch_drift")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = sorted(json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
                key=lambda m: m["seed"])  # fmt: skip
INDEX = {c: i for i, c in enumerate(CLASSES)}
device = torch.device("cpu")

blob = Path(dbutils.widgets.get("benchmarks")).read_bytes()
assert hashlib.sha256(blob).hexdigest() == dbutils.widgets.get("benchmarks_sha256"), "benchmarks"
bench: dict[str, list[bytes]] = {}
with zipfile.ZipFile(io.BytesIO(blob)) as z:  # same order as the drift run read them
    for row in csv.DictReader(io.StringIO(z.read("index.csv").decode())):
        if row["use"] == "eval":
            bench.setdefault(f"{row['dataset']}_eval", []).append(z.read(row["path"]))
del blob
rng = np.random.default_rng(0)
PICK = {name: np.sort(rng.choice(len(bench[name]), N_EACH, replace=False))
        for name in sorted(bench)}  # fmt: skip

clients = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND label IS NOT NULL AND split IN ('train', 'val')")
    .select("isic_id", "split", "label", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
SETS = {s: clients[clients["split"] == s].reset_index(drop=True) for s in ("train", "val")}
print({s: len(f) for s, f in SETS.items()}, {k: len(v) for k, v in PICK.items()}, flush=True)

# COMMAND ----------


def feats(model: torch.nn.Module, images: list[bytes], labels: list[int]) -> np.ndarray:
    data = EncodedImages(images, labels, eval_transform(CONFIG.get("image_size", 224)))
    f, _, _ = extract(model, DataLoader(data, batch_size=64, num_workers=0), device)
    return f


for m in MODELS:
    key = f"v2-{m['kind']}-seed{m['seed']}"
    check_file, scores_file = OUT / f"{key}_check.json", OUT / f"{key}_val_scores.npz"
    if (
        check_file.exists()
        and scores_file.exists()
        and json.loads(check_file.read_text())["passed"]
    ):
        print(f"{key}: check passed and scores saved earlier, skipped", flush=True)
        continue
    drift = json.loads((DRIFT / f"{key}.json").read_text())
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"] == drift["weights_sha256"], key
    model.eval()
    train_labels = SETS["train"]["label"].map(INDEX)
    train = feats(model, list(SETS["train"]["image"]), train_labels.tolist())
    k_refit = pca_size_for_variance(train)
    k_saved = int(drift["pca"]["method_d_components"])
    maha = MahalanobisDetector(k_saved).fit(train, train_labels.to_numpy())
    with np.load(DRIFT / f"{key}_scores.npz") as gpu:
        gpu_scores = {n: gpu[f"{n}__mahalanobis"] for n in PICK}
    numeric, rel_all = {}, []
    for name, pos in PICK.items():
        imgs = [bench[name][i] for i in pos]
        cpu = maha.score(feats(model, imgs, [-1] * len(imgs)))
        ref = gpu_scores[name][pos]
        rel = np.abs(cpu - ref) / np.abs(ref)
        rel_all.append(rel)
        numeric[name] = {"images": int(len(pos)), "spearman": float(spearmanr(cpu, ref).statistic),
                         "share_within_1pct": float((rel <= WITHIN).mean()),
                         "max_relative_difference": float(rel.max())}  # fmt: skip
    rho = min(v["spearman"] for v in numeric.values())
    share = float((np.concatenate(rel_all) <= WITHIN).mean())
    check = {
        "model": key, "code_version": CODE,
        "pca_size": {"refit": k_refit, "drift_run": k_saved, "used": k_saved,
                     "within_tolerance": abs(k_refit - k_saved) <= K_TOL},
        "cpu_vs_gpu_scores": {"per_set": numeric, "spearman_min": rho,
                              "share_within_1pct_all_300": share,
                              "passed": rho >= RHO_MIN and share >= SHARE_MIN},
        "note": "scores only: no metric was computed on the benchmark images",
    }  # fmt: skip
    check["passed"] = check["pca_size"]["within_tolerance"] and check["cpu_vs_gpu_scores"]["passed"]
    check_file.write_text(json.dumps(check, indent=2))  # saved before anything else
    print(key, json.dumps(check), flush=True)
    if not check["passed"]:
        raise RuntimeError(f"{key}: check failed, stopping as the spec requires: {check}")
    val_scores = maha.score(feats(model, list(SETS["val"]["image"]),
                                  SETS["val"]["label"].map(INDEX).tolist()))  # fmt: skip
    local = Path(tempfile.gettempdir()) / scores_file.name
    np.savez_compressed(local, mahalanobis=val_scores)
    scores_file.write_bytes(local.read_bytes())
    with np.load(scores_file) as z:
        assert z["mahalanobis"].shape == (len(SETS["val"]),), "scores did not read back"

summary = {k: json.loads((OUT / f"{k}_check.json").read_text())["passed"]
           for k in (f"v2-{m['kind']}-seed{m['seed']}" for m in MODELS)}  # fmt: skip
dbutils.notebook.exit(json.dumps(summary))
