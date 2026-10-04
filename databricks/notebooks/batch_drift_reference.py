# Databricks notebook source
# MAGIC %md
# MAGIC # Batch-level drift: validation reference scores (CPU, no test or shift images)
# MAGIC For each v2 centralized model, refit the drift run's Mahalanobis (PCA size by method D)
# MAGIC and Gram detectors on training images only, then score validation. Stops unless the
# MAGIC refit reproduces the drift run's method-D PCA size and Gram layer normalizer for that
# MAGIC model (`results/phase2/drift/v2`), so the reference comes from the same fitted detector
# MAGIC that scored the test and shift images. Spec: `results/phase2/batch_drift/spec.json`.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "drift_results": "/Volumes/workspace/cdm/raw/results/phase2/drift/v2",
    "normalizer_rtol": "1e-3",  # 0.1% relative, from the spec; not to be loosened
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

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.ood import (
    GramDetector,
    MahalanobisDetector,
    gram_features,
    pca_size_for_variance,
    stage_maps,
)
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
CODE = dbutils.widgets.get("code_version")
DRIFT = Path(dbutils.widgets.get("drift_results"))
RTOL = float(dbutils.widgets.get("normalizer_rtol"))
K_TOL = int(dbutils.widgets.get("pca_size_tolerance"))
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/batch_drift")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = sorted(json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
                key=lambda m: m["seed"])  # fmt: skip
INDEX = {c: i for i, c in enumerate(CLASSES)}
torch.set_num_threads(max(1, torch.get_num_threads()))
clients = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND label IS NOT NULL AND split IN ('train', 'val')")
    .select("isic_id", "client", "split", "label", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
SETS = {s: clients[clients["split"] == s].reset_index(drop=True) for s in ("train", "val")}
print({s: len(f) for s, f in SETS.items()}, flush=True)

# COMMAND ----------


@torch.no_grad()
def passes(model, name: str, gram: GramDetector, fit: bool) -> dict[str, np.ndarray]:
    frame = SETS[name]
    data = EncodedImages(list(frame["image"]), frame["label"].map(INDEX).tolist(),
                         eval_transform(CONFIG.get("image_size", 224)))  # fmt: skip
    feats, devs = [], []
    for x, _ in DataLoader(data, batch_size=64, num_workers=0):
        f, z, maps = stage_maps(model, x)
        layers, preds = gram_features(maps), z.argmax(1).numpy()
        if fit:
            gram.update(layers, preds)
        else:
            devs.append(gram.deviations(layers, preds))
        feats.append(f.float().numpy())
    out = {"feats": np.concatenate(feats).astype(np.float64)}
    if devs:
        out["devs"] = np.concatenate(devs)
    return out


summary = {"code_version": CODE, "tolerance": {"pca_size": K_TOL, "normalizer_rtol": RTOL},
           "gram_normalizer_fit_on": "validation (mean deviation per layer), in the drift run and here",
           "models": {}}  # fmt: skip
for m in MODELS:
    key = f"v2-{m['kind']}-seed{m['seed']}"
    drift = json.loads((DRIFT / f"{key}.json").read_text())
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"] == drift["weights_sha256"], key
    model.eval()
    gram = GramDetector(len(CLASSES))
    train = passes(model, "train", gram, fit=True)
    val = passes(model, "val", gram, fit=False)
    gram.fit_normalizer(val["devs"])
    k_d = pca_size_for_variance(train["feats"])
    saved_norm = np.asarray(drift["gram_layer_normalizer"])
    norm_ok = bool(np.all(np.abs(gram.norm - saved_norm) <= RTOL * np.abs(saved_norm)))
    k_saved = int(drift["pca"]["method_d_components"])
    check = {"method_d_components": {"refit": k_d, "drift_run": k_saved, "used": k_saved},
             "gram_normalizer_max_rel_diff": float(np.max(np.abs(gram.norm - saved_norm)
                                                          / np.abs(saved_norm))),
             "pca_size_within_tolerance": abs(k_d - k_saved) <= K_TOL,
             "normalizer_within_tolerance": norm_ok,
             "matches": abs(k_d - k_saved) <= K_TOL and norm_ok}  # fmt: skip
    summary["models"][key] = check
    print(key, json.dumps(check), flush=True)
    if not check["matches"]:
        raise RuntimeError(f"{key}: the refit does not reproduce the drift run's detector: {check}")
    train_labels = SETS["train"]["label"].map(INDEX).to_numpy()
    # The drift run's own PCA size, so validation is scored by the detector that scored test.
    maha = MahalanobisDetector(k_saved).fit(train["feats"], train_labels)
    local = Path(tempfile.gettempdir()) / f"{key}_val_scores.npz"
    np.savez_compressed(local, mahalanobis=maha.score(val["feats"]), gram=gram.score(val["devs"]))
    assert set(np.load(local).files) == {"mahalanobis", "gram"}
    (OUT / local.name).write_bytes(local.read_bytes())

(OUT / "reference_check.json").write_text(json.dumps(summary, indent=2))
dbutils.notebook.exit(json.dumps(summary))
