# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2 drift table: 4 detectors x 4 shifts x 9 final models (one GPU job)
# MAGIC For each final model (v2 data: 3 centralized, 3 FedAvg, 3 FedProx), one detection look:
# MAGIC Mahalanobis (PCA size by method D, the primary row: smallest size explaining 95% of
# MAGIC training-feature variance; method C as a sensitivity row: best of 16-512 on benchmark
# MAGIC selection draws), Gram matrices, max softmax and energy. The in-distribution side is
# MAGIC the clients' pooled test split; the shifts are Buenos Aires and PAD-UFES-20 (real) and
# MAGIC PathMNIST and CIFAR-10 (benchmark). Every statistic is fitted on training data only
# MAGIC (Gram's normalizer on validation). A detection lock per model is written before its
# MAGIC first test or shift image is scored; finished models are skipped on a retry. Also counts
# MAGIC bordered images per source on the current silver, held-out sites included.

# COMMAND ----------

for name, default in {
    "wheel": "",
    "code_version": "",
    "config": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "federated_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/federated_models.json",
    "version": "v2",
    "benchmarks": "/Volumes/workspace/cdm/raw/benchmarks/benchmarks.zip",
    "benchmarks_sha256": "",
    "deadline_minutes": "100",
    "job_run_id": "",
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

import csv
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.eval import detection_metrics
from cdm.images import corner_brightness
from cdm.ood import (
    PCA_CANDIDATES,
    GramDetector,
    MahalanobisDetector,
    energy_score,
    gram_features,
    msp_score,
    pca_size_by_selection,
    pca_size_for_variance,
    stage_maps,
)
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CONFIG = json.loads(dbutils.widgets.get("config"))
VERSION = dbutils.widgets.get("version")
PREFIX = f"{VERSION}-" if VERSION else ""
CODE = dbutils.widgets.get("code_version")
SECOND_LOOK = dbutils.widgets.get("allow_second_look").lower() == "true"
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
OUT = Path("/Volumes/workspace/cdm/raw/results/phase2/drift") / VERSION
LOCKS = Path("/Volumes/workspace/cdm/raw/checkpoints/phase2/detection_looks")
OUT.mkdir(parents=True, exist_ok=True)
LOCKS.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
MODELS = [
    *json.loads(Path(dbutils.widgets.get("models_file")).read_text()),
    *json.loads(Path(dbutils.widgets.get("federated_file")).read_text()),
]
assert len(MODELS) == 9, f"expected 9 final models, got {len(MODELS)}"
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
first_start = OUT / f"first_code_start-{JOB_RUN}.txt"
if not first_start.exists():
    first_start.write_text(str(time.time()))
DEADLINE = float(first_start.read_text()) + 60 * float(dbutils.widgets.get("deadline_minutes"))

# COMMAND ----------

# Benchmarks: the zip must match the committed manifest's SHA-256, and every image its own.
blob = Path(dbutils.widgets.get("benchmarks")).read_bytes()
got = hashlib.sha256(blob).hexdigest()
assert got == dbutils.widgets.get("benchmarks_sha256"), f"benchmarks.zip sha256 {got}"
bench: dict[str, list[bytes]] = {}
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for row in csv.DictReader(io.StringIO(z.read("index.csv").decode())):
        data = z.read(row["path"])
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        bench.setdefault(f"{row['dataset']}_{row['use']}", []).append(data)
del blob

table = spark.read.table("workspace.cdm.silver_images")
everything = (
    table.select("isic_id", "source", "role", "client", "split", "label", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
everything["bordered"] = [corner_brightness(b) < 20 for b in everything["image"]]
borders = {
    "silver": "current silver (384 px shorter side, full frame, JPEG q95)",
    "by_source": {s: {"images": len(g), "bordered": int(g["bordered"].sum()),
                      "share": round(float(g["bordered"].mean()), 4)}
                  for s, g in everything.groupby("source")},
    "by_client_split": {f"{c}/{s}": {"images": len(g), "bordered": int(g["bordered"].sum())}
                        for (c, s), g in everything[everything["role"] == "client"].groupby(
                            ["client", "split"])},
}  # fmt: skip
(OUT / "borders.json").write_text(json.dumps(borders, indent=2))
print(json.dumps(borders["by_source"]), flush=True)

clients = everything[(everything["role"] == "client") & everything["label"].notna()]
INDEX = {c: i for i, c in enumerate(CLASSES)}
SETS = {
    "train": clients[clients["split"] == "train"],
    "val": clients[clients["split"] == "val"],
    "test": clients[clients["split"] == "test"],
    "buenos_aires": everything[everything["source"] == "buenos_aires"],
    "pad_ufes": everything[everything["source"] == "pad_ufes"],
}
IMAGES = {k: list(v["image"]) for k, v in SETS.items()}
LABELS = {k: v["label"].map(INDEX).fillna(-1).astype(int).tolist() for k, v in SETS.items()}
for name in ("pathmnist_eval", "cifar10_eval", "pathmnist_select", "cifar10_select"):
    IMAGES[name], LABELS[name] = bench[name], [-1] * len(bench[name])
SIZES = {k: len(v) for k, v in IMAGES.items()}
print(SIZES, flush=True)
del everything, bench

# COMMAND ----------


def loader(name: str) -> DataLoader:
    data = EncodedImages(IMAGES[name], LABELS[name], eval_transform(CONFIG.get("image_size", 224)))
    return DataLoader(data, batch_size=64, num_workers=WORKERS, pin_memory=True)


@torch.no_grad()
def passes(
    model: torch.nn.Module, name: str, gram: GramDetector, fit: bool
) -> dict[str, np.ndarray]:
    """Features, logits and Gram deviations for one set (or, with ``fit``, the Gram fit)."""
    feats, logits, devs = [], [], []
    for x, _ in loader(name):
        f, z, maps = stage_maps(model, x.to(device, non_blocking=True))
        layers = gram_features(maps)
        preds = z.argmax(1).cpu().numpy()
        if fit:
            gram.update(layers, preds)
        else:
            devs.append(gram.deviations(layers, preds))
        feats.append(f.float().cpu().numpy())
        logits.append(z.float().cpu().numpy())
    out = {"feats": np.concatenate(feats).astype(np.float64),
           "logits": np.concatenate(logits).astype(np.float64)}  # fmt: skip
    if devs:
        out["devs"] = np.concatenate(devs)
    return out


SHIFTS = {"buenos_aires": "real", "pad_ufes": "real", "pathmnist_eval": "benchmark",
          "cifar10_eval": "benchmark"}  # fmt: skip


def run(m: dict[str, object]) -> None:
    key = f"{PREFIX}{m['kind']}-seed{m['seed']}"
    out_file = OUT / f"{key}.json"
    if out_file.exists():
        print(f"{key}: result exists, skipped", flush=True)
        return
    lock = LOCKS / f"{key}.json"
    if lock.exists() and not SECOND_LOOK:
        raise RuntimeError(f"{key} already had its detection look: {lock.read_text()}")
    t0 = time.perf_counter()
    model = build_model(pretrained=False, arch=CONFIG.get("arch", "b0"))
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"{key} checkpoint mismatch"
    model.to(device).eval()
    gram = GramDetector(len(CLASSES))
    got = {"train": passes(model, "train", gram, fit=True)}
    got["val"] = passes(model, "val", gram, fit=False)
    gram.fit_normalizer(got["val"]["devs"])
    for name in ("pathmnist_select", "cifar10_select"):  # selection draws: method C only
        got[name] = passes(model, name, gram, fit=False)
    second = lock.exists()
    lock.write_text(json.dumps({"utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                "code_version": CODE, "second_look": second}))  # fmt: skip
    for name in ("test", *SHIFTS):
        got[name] = passes(model, name, gram, fit=False)

    train_labels = np.asarray(LABELS["train"], dtype=np.int64)
    k_d = pca_size_for_variance(got["train"]["feats"])
    k_c, c_table = pca_size_by_selection(
        got["train"]["feats"], train_labels, got["val"]["feats"],
        {n: got[n]["feats"] for n in ("pathmnist_select", "cifar10_select")},
    )  # fmt: skip
    maha = {k: MahalanobisDetector(k).fit(got["train"]["feats"], train_labels) for k in {k_d, k_c}}
    scores = {}
    for name in ("test", *SHIFTS):
        g = got[name]
        scores[name] = {
            "mahalanobis": maha[k_d].score(g["feats"]),
            "mahalanobis_pca_c": maha[k_c].score(g["feats"]),
            "gram": gram.score(g["devs"]),
            "max_softmax": msp_score(g["logits"]),
            "energy": energy_score(g["logits"]),
        }
    metrics = {
        det: {shift: detection_metrics(scores["test"][det], scores[shift][det]) for shift in SHIFTS}
        for det in scores["test"]
    }
    np.savez_compressed(OUT / f"{key}_scores.npz",
                        **{f"{n}__{d}": v for n, ds in scores.items() for d, v in ds.items()})  # fmt: skip
    test_acc = float((got["test"]["logits"].argmax(1) == np.asarray(LABELS["test"])).mean())
    result = {
        "model": key, "kind": m["kind"], "seed": m["seed"], "code_version": CODE,
        "weights_sha256": m["weights_sha256"], "images": SIZES,
        "pca": {"method_d_components": k_d, "method_c_components": k_c,
                "method_c_candidates": list(PCA_CANDIDATES),
                "method_c_selection_auroc": {str(k): v for k, v in c_table.items()}},
        "metrics": metrics, "shift_kind": SHIFTS,
        "gram_layer_normalizer": gram.norm.tolist(),
        "test_accuracy_check": test_acc, "seconds": round(time.perf_counter() - t0),
    }  # fmt: skip
    out_file.write_text(json.dumps(result))
    summary = {d: round(metrics[d]["buenos_aires"]["auroc"], 4) for d in metrics}
    print(json.dumps({"model": key, "k_d": k_d, "k_c": k_c, "ba_auroc": summary}), flush=True)


# COMMAND ----------

failed, not_started = {}, []
for m in MODELS:
    key = f"{PREFIX}{m['kind']}-seed{m['seed']}"
    if time.time() > DEADLINE and not (OUT / f"{key}.json").exists():
        not_started.append(key)
        continue
    try:
        run(m)
    except Exception as e:  # recorded, then the next model
        import traceback

        failed[key] = traceback.format_exc()[-3000:]
        print(f"{key} FAILED: {e}", flush=True)
        torch.cuda.empty_cache()

summary = {"code_version": CODE, "job_run_id": JOB_RUN,
           "finished": sorted(f.stem for f in OUT.glob(f"{PREFIX}*-seed*.json")),
           "failed": failed, "not_started_deadline": not_started}  # fmt: skip
(OUT / "summary.json").write_text(json.dumps(summary))
print("DRIFT_SUMMARY " + json.dumps(summary), flush=True)
dbutils.notebook.exit(json.dumps(summary))
