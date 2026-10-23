# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: the serving bundle, the abstain threshold and risk-coverage (serverless CPU)
# MAGIC Under `results/phase3/serving/spec.json`. Everything is computed on CPU, the serving
# MAGIC hardware. Each shipped seed's features and logits are computed for the clients' train,
# MAGIC validation and test splits and the two new hospitals; each seed's Mahalanobis detector
# MAGIC (PCA by method D) is fit on its train features; the serving code's NumPy scorer must match
# MAGIC the detector. The abstain threshold is the 95th percentile of the ensemble OOD score on
# MAGIC validation, and risk-coverage is computed on validation. Then one look at test and the new
# MAGIC hospitals: the attempt is logged first, the bundle and results are saved, then the lock.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/serving",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import eval_transform
from cdm.ood import MahalanobisDetector, extract, pca_size_for_variance
from cdm.serving import balanced_accuracy, detector_arrays, mahalanobis, risk_coverage
from cdm.silver import EncodedImages
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
BUNDLE = Path(f"/Volumes/workspace/cdm/raw/deploy/serving/{CODE}")

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
for name in ("results.json", "test_look.json"):
    if (OUT / name).exists():
        raise RuntimeError(f"{OUT / name} exists: the test look already happened")
spec = json.loads((REPO / "results/phase3/serving/spec.json").read_text(encoding="utf-8"))
REFER = json.loads((REPO / "results/phase2/triage/threshold.json").read_text(encoding="utf-8"))
refer_threshold = float(REFER["thresholds"]["v2-ensemble-final-seeds0-1-2"])
GRID = spec["risk_coverage"]["coverage_grid"]
INDEX = {c: i for i, c in enumerate(CLASSES)}

# COMMAND ----------

everything = (
    spark.read.table("workspace.cdm.silver_images")
    .select("isic_id", "source", "role", "client", "split", "label", "image")
    .orderBy("source", "isic_id")
    .toPandas()
)
clients = everything[(everything["role"] == "client") & everything["label"].notna()]
SETS = {
    "train": clients[clients["split"] == "train"],
    "val": clients[clients["split"] == "val"],
    "test": clients[clients["split"] == "test"],
    "buenos_aires": everything[everything["source"] == "buenos_aires"],
    "pad_ufes": everything[everything["source"] == "pad_ufes"],
}
LABELS = {k: v["label"].map(INDEX).fillna(-1).astype(int).to_numpy() for k, v in SETS.items()}
print({k: len(v) for k, v in SETS.items()}, flush=True)

manifest_models = sorted(
    json.loads(Path(dbutils.widgets.get("models_file")).read_text()), key=lambda m: m["seed"]
)
feats, probs, scores, detectors, checks = {}, {}, {}, [], {}
for m in manifest_models:
    model = build_model(pretrained=False, arch="b0")
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"seed {m['seed']} checkpoint mismatch"
    model.eval()
    per_set = {}
    for name, frame in SETS.items():
        data = EncodedImages(list(frame["image"]), [0] * len(frame), eval_transform(224))
        with torch.no_grad():
            f, z, _ = extract(
                model, DataLoader(data, batch_size=64, num_workers=0), torch.device("cpu")
            )
        per_set[name] = (f, z)
        print(f"seed {m['seed']} {name}: {f.shape}", flush=True)
    k = pca_size_for_variance(per_set["train"][0])
    det = MahalanobisDetector(k).fit(per_set["train"][0], LABELS["train"])
    arrays = detector_arrays(det)
    ref, mine = det.score(per_set["val"][0]), mahalanobis(arrays, per_set["val"][0])
    rel = float(np.max(np.abs(ref - mine) / np.maximum(np.abs(ref), 1e-12)))
    if rel > 1e-6:
        raise RuntimeError(f"seed {m['seed']}: NumPy scorer differs from the detector ({rel})")
    checks[f"seed{m['seed']}"] = {
        "pca_components_method_d": int(k),
        "numpy_scorer_max_relative_diff": rel,
    }
    detectors.append((m, arrays))
    for name, (f, z) in per_set.items():
        sm = np.exp(z - z.max(1, keepdims=True))
        probs.setdefault(name, []).append(sm / sm.sum(1, keepdims=True))
        scores.setdefault(name, []).append(
            mahalanobis(arrays, f) if name != "train" else np.zeros(len(f))
        )
    del model, per_set

P = {k: np.mean(v, axis=0) for k, v in probs.items()}
S = {k: np.mean(v, axis=0) for k, v in scores.items()}
threshold = float(np.percentile(S["val"], 95))


def report(name: str, mask: np.ndarray) -> dict:
    y, p, s = LABELS[name][mask], P[name][mask], S[name][mask]
    labelled = y >= 0
    out = {
        "images": int(mask.sum()),
        "workload_share_above_threshold": float(np.mean(s > threshold)),
        "labelled": int(labelled.sum()),
    }
    if labelled.any():
        yl, pl, sl = y[labelled], p[labelled], s[labelled]
        out["balanced_accuracy_all"] = balanced_accuracy(yl, pl.argmax(1))
        out["risk_coverage"] = {
            "confidence": risk_coverage(yl, pl, -pl.max(axis=1), GRID),
            "ood": risk_coverage(yl, pl, sl, GRID),
        }
    return out


validation = report("val", np.ones(len(SETS["val"]), bool))

# COMMAND ----------

# The one look at test and the new hospitals.
OUT.mkdir(parents=True, exist_ok=True)
started = datetime.now(UTC).isoformat(timespec="seconds")
with (OUT / "look_attempts.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps({"started_utc": started, "job_run": JOB_RUN, "code_version": CODE}) + "\n")
test_rows = {"client_test": report("test", np.ones(len(SETS["test"]), bool))}
for client in sorted(SETS["test"]["client"].unique()):
    test_rows[f"client_test_{client}"] = report(
        "test", (SETS["test"]["client"] == client).to_numpy()
    )
for site in ("buenos_aires", "pad_ufes"):
    test_rows[site] = report(site, np.ones(len(SETS[site]), bool))

# COMMAND ----------

local = Path(tempfile.mkdtemp())
seeds = []
for m, arrays in detectors:
    ck, dt = f"seed{m['seed']}.pt", f"seed{m['seed']}_detector.npz"
    shutil.copyfile(m["checkpoint"], local / ck)
    np.savez(local / dt, **arrays)
    seeds.append(
        {"seed": m["seed"], "checkpoint": ck, "detector": dt, "weights_sha256": m["weights_sha256"]}
    )
files = [s["checkpoint"] for s in seeds] + [s["detector"] for s in seeds]
manifest = {
    "version": f"v2-ensemble-cpu-{CODE}",
    "classes": list(CLASSES),
    "abstain_threshold": threshold,
    "refer_threshold": refer_threshold,
    "seeds": seeds,
    "sha256": {f: hashlib.sha256((local / f).read_bytes()).hexdigest() for f in files},
    "spec": "results/phase3/serving/spec.json",
}
(local / "manifest.json").write_text(json.dumps(manifest, indent=2))
BUNDLE.mkdir(parents=True, exist_ok=True)
for f in [*files, "manifest.json"]:
    (BUNDLE / f).write_bytes((local / f).read_bytes())
    assert (
        hashlib.sha256((BUNDLE / f).read_bytes()).hexdigest()
        == hashlib.sha256((local / f).read_bytes()).hexdigest()
    )

results = {
    "spec": "results/phase3/serving/spec.json",
    "spec_sha256": expected["results/phase3/serving/spec.json"],
    "code_version": CODE,
    "wheel_sha256": WHEEL_SHA,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "abstain_threshold": threshold,
    "refer_threshold": refer_threshold,
    "detector_checks": checks,
    "validation": validation,
    "test_look": test_rows,
    "bundle": {
        "path": str(BUNDLE),
        "version": manifest["version"],
        "manifest_sha256": hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest(),
    },
    "note": "research prototype, not for medical use; CPU scores, the serving hardware",
}
(OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
lock = {
    "started_utc": started,
    "code_version": CODE,
    "spec_sha256": results["spec_sha256"],
    "results_sha256": hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest(),
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
}
(OUT / "test_look.json").write_text(json.dumps(lock, indent=2) + "\n")
print(
    json.dumps(
        {
            "threshold": threshold,
            "val_workload": validation["workload_share_above_threshold"],
            "test_workload": test_rows["client_test"]["workload_share_above_threshold"],
        }
    )
)
