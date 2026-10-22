# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: ONNX export, parity with PyTorch, and CPU latency (serverless CPU)
# MAGIC Under `results/phase3/onnx/spec.json`. Exports each shipped EfficientNet-B0 v2 seed with
# MAGIC its features and logits, checks ONNX Runtime against PyTorch on 512 validation images
# MAGIC (per seed and for the 3-model ensemble) against tolerances fixed in the spec, and times
# MAGIC one-image inference on this machine. Results are written first; a tolerance miss then
# MAGIC fails the job. The ONNX files go to the volume, never to git.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/onnx",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import importlib.metadata as md
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyspark.sql import functions as F

from cdm.data import eval_transform
from cdm.serve_onnx import export, latency_ms, parity, session, softmax, within
from cdm.train import build_model, weights_hash

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
VOLUME_ONNX = Path(f"/Volumes/workspace/cdm/raw/deploy/onnx/{CODE}")

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
if (OUT / "results.json").exists():
    raise RuntimeError(f"{OUT / 'results.json'} exists: this export already ran")
spec = json.loads((REPO / "results/phase3/onnx/spec.json").read_text(encoding="utf-8"))
TOL = spec["tolerance"]

# COMMAND ----------

# 512 validation images at seeded positions (the spec's parity set; no test image).
val = (
    spark.read.table("workspace.cdm.silver_images")
    .where("role = 'client' AND split = 'val' AND label IS NOT NULL")
    .select("source", "isic_id")
    .orderBy("source", "isic_id")
    .toPandas()
)
pick = np.sort(np.random.default_rng(0).choice(len(val), 512, replace=False))
ids = list(val.iloc[pick]["isic_id"])
blobs = (
    spark.read.table("workspace.cdm.silver_images")
    .where(F.col("isic_id").isin(ids))
    .select("isic_id", "image")
    .toPandas()
    .set_index("isic_id")
    .loc[ids, "image"]
)
transform = eval_transform(224)
images = torch.stack([transform(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs])

manifest = json.loads(Path(dbutils.widgets.get("models_file")).read_text())
local = Path(tempfile.mkdtemp())
models, sessions, files, checks = {}, {}, {}, {}
for m in sorted(manifest, key=lambda m: m["seed"]):
    key = f"v2-final-seed{m['seed']}"
    model = build_model(pretrained=False, arch="b0")
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"{key} checkpoint mismatch"
    models[key] = model.eval()
    path = export(models[key], local / f"{key}.onnx")
    sessions[key] = session(path)
    checks[key] = parity(models[key], sessions[key], images)
    data = path.read_bytes()
    files[key] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    print(key, checks[key], flush=True)

with torch.no_grad():
    ens_t = np.mean([softmax(models[k](images).double().numpy()) for k in models], axis=0)
ens_o = np.mean(
    [
        softmax(
            np.asarray(sessions[k].run(["logits"], {"image": images.numpy()})[0], dtype=np.float64)
        )
        for k in sessions
    ],
    axis=0,
)
checks["ensemble"] = {
    "images": len(images),
    "probabilities_max_abs_diff": float(np.abs(ens_t - ens_o).max()),
    "top1_agreement": float(np.mean(ens_t.argmax(1) == ens_o.argmax(1))),
}
failures = {
    k: within(v, TOL["ensemble" if k == "ensemble" else "per_seed"]) for k, v in checks.items()
}

# COMMAND ----------

one = images[:1]
one_np = one.numpy()
latency = {}
with torch.no_grad():
    for k in models:
        latency[k] = {
            "pytorch": latency_ms(lambda k=k: models[k](one)),
            "onnxruntime": latency_ms(
                lambda k=k: sessions[k].run(["features", "logits"], {"image": one_np})
            ),
        }
    latency["ensemble"] = {
        "pytorch": latency_ms(lambda: [models[k](one) for k in models]),
        "onnxruntime": latency_ms(
            lambda: [sessions[k].run(["features", "logits"], {"image": one_np}) for k in sessions]
        ),
    }

# COMMAND ----------

VOLUME_ONNX.mkdir(parents=True, exist_ok=True)
for k in models:
    (VOLUME_ONNX / f"{k}.onnx").write_bytes((local / f"{k}.onnx").read_bytes())
    assert (
        hashlib.sha256((VOLUME_ONNX / f"{k}.onnx").read_bytes()).hexdigest() == files[k]["sha256"]
    )
results = {
    "spec": "results/phase3/onnx/spec.json",
    "spec_sha256": expected["results/phase3/onnx/spec.json"],
    "code_version": CODE,
    "wheel_sha256": WHEEL_SHA,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "versions": {p: md.version(p) for p in ("torch", "onnx", "onnxruntime", "onnxscript")},
    "machine": {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()},
    "parity": checks,
    "tolerance": TOL,
    "failures": failures,
    "passed": not any(failures.values()),
    "latency_ms_batch_1": latency,
    "files": {k: {**v, "volume_path": str(VOLUME_ONNX / f"{k}.onnx")} for k, v in files.items()},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"passed": results["passed"], "failures": failures}))
if not results["passed"]:
    raise RuntimeError(f"parity outside the spec's tolerance: {failures}")
