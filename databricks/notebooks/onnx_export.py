# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: ONNX export, parity with PyTorch, and CPU latency (serverless CPU)
# MAGIC Two modes on the same 512 validation images (no test image):
# MAGIC - `mode=noise`: PyTorch against itself, batch 1 against batch 32, per seed and for the
# MAGIC   ensemble. No ONNX. Writes `noise.json`, which the amended spec (`spec_v2.json`) quotes.
# MAGIC - `mode=parity`: exports each shipped seed (features and logits, dynamic batch), runs ONNX
# MAGIC   Runtime and PyTorch at batch 32, and applies the amended rule: every gap at most 2x
# MAGIC   PyTorch's own noise, and a top-class disagreement only where PyTorch also disagrees with
# MAGIC   itself or the top two classes are within 0.01. Times one-image inference. Results are
# MAGIC   written first; a miss then fails the job. ONNX files go to the volume, never to git.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "models_file": "/Volumes/workspace/cdm/raw/results/phase2/overnight/v2/final_models.json",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/onnx",
    "mode": "noise",
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
from cdm.serve_onnx import (
    export,
    gaps,
    latency_ms,
    onnx_outputs,
    passes_against_noise,
    session,
    softmax,
    top2_margin,
    torch_outputs,
)
from cdm.train import build_model, weights_hash

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
MODE = dbutils.widgets.get("mode")
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
assert MODE in ("noise", "parity"), MODE
VOLUME_ONNX = Path(f"/Volumes/workspace/cdm/raw/deploy/onnx/{CODE}")
BATCH = 32

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
target = OUT / ("noise.json" if MODE == "noise" else "results.json")
if target.exists():
    raise RuntimeError(f"{target} exists: this step already ran")
if MODE == "parity":
    assert "results/phase3/onnx/spec_v2.json" in expected, "parity needs the committed amended spec"
    SPEC = json.loads((REPO / "results/phase3/onnx/spec_v2.json").read_text(encoding="utf-8"))

# COMMAND ----------

# 512 validation images at seeded positions (no test image).
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
images_sha256 = hashlib.sha256(images.numpy().tobytes()).hexdigest()

models = {}
for m in sorted(
    json.loads(Path(dbutils.widgets.get("models_file")).read_text()), key=lambda m: m["seed"]
):
    key = f"v2-final-seed{m['seed']}"
    model = build_model(pretrained=False, arch="b0")
    model.load_state_dict(torch.load(m["checkpoint"], map_location="cpu"))
    assert weights_hash(model) == m["weights_sha256"], f"{key} checkpoint mismatch"
    models[key] = model.eval()
reference = {k: torch_outputs(m, images, BATCH) for k, m in models.items()}
ens_ref = np.mean([softmax(v[1]) for v in reference.values()], axis=0)


def ensemble_gap(p: np.ndarray, q: np.ndarray) -> dict:
    return {
        "probabilities_max_abs_diff": float(np.abs(p - q).max()),
        "top1_disagreements": [int(i) for i in np.flatnonzero(p.argmax(1) != q.argmax(1))],
    }


common = {
    "code_version": CODE,
    "wheel_sha256": WHEEL_SHA,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": None,
    "images": len(images),
    "images_sha256": images_sha256,
    "versions": {p: md.version(p) for p in ("torch", "onnx", "onnxruntime", "onnxscript")},
    "machine": {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()},
}

# COMMAND ----------

if MODE == "noise":
    single = {k: torch_outputs(m, images, 1) for k, m in models.items()}
    noise = {k: gaps(single[k], reference[k]) for k in models}
    noise["ensemble"] = ensemble_gap(
        np.mean([softmax(v[1]) for v in single.values()], axis=0), ens_ref
    )
    out = {**common, "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
           "what": "PyTorch at batch 1 against PyTorch at batch 32, same images and weights",
           "noise": noise}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "noise.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {m: v[m] for m in v if m != "top1_disagreements"}
                | {"flips": len(v["top1_disagreements"])}
                for k, v in noise.items()
            }
        )
    )

# COMMAND ----------

if MODE == "parity":
    local = Path(tempfile.mkdtemp())
    sessions, files, checks, failures = {}, {}, {}, {}
    for k, m in models.items():
        path = export(m, local / f"{k}.onnx")
        sessions[k] = session(path)
        data = path.read_bytes()
        files[k] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        got = onnx_outputs(sessions[k], images, BATCH)
        checks[k] = gaps(reference[k], got)
        failures[k] = passes_against_noise(
            checks[k], SPEC["noise"][k], top2_margin(reference[k][1]),
            factor=SPEC["rule"]["factor"], close=SPEC["rule"]["top2_close"],
        )  # fmt: skip
        print(k, checks[k], failures[k], flush=True)
    ens_onnx = np.mean(
        [softmax(onnx_outputs(s, images, BATCH)[1]) for s in sessions.values()], axis=0
    )
    checks["ensemble"] = ensemble_gap(ens_ref, ens_onnx)
    srt = np.sort(ens_ref, axis=1)
    failures["ensemble"] = passes_against_noise(
        checks["ensemble"], SPEC["noise"]["ensemble"], srt[:, -1] - srt[:, -2],
        factor=SPEC["rule"]["factor"], close=SPEC["rule"]["top2_close"],
    )  # fmt: skip

    one, one_np = images[:1], images[:1].numpy()
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
                lambda: [
                    s.run(["features", "logits"], {"image": one_np}) for s in sessions.values()
                ]
            ),
        }

    VOLUME_ONNX.mkdir(parents=True, exist_ok=True)
    for k in models:
        (VOLUME_ONNX / f"{k}.onnx").write_bytes((local / f"{k}.onnx").read_bytes())
        assert (
            hashlib.sha256((VOLUME_ONNX / f"{k}.onnx").read_bytes()).hexdigest()
            == files[k]["sha256"]
        )
    results = {
        **common,
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": "results/phase3/onnx/spec_v2.json",
        "spec_sha256": expected["results/phase3/onnx/spec_v2.json"],
        "parity": checks,
        "failures": failures,
        "passed": not any(failures.values()),
        "latency_ms_batch_1": latency,
        "files": {
            k: {**v, "volume_path": str(VOLUME_ONNX / f"{k}.onnx")} for k, v in files.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"passed": results["passed"], "failures": failures}))
    if not results["passed"]:
        raise RuntimeError(f"parity outside the amended rule: {failures}")
