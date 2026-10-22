# Databricks notebook source
# MAGIC %md
# MAGIC # Research Phase B: the single test look (serverless CPU)
# MAGIC Runs `scripts/vlm_baseline.py` at a committed version on the GPU job's BiomedCLIP
# MAGIC embeddings. The script and its inputs (the committed spec, the EfficientNet analysis-look
# MAGIC predictions and the patient-overlap list) are copied from that commit into the volume and
# MAGIC checked here by SHA-256.
# MAGIC
# MAGIC `mode=dry_run` (the default) does the whole computation on validation and the drift
# MAGIC selection draws, and reports time and peak memory: no test or external-site image is scored.
# MAGIC `mode=look` is the one test look. It refuses if a results file or lock is already in the
# MAGIC output folder, computes everything in memory without printing a metric, writes
# MAGIC `results.json`, and only then the lock `test_look.json`. A crash before the results are
# MAGIC saved leaves nothing behind, so the look stays unused. The owner's computer runs
# MAGIC `vlm_baseline.py preflight --code-version <commit>` (git checks) before the look.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "repo": "",
    "files_sha256": "{}",
    "embeddings": "/Volumes/workspace/cdm/raw/results/research/b_vlm/embeddings",
    "out": "/Volumes/workspace/cdm/raw/results/research/b_vlm/look",
    "dry_run_out": "/Volumes/workspace/cdm/raw/results/research/b_vlm/dry_run",
    "mode": "dry_run",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
MODE = dbutils.widgets.get("mode")
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
assert MODE in ("dry_run", "look"), f"mode {MODE}"

# Every copied file must be the committed one.
expected = json.loads(dbutils.widgets.get("files_sha256"))
assert expected, "files_sha256 is empty"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"

spec = importlib.util.spec_from_file_location("vlm_baseline", REPO / "scripts" / "vlm_baseline.py")
assert spec is not None and spec.loader is not None
vb = importlib.util.module_from_spec(spec)
sys.modules["vlm_baseline"] = vb  # dataclasses look their module up
spec.loader.exec_module(vb)
assert vb.REPO == REPO, vb.REPO


def checked(fn, *args):
    """The script refuses with sys.exit, which a notebook may not treat as a failure."""
    try:
        return fn(*args)
    except SystemExit as refusal:
        raise RuntimeError(str(refusal)) from None


emb = checked(vb.Embeddings, Path(dbutils.widgets.get("embeddings")))
print("embeddings checked against the GPU job's summary", flush=True)

# COMMAND ----------

if MODE == "dry_run":
    usage = checked(vb.dry_run, emb, Path(dbutils.widgets.get("dry_run_out")))
    print(f"dry run on validation: {usage}")
else:
    OUT = Path(dbutils.widgets.get("out"))
    for name in (vb.RESULTS.name, vb.LOCK.name, vb.RESULTS.name + ".tmp", vb.LOCK.name + ".tmp"):
        if (OUT / name).exists():
            raise RuntimeError(f"{OUT / name} exists: the test look already happened or began")
    plan = json.loads(vb.SPEC.read_text(encoding="utf-8"))
    checked(vb.check_inputs, emb, plan, vb.LOOK)
    look = vb.new_look(emb, CODE, f"Databricks serverless CPU, job run {JOB_RUN}")
    vb.record(vb.compute(emb, plan, vb.LOOK), look, OUT)
    print(f"wrote {OUT / vb.RESULTS.name}, then {OUT / vb.LOCK.name}")
    print(f"results sha256 {vb.sha256(OUT / vb.RESULTS.name)}")
