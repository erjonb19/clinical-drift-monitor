# Databricks notebook source
# MAGIC %md
# MAGIC # Backlog item 4: backfill the melanoma referral count of replay months 1 to 12
# MAGIC Rebuilds each replay month's delayed-accuracy record, with `melanoma_referred`, from the rows
# MAGIC the replay wrote to `gold_prediction_audit` and the labels in `silver_images` (nothing is
# MAGIC re-scored), so the melanoma-referral target (`cdm.slo`) can be judged. It refuses unless the
# MAGIC stored referral flags match the bundle's threshold and every rebuilt record matches the one
# MAGIC the replay committed (`cdm.backfill`). Writes `gold_melanoma_backfill` once (refuses if it
# MAGIC exists) and `melanoma_backfill.json` for the repo. The monthly job reads the table.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "bundle": "",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/monthly",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from cdm.backfill import check_committed, committed_records, month_record
from cdm.splits import CLASSES

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
BUNDLE = Path(dbutils.widgets.get("bundle"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
AUDIT, TABLE = "workspace.cdm.gold_prediction_audit", "workspace.cdm.gold_melanoma_backfill"
MONTHS = list(range(1, 13))
INDEX = {c: i for i, c in enumerate(CLASSES)}

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
committed_files = [f"results/phase3/monthly/month_{m:02d}.json" for m in MONTHS]
missing = [
    f for f in ["results/phase3/serving/results.json", *committed_files] if f not in expected
]
assert not missing, f"not staged and hashed: {missing}"
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
evaluated = json.loads((REPO / "results/phase3/serving/results.json").read_text(encoding="utf-8"))
manifest_bytes = (BUNDLE / "manifest.json").read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == evaluated["bundle"]["manifest_sha256"], (
    "not the evaluated bundle"
)
manifest = json.loads(manifest_bytes)
REFER, VERSION = float(manifest["refer_threshold"]), manifest["version"]
assert spark.catalog.tableExists(AUDIT), f"{AUDIT} does not exist"
assert not spark.catalog.tableExists(TABLE), f"{TABLE} exists already: the backfill runs once"

# COMMAND ----------

# The replay's audit rows with their labels; a Spark join, so lineage reaches the new table.
rows = (
    spark.read.table(AUDIT)
    .where(F.col("month").isin(MONTHS))
    .join(
        spark.read.table("workspace.cdm.silver_images").select("isic_id", "label"),
        "isic_id",
        "left",
    )
    .select(
        "month", "isic_id", "model_version", "refer_melanoma", "label", *[f"p_{c}" for c in CLASSES]
    )  # fmt: skip
    .orderBy("month", "isic_id")
    .toPandas()
)
if sorted(rows["month"].unique()) != MONTHS:
    raise RuntimeError(f"audit months {sorted(rows['month'].unique())}, expected 1 to 12")
if rows.duplicated(["month", "isic_id"]).any():
    raise RuntimeError("an image appears twice in one month: the silver join is not one-to-one")
if set(rows["model_version"]) != {VERSION}:
    raise RuntimeError(f"audit rows were scored by {set(rows['model_version'])}, not {VERSION}")

records = []
for month in MONTHS:
    part = rows[rows["month"] == month]
    labels = np.array(
        [INDEX.get(x, -1) if isinstance(x, str) else -1 for x in part["label"]], dtype=np.int64
    )
    probs = part[[f"p_{c}" for c in CLASSES]].to_numpy(np.float64)
    records.append(
        {
            **month_record(month, labels, probs, part["refer_melanoma"].to_numpy(bool), REFER),
            "audit_rows": int(len(part)),
        }
    )
committed = committed_records(
    json.loads((REPO / f).read_text(encoding="utf-8")) for f in committed_files
)
checks = check_committed(records, committed)  # raises on any difference

# COMMAND ----------

now = datetime.now(UTC).isoformat(timespec="seconds")
table = pd.DataFrame(
    [
        {
            "month": r["month"],
            "labelled": r["labelled"],
            "melanoma_images": r["melanoma_images"],
            "melanoma_referred": r["melanoma_referred"],
            "melanoma_correct": r["melanoma_correct"],
            "balanced_accuracy": r["balanced_accuracy"],
            "audit_rows": r["audit_rows"],
            "check": checks[r["month"]],
            "bundle_version": VERSION,
            "code_version": CODE,
            "job_run": JOB_RUN,
            "built_utc": now,
        }
        for r in records
    ]
)
# The per-month row counts come from the audit table, so lineage reaches it and a count that
# changed between the read above and this write fails the run.
counts = spark.read.table(AUDIT).where(F.col("month").isin(MONTHS)).groupBy("month").count()
out = spark.createDataFrame(table).join(counts, "month", "inner")
if out.where(F.col("count") != F.col("audit_rows")).count() or out.count() != len(MONTHS):
    raise RuntimeError("audit row counts changed during the backfill")
out.drop("count").write.mode("errorifexists").saveAsTable(TABLE)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "melanoma_backfill.json").write_text(
    json.dumps(
        {
            "what": "replay months 1 to 12 rebuilt from gold_prediction_audit and silver labels "
            "(nothing re-scored), with the melanoma referral count; real images in an order we "
            "chose, not drift over time",
            "built_by": "databricks/notebooks/melanoma_backfill.py",
            "code_version": CODE,
            "job_run": JOB_RUN,
            "bundle_version": VERSION,
            "refer_threshold": REFER,
            "months": [{**r, "check": checks[r["month"]]} for r in records],
        },
        indent=2,
    )
    + "\n"
)
for r in records:
    print(r["month"], r["melanoma_images"], r["melanoma_referred"], checks[r["month"]])
