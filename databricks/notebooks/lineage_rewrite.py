# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: rewrite the audit and drift tables through Spark, for lineage (serverless CPU)
# MAGIC Under `results/phase3/lineage/spark_writes_spec.json`. The monthly job wrote
# MAGIC `gold_prediction_audit`, `gold_drift_baseline` and `gold_drift_monthly` from pandas, so Unity
# MAGIC Catalog lineage records them with no source table. This run rewrites each table with its own
# MAGIC rows joined to the tables they come from, checks the rows are unchanged, and restores every
# MAGIC table to its recorded version if any check fails. SQL only; no model, no rescoring.

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/lineage",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
SILVER = "workspace.cdm.silver_images"
AUDIT, BASELINE, MONTHLY = (
    f"workspace.cdm.{t}"
    for t in ("gold_prediction_audit", "gold_drift_baseline", "gold_drift_monthly")
)
MONTHLY_COLUMNS = "month, run_utc, alert, live_count, summary"

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
for rel, digest in json.loads(dbutils.widgets.get("files_sha256")).items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"


def one(statement: str) -> int:
    return int(spark.sql(statement).first()[0])


def version(table: str) -> int:
    return int(spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"])


def differences(table: str, before: int, columns: str = "*") -> dict:
    old = f"SELECT {columns} FROM {table} VERSION AS OF {before}"
    new = f"SELECT {columns} FROM {table}"
    return {
        "rows_before": one(f"SELECT COUNT(*) FROM ({old})"),
        "rows_after": one(f"SELECT COUNT(*) FROM ({new})"),
        "only_before": one(f"SELECT COUNT(*) FROM ({old} EXCEPT ALL {new})"),
        "only_after": one(f"SELECT COUNT(*) FROM ({new} EXCEPT ALL {old})"),
    }


def unchanged(d: dict) -> bool:
    return d["rows_before"] == d["rows_after"] and d["only_before"] == 0 and d["only_after"] == 0


# COMMAND ----------

before = {t: version(t) for t in (AUDIT, BASELINE, MONTHLY)}
assert one(f"SELECT COUNT(*) FROM (SELECT isic_id FROM {SILVER} GROUP BY isic_id HAVING COUNT(*) > 1)") == 0, "duplicate ISIC ID in silver"  # fmt: skip
checks: dict = {}
try:
    for table in (AUDIT, BASELINE):
        spark.sql(
            f"INSERT OVERWRITE {table} SELECT t.* FROM {table} t "
            f"INNER JOIN (SELECT isic_id FROM {SILVER}) s ON t.isic_id = s.isic_id"
        )
        checks[table] = differences(table, before[table])
    spark.sql(f"ALTER TABLE {MONTHLY} ADD COLUMNS (audit_rows BIGINT, baseline_rows BIGINT)")
    spark.sql(
        f"""
        INSERT OVERWRITE {MONTHLY}
        SELECT m.month, m.run_utc, m.alert, m.live_count, m.summary,
               COALESCE(a.n, 0) AS audit_rows, b.n AS baseline_rows
        FROM {MONTHLY} m
        LEFT JOIN (SELECT month, COUNT(*) AS n FROM {AUDIT} GROUP BY month) a
          ON a.month = m.month
        LEFT JOIN (SELECT bundle_version, COUNT(*) AS n FROM {BASELINE} GROUP BY bundle_version) b
          ON b.bundle_version = get_json_object(m.summary, '$.bundle_version')
        """
    )
    checks[MONTHLY] = differences(MONTHLY, before[MONTHLY], MONTHLY_COLUMNS)
    rows = spark.sql(f"SELECT month, summary, audit_rows, baseline_rows FROM {MONTHLY}").collect()
    mismatched = []
    for r in rows:
        s = json.loads(r["summary"])
        if r["audit_rows"] != sum(s.get("by_source", {}).values()) or r["baseline_rows"] != s["sizes"]["baseline_images"]:  # fmt: skip
            mismatched.append(int(r["month"]))
    checks[MONTHLY]["count_columns_match_summary"] = not mismatched
    checks[MONTHLY]["months_mismatched"] = mismatched
    ok = all(unchanged(checks[t]) for t in (AUDIT, BASELINE, MONTHLY)) and not mismatched
    if not ok:
        raise RuntimeError(f"a check failed: {json.dumps(checks)}")
except Exception:
    for table, v in before.items():
        if version(table) != v:
            spark.sql(f"RESTORE TABLE {table} TO VERSION AS OF {v}")
    raise

# COMMAND ----------

out = {
    "spec": "results/phase3/lineage/spark_writes_spec.json",
    "versions_before": before,
    "versions_after": {t: version(t) for t in before},
    "checks": checks,
    "all_checks_passed": True,
    "code_version": CODE,
    "ran_on": f"Databricks serverless CPU, job run {JOB_RUN}",
    "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "spark_writes.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
