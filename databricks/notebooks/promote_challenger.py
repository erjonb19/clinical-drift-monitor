# Databricks notebook source
# MAGIC %md
# MAGIC # Backlog item 17: promote a challenger (run by hand by the owner: the human approval)
# MAGIC Moves `workspace.cdm.skin_lesion_classifier@champion` to a registered challenger, and only
# MAGIC when: the decision file is the one the owner names by SHA-256, its rule promoted, the
# MAGIC registry version was registered from that decision with a checked serving bundle
# MAGIC (`cdm.pipeline.verify_promotion`), and the move itself is safe (`cdm.registry.plan_move`:
# MAGIC the alias is where the owner expects, the version is tagged and not rejected). It reads the
# MAGIC alias back and appends the move to `gold_model_decisions`. A rollback is a separate,
# MAGIC simpler step (`scripts/move_champion.py`, `docs/rollback.md`).

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "version": "",
    "expect_current": "",
    "decision_path": "",
    "decision_sha256": "",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

from cdm.pipeline import verify_promotion
from cdm.registry import ALIAS, NAME, move_record, plan_move

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
VERSION = int(dbutils.widgets.get("version"))
EXPECT = int(dbutils.widgets.get("expect_current"))
SHA = dbutils.widgets.get("decision_sha256").strip().lower()
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"

client = mlflow.MlflowClient(registry_uri="databricks-uc")
tags = dict(client.get_model_version(NAME, str(VERSION)).tags or {})
decision = verify_promotion(Path(dbutils.widgets.get("decision_path")).read_bytes(), SHA, tags)
# Unity Catalog's search results carry no tags, so each version's tags are read one by one.
numbers = sorted({int(v.version) for v in client.search_model_versions(f"name='{NAME}'")})
versions = [
    {"version": n, "tags": dict(client.get_model_version(NAME, str(n)).tags or {})} for n in numbers
]
current = int(client.get_model_version_by_alias(NAME, ALIAS).version)
move = plan_move(versions, current, EXPECT, VERSION, SHA)

# COMMAND ----------

client.set_registered_model_alias(NAME, ALIAS, move.to_version)
after = int(client.get_model_version_by_alias(NAME, ALIAS).version)
if after != move.to_version:
    raise RuntimeError(f"the alias reads v{after} after the move, not v{move.to_version}")
now = datetime.now(UTC).isoformat(timespec="seconds")
record = move_record(
    move, who=f"promote_challenger job run {JOB_RUN}", when_utc=now,
    check="pending: run check_registered_model with the challenger's serving bundle",
)  # fmt: skip
spark.createDataFrame(
    pd.DataFrame(
        [
            {
                "decided_utc": now,
                "job_run": JOB_RUN,
                "promoted": True,
                "held_out_gain_points": float(decision.get("held_out_gain_points", "nan")),
                "client_val_change_points": float(decision.get("client_val_change_points", "nan")),
                "record": json.dumps({"alias_move": record, "decision_sha256": SHA}),
            }
        ]
    )  # fmt: skip
).write.mode("append").saveAsTable("workspace.cdm.gold_model_decisions")
print(json.dumps(record))
