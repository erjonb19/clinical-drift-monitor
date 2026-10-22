"""Per-run cost tracking (Phase 3): every job run's DBUs and run time, from Databricks system
tables, written to ``workspace.cdm.gold_job_costs``, with a per-job summary committed to
``results/phase3/costs/job_costs.json``.

The SQL runs on the workspace's SQL warehouse, so nothing heavy runs locally. Databricks Free
Edition charges nothing; ``list_price_usd`` is the list price of the same DBUs, the cost the
run would have had on a paid workspace. Billing records arrive with a delay, so the newest runs
can be missing until a later refresh.

Usage: python scripts/cost_table.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, records  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase3" / "costs" / "job_costs.json"
TABLE = "workspace.cdm.gold_job_costs"

BUILD = f"""
CREATE OR REPLACE TABLE {TABLE}
COMMENT 'Per job run DBUs, run time and list-price equivalent, from system.billing and
system.lakeflow. Free Edition charges nothing.'
AS
WITH usage AS (
  SELECT u.usage_metadata.job_id AS job_id, u.usage_metadata.job_run_id AS job_run_id,
         u.sku_name, u.usage_unit, u.usage_quantity, u.usage_start_time, u.usage_end_time,
         p.pricing.default AS unit_price
  FROM system.billing.usage u
  LEFT JOIN system.billing.list_prices p
    ON p.sku_name = u.sku_name AND p.usage_unit = u.usage_unit
   AND u.usage_start_time >= p.price_start_time
   AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
  WHERE u.usage_metadata.job_run_id IS NOT NULL
),
names AS (
  SELECT job_id, name, ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY change_time DESC) AS rn
  FROM system.lakeflow.jobs
),
runs AS (
  SELECT job_id, run_id, MIN(period_start_time) AS run_start, MAX(period_end_time) AS run_end,
         MAX_BY(result_state, period_end_time) AS result_state
  FROM system.lakeflow.job_run_timeline GROUP BY job_id, run_id
)
SELECT u.job_id, n.name AS job_name, u.job_run_id,
       CASE WHEN u.sku_name LIKE '%MODEL_TRAINING%' THEN 'gpu' ELSE 'cpu' END AS compute,
       u.sku_name, ROUND(SUM(u.usage_quantity), 6) AS dbus,
       ROUND(SUM(u.usage_quantity * CAST(u.unit_price AS DOUBLE)), 6) AS list_price_usd,
       r.run_start, r.run_end,
       ROUND((UNIX_TIMESTAMP(r.run_end) - UNIX_TIMESTAMP(r.run_start)) / 60.0, 2) AS run_minutes,
       r.result_state
FROM usage u
LEFT JOIN names n ON n.job_id = u.job_id AND n.rn = 1
LEFT JOIN runs r ON r.job_id = u.job_id AND r.run_id = u.job_run_id
GROUP BY u.job_id, n.name, u.job_run_id, u.sku_name, r.run_start, r.run_end, r.result_state
"""

SUMMARY = f"""
SELECT job_name, compute, COUNT(DISTINCT job_run_id) AS runs, ROUND(SUM(dbus), 4) AS dbus,
       ROUND(SUM(list_price_usd), 4) AS list_price_usd, ROUND(SUM(run_minutes), 1) AS run_minutes
FROM {TABLE} GROUP BY job_name, compute ORDER BY dbus DESC
"""


NUMERIC = ("runs", "dbus", "list_price_usd", "run_minutes")


def typed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """The SQL API returns every value as text: numbers back to numbers, and a missing job name
    (a one-off run, or a job since deleted) said as such."""
    out = []
    for row in rows:
        fixed = {
            k: (float(str(v)) if k in NUMERIC and v is not None else v) for k, v in row.items()
        }
        if fixed.get("job_name") is None:
            fixed["job_name"] = "unnamed (one-off run or deleted job)"
        out.append(fixed)
    return out


def main() -> None:
    sql = Sql()
    sql(BUILD)
    by_job = typed(records(sql(SUMMARY)))
    columns = "job_name, job_run_id, compute, dbus, list_price_usd, run_minutes, result_state"
    runs = typed(records(sql(f"SELECT {columns} FROM {TABLE} ORDER BY run_start")))
    newest = sql("SELECT MAX(usage_end_time) FROM system.billing.usage").iloc[0, 0]
    out = {
        "table": TABLE,
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "billing_records_up_to": newest,
        "note": "Databricks Free Edition charges nothing; list_price_usd is the list price of the "
        "same DBUs. Billing arrives with a delay, so the newest runs may be missing. A run that "
        "used GPU and CPU appears under both, with its full run time each time.",
        "by_job": by_job,
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}: {len(runs)} job runs")


if __name__ == "__main__":
    main()
