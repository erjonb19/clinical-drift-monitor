"""Export Phase 2 stage 2 evidence: the centralized run, its re-evaluation, and GPU billing.

Saves the training notebook's and the evaluation notebook's exit values as returned by
Databricks, plus every serverless GPU billing record (MODEL_TRAINING SKU) per job run.

Usage: python scripts/phase2_stage2_evidence.py --train-run <id> --eval-run <id>
Writes results/phase2/stage2_centralized.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql, cli, records  # noqa: E402

OUT = Path("results/phase2/stage2_centralized.json")
GPU_BILLING = """
SELECT usage_date, sku_name, usage_metadata.job_run_id AS job_run_id,
       round(sum(usage_quantity), 4) AS dbu,
       min(usage_start_time) AS first_window, max(usage_end_time) AS last_window
FROM system.billing.usage
WHERE sku_name LIKE '%MODEL_TRAINING%'
GROUP BY ALL ORDER BY first_window"""


def notebook_output(run_id: str) -> dict[str, Any]:
    """Exit value of the run's last task attempt, plus every attempt's outcome."""
    run = cli("jobs", "get-run", run_id)
    attempts = [
        {"task_run_id": t["run_id"], "attempt": t.get("attempt_number", 0),
         "result": t["state"].get("result_state"), "message": t["state"].get("state_message"),
         "execution_ms": t.get("execution_duration"), "setup_ms": t.get("setup_duration")}
        for t in run["tasks"]
    ]  # fmt: skip
    last = max(run["tasks"], key=lambda t: t.get("attempt_number", 0))
    output = cli("jobs", "get-run-output", str(last["run_id"]))
    return {
        "run_id": run["run_id"],
        "result": run["state"].get("result_state"),
        "attempts": attempts,
        "output": json.loads(output["notebook_output"]["result"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--eval-run", required=True)
    args = parser.parse_args()
    evidence = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "training": notebook_output(args.train_run),
        "re_evaluation": notebook_output(args.eval_run),
        "gpu_billing_by_job_run": records(Sql()(GPU_BILLING)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
