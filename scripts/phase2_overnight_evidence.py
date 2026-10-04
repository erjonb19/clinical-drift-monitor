"""Save the Phase 2 overnight job run as evidence in results/phase2/overnight/.

Writes job_run.json (every task attempt and both tasks' exit values), final_test.json
(the test task's exit value) and copies the GPU task's per-run result files, manifests and
summary from the volume.

Usage: python scripts/phase2_overnight_evidence.py --run <job run id>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import cli, cli_text  # noqa: E402

VOLUME = "dbfs:/Volumes/workspace/cdm/raw/results/phase2/overnight"
OUT = Path("results/phase2/overnight")


def task_outputs(run_id: str) -> dict[str, Any]:
    run = cli("jobs", "get-run", run_id)
    attempts = [
        {"task": t["task_key"], "task_run_id": t["run_id"], "attempt": t.get("attempt_number", 0),
         "result": t["state"].get("result_state"), "message": t["state"].get("state_message"),
         "execution_ms": t.get("execution_duration"), "setup_ms": t.get("setup_duration")}
        for t in run["tasks"]
    ]  # fmt: skip
    outputs: dict[str, Any] = {}
    for key in sorted({t["task_key"] for t in run["tasks"]}):
        last = max(
            (t for t in run["tasks"] if t["task_key"] == key),
            key=lambda t: t.get("attempt_number", 0),
        )
        result = cli("jobs", "get-run-output", str(last["run_id"])).get("notebook_output", {})
        outputs[key] = json.loads(result["result"]) if "result" in result else None
    return {"run_id": run["run_id"], "job_id": run.get("job_id"),
            "result": run["state"].get("result_state"), "attempts": attempts,
            "outputs": outputs}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = task_outputs(args.run)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    test = evidence["outputs"].pop("cpu_test")
    (OUT / "job_run.json").write_text(
        json.dumps({"exported_utc": stamp, **evidence}, indent=2) + "\n", encoding="utf-8"
    )
    if test is not None:
        (OUT / "final_test.json").write_text(
            json.dumps({"exported_utc": stamp, "job_run_id": evidence["run_id"], **test},
                       indent=2) + "\n", encoding="utf-8"
        )  # fmt: skip
    listing = cli("fs", "ls", VOLUME)
    names = [e["name"] for e in listing]
    for name in names:
        if name.endswith(".json"):
            text = cli_text("fs", "cat", f"{VOLUME}/{name}")
            (OUT / name).write_text(json.dumps(json.loads(text), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(names)} volume files)")


if __name__ == "__main__":
    main()
