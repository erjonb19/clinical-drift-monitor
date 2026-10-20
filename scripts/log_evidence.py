"""Log committed results files to Databricks MLflow and record the runs in
results/mlflow_runs.json (cdm.evidence explains the rule).

Refuses a file that is not committed or has local changes: the run must describe exactly
what the repository holds. A file already indexed with the same hash is skipped; one whose
hash changed is refused, because its earlier run would then describe a different file.

Usage: python scripts/log_evidence.py results/phase2/triage/test.json [...]
           [--job-run ID] [--after-the-fact] [--note TEXT]
Needs ``pip install -e ".[evidence]"`` and a logged-in Databricks CLI profile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from cdm.evidence import INDEX, load_index, numeric_leaves, sha256

REPO = Path(__file__).resolve().parents[1]
BATCH = 900  # MLflow accepts at most 1,000 metrics per batch call


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--job-run", help="Databricks job run that produced the file(s)")
    parser.add_argument(
        "--after-the-fact",
        action="store_true",
        help="the file was produced before MLflow logging was required",
    )
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # optional dependency: the "evidence" extra
    from mlflow.entities import Metric

    mlflow.set_tracking_uri("databricks")
    me = json.loads(
        subprocess.run(
            ["databricks", "current-user", "me", "--output", "json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout  # fmt: skip
    )["userName"]
    mlflow.set_experiment(f"/Users/{me}/cdm-evidence")
    client = mlflow.MlflowClient()
    index = load_index(REPO)

    for file in args.files:
        name = file.resolve().relative_to(REPO).as_posix()
        if git("status", "--porcelain", "--", name):
            sys.exit(f"{name}: not committed or has local changes")
        digest = sha256(REPO / name)
        if name in index:
            if index[name]["file_sha256"] == digest:
                print(f"{name}: already logged as run {index[name]['run_id']}")
                continue
            sys.exit(f"{name}: changed since run {index[name]['run_id']} logged it")
        metrics = numeric_leaves(json.loads((REPO / name).read_text(encoding="utf-8")))
        commit = git("log", "-1", "--format=%h", "--", name)
        tags = {"source_file": name, "file_sha256": digest, "file_commit": commit,
                "logged_after_the_fact": str(args.after_the_fact).lower()}  # fmt: skip
        if args.job_run:
            tags["databricks_job_run"] = args.job_run
        if args.note:
            tags["note"] = args.note
        with mlflow.start_run(run_name=name, tags=tags) as run:
            now = int(datetime.now(UTC).timestamp() * 1000)
            items = [Metric(k, v, now, 0) for k, v in sorted(metrics.items())]
            for i in range(0, len(items), BATCH):
                client.log_batch(run.info.run_id, metrics=items[i : i + BATCH])
            mlflow.log_artifact(str(REPO / name))
        index[name] = {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id,
                       "file_sha256": digest, "file_commit": commit, "metrics": len(metrics),
                       "logged_after_the_fact": args.after_the_fact,
                       "databricks_job_run": args.job_run}  # fmt: skip
        print(f"{name}: run {run.info.run_id} ({len(metrics)} metrics)")
    (REPO / INDEX).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", "utf-8")


if __name__ == "__main__":
    main()
