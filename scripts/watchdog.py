"""Backlog item 1's watchdog: fail loudly when the monthly drift job missed its scheduled run.

The 1 November 2026 run never started, with no error and no email, because Databricks job
notifications only cover runs that start. This script asks the Jobs API, with a view-only
service principal, whether the job has a run started by its schedule since 06:00 UTC on the 1st
of the month, and exits:

- 0: a scheduled run started (its state is printed; a failed run already emails the owner);
- 1: no scheduled run started: the run was missed;
- 2: inconclusive: no credential configured, or the API could not be read.

Usage: python scripts/watchdog.py [--month YYYY-MM] [--job-id ID]
Credentials come from the environment only: DATABRICKS_HOST, DATABRICKS_CLIENT_ID,
DATABRICKS_CLIENT_SECRET (OAuth machine-to-machine). Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

JOB_ID = 524433148303929  # cdm-phase3-monthly-drift (owner confirmed, 2026-11-09)
SCHEDULE_HOUR_UTC = 6  # databricks/monthly_drift_job.json: 0 0 6 1 * ? UTC
OK, MISSED, INCONCLUSIVE = 0, 1, 2
Fetch = Callable[[urllib.request.Request], dict[str, Any]]


def window_start(month: str) -> datetime:
    """06:00 UTC on the 1st of the month (YYYY-MM): when the scheduled run is due."""
    year, mon = (int(x) for x in month.split("-"))
    return datetime(year, mon, 1, SCHEDULE_HOUR_UTC, tzinfo=UTC)


def http_json(request: urllib.request.Request) -> dict[str, Any]:
    with urllib.request.urlopen(request, timeout=60) as response:
        data: dict[str, Any] = json.load(response)
    return data


def token(host: str, client_id: str, secret: str, fetch: Fetch) -> str:
    basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
    body = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": "all-apis"})
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    request = urllib.request.Request(
        f"{host}/oidc/v1/token", data=body.encode(), headers=headers, method="POST"
    )
    return str(fetch(request)["access_token"])


def scheduled_runs(
    host: str, bearer: str, job_id: int, since: datetime, fetch: Fetch
) -> list[dict[str, Any]]:
    """Runs of the job started since `since` whose trigger is the schedule (PERIODIC)."""
    query = urllib.parse.urlencode(
        {"job_id": job_id, "start_time_from": int(since.timestamp() * 1000), "limit": 25}
    )
    request = urllib.request.Request(
        f"{host}/api/2.2/jobs/runs/list?{query}", headers={"Authorization": f"Bearer {bearer}"}
    )
    runs = fetch(request).get("runs", [])
    return [r for r in runs if r.get("trigger") == "PERIODIC"]


def run_state(run: dict[str, Any]) -> str:
    state = run.get("state") or {}
    return str(state.get("result_state") or state.get("life_cycle_state") or "unknown")


def check(month: str, job_id: int, env: dict[str, str], fetch: Fetch) -> tuple[int, str]:
    missing = [
        k
        for k in ("DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET")
        if not env.get(k)
    ]
    if missing:
        return INCONCLUSIVE, f"watchdog not configured: {', '.join(missing)} not set"
    host = env["DATABRICKS_HOST"].rstrip("/")
    since = window_start(month)
    try:
        bearer = token(host, env["DATABRICKS_CLIENT_ID"], env["DATABRICKS_CLIENT_SECRET"], fetch)
        runs = scheduled_runs(host, bearer, job_id, since, fetch)
    except Exception as err:  # noqa: BLE001 - any API failure is inconclusive, never a pass
        return INCONCLUSIVE, f"could not read the Jobs API: {type(err).__name__}: {err}"
    if not runs:
        return MISSED, (
            f"MISSED: job {job_id} has no run started by its schedule since "
            f"{since.isoformat()} (month {month})"
        )
    states = ", ".join(f"run {r.get('run_id')} {run_state(r)}" for r in runs)
    return OK, f"ok: job {job_id} started by its schedule since {since.isoformat()}: {states}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default=datetime.now(UTC).strftime("%Y-%m"))
    parser.add_argument("--job-id", type=int, default=JOB_ID)
    args = parser.parse_args(argv)
    code, message = check(args.month, args.job_id, dict(os.environ), http_json)
    prefix = {
        OK: "",
        MISSED: "::error title=Scheduled run missed::",
        INCONCLUSIVE: "::error title=Watchdog inconclusive::",
    }[code]
    print(prefix + message)
    return code


if __name__ == "__main__":
    sys.exit(main())
