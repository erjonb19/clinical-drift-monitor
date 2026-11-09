"""Backlog item 1's watchdog: a missed scheduled run fails loudly, never as a pass."""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV = {
    "DATABRICKS_HOST": "https://x.example/",
    "DATABRICKS_CLIENT_ID": "id",
    "DATABRICKS_CLIENT_SECRET": "s",
}


def watchdog() -> ModuleType:
    spec = importlib.util.spec_from_file_location("watchdog", REPO / "scripts" / "watchdog.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake(runs: list[dict[str, Any]], seen: list[urllib.request.Request]) -> Any:
    def fetch(request: urllib.request.Request) -> dict[str, Any]:
        seen.append(request)
        if request.full_url.endswith("/oidc/v1/token"):
            return {"access_token": "t"}
        return {"runs": runs}

    return fetch


def test_window_runs_from_this_schedule_to_the_next() -> None:
    w = watchdog().window
    assert w("2026-12") == (
        datetime(2026, 12, 1, 6, tzinfo=UTC),
        datetime(2027, 1, 1, 6, tzinfo=UTC),
    )
    assert w("2026-11")[1] == datetime(2026, 12, 1, 6, tzinfo=UTC)


def test_the_window_matches_the_committed_schedule() -> None:
    job = json.loads((REPO / "databricks" / "monthly_drift_job.json").read_text("utf-8"))
    cron = job["schedule"]["quartz_cron_expression"].split()
    assert cron[2] == str(watchdog().SCHEDULE_HOUR_UTC) and cron[3] == "1"
    assert job["schedule"]["timezone_id"] == "UTC"


def test_no_scheduled_run_is_missed() -> None:
    """A manual run (ONE_TIME), like the 22 October backfill, does not count."""
    seen: list[urllib.request.Request] = []
    runs = [{"run_id": 1, "trigger": "ONE_TIME", "state": {"result_state": "SUCCESS"}}]
    code, message = watchdog().check("2026-11", 524433148303929, ENV, fake(runs, seen))
    assert code == 1 and message.startswith("MISSED")
    assert "start_time_from=1793512800000" in seen[1].full_url  # 2026-11-01T06:00Z
    assert "start_time_to=1796104799999" in seen[1].full_url  # just before 2026-12-01T06:00Z
    assert "job_id=524433148303929" in seen[1].full_url


def test_a_scheduled_run_passes_and_reports_its_state() -> None:
    runs = [{"run_id": 9, "trigger": "PERIODIC", "state": {"result_state": "FAILED"}}]
    code, message = watchdog().check("2026-12", 524433148303929, ENV, fake(runs, []))
    assert code == 0 and "run 9 FAILED" in message


def test_no_credential_is_inconclusive_not_a_pass() -> None:
    code, message = watchdog().check("2026-12", 1, {}, fake([], []))
    assert code == 2 and "not configured" in message


def test_an_api_error_is_inconclusive_not_a_pass() -> None:
    def broken(request: urllib.request.Request) -> dict[str, Any]:
        raise OSError("connection refused")

    code, message = watchdog().check("2026-12", 1, ENV, broken)
    assert code == 2 and "could not read" in message


def test_the_token_request_uses_client_credentials() -> None:
    seen: list[urllib.request.Request] = []
    watchdog().check("2026-12", 1, ENV, fake([], seen))
    assert seen[0].full_url == "https://x.example/oidc/v1/token"
    assert seen[0].get_method() == "POST"
    body = seen[0].data
    assert isinstance(body, bytes) and b"grant_type=client_credentials" in body


def test_the_workflow_runs_on_the_2nd_and_never_on_push() -> None:
    text = (REPO / ".github" / "workflows" / "drift-watchdog.yml").read_text("utf-8")
    assert 'cron: "17 7 2 * *"' in text and "workflow_dispatch" in text
    assert "push:" not in text and "pull_request" not in text
    assert "permissions:\n  contents: read" in text


def test_every_page_is_read() -> None:
    """The scheduled run is the oldest in the window, so it can sit on a later page."""
    pages: list[dict[str, Any]] = [
        {"runs": [{"run_id": 2, "trigger": "ONE_TIME"}], "next_page_token": "p2"},
        {"runs": [{"run_id": 1, "trigger": "PERIODIC", "state": {"result_state": "SUCCESS"}}]},
    ]
    seen: list[urllib.request.Request] = []

    def fetch(request: urllib.request.Request) -> dict[str, Any]:
        seen.append(request)
        if request.full_url.endswith("/oidc/v1/token"):
            return {"access_token": "t"}
        return pages.pop(0)

    code, message = watchdog().check("2026-12", 1, ENV, fetch)
    assert code == 0 and "run 1 SUCCESS" in message
    assert "page_token=p2" in seen[2].full_url


def test_a_skipped_scheduled_run_is_missed() -> None:
    runs = [{"run_id": 5, "trigger": "PERIODIC", "state": {"life_cycle_state": "SKIPPED"}}]
    code, message = watchdog().check("2026-12", 1, ENV, fake(runs, []))
    assert code == 1 and "1 skipped" in message


def test_a_bad_month_is_inconclusive_with_a_message() -> None:
    code, message = watchdog().check("2026-13", 1, ENV, fake([], []))
    assert code == 2 and "bad month" in message


def test_main_annotates_and_exits_with_the_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    assert watchdog().main(["--month", "2026-12"]) == 2
    assert capsys.readouterr().out.startswith("::error title=Watchdog inconclusive::")


def test_the_token_request_sends_basic_auth() -> None:
    seen: list[urllib.request.Request] = []
    watchdog().check("2026-12", 1, ENV, fake([], seen))
    assert seen[0].get_header("Authorization") == "Basic aWQ6cw=="  # base64 of "id:s"
