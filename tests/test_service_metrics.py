"""Service metrics: turnaround, reuse and refusals computed from fixtures; model rows from
the committed results files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]


def metrics() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sm", REPO / "scripts" / "service_metrics.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def issue(created: str, closed: str | None, *labels: str) -> dict[str, object]:
    return {"createdAt": created, "closedAt": closed,
            "labels": [{"name": n} for n in ("data-request", *labels)]}  # fmt: skip


def test_turnaround_separates_synthetic_requests() -> None:
    m = metrics()
    got = m.request_metrics([
        issue("2026-10-01T00:00:00Z", "2026-10-03T00:00:00Z"),
        issue("2026-10-01T00:00:00Z", "2026-10-07T00:00:00Z"),
        issue("2026-10-01T00:00:00Z", None),
        issue("2026-10-01T00:00:00Z", "2026-10-02T12:00:00Z", "synthetic"),
    ])  # fmt: skip
    assert got["real"] == {"opened": 3, "closed": 2, "median_days_to_close": 4.0,
                           "max_days_to_close": 6.0}  # fmt: skip
    assert got["synthetic"]["median_days_to_close"] == 1.5
    page = m.render(got, None, None, {"rows_by_match": {"exact": 1}}, [])
    assert "says nothing about a real service" in page


def test_reuse_refusals_and_time_to_approval_from_the_audit_log() -> None:
    def e(ts: str, event: str, **kw: object) -> dict[str, object]:
        return {"ts": f"2026-10-21T{ts}:00+00:00", "event": event, "phenotype": "p",
                "version": 1, "sha256": "h"} | kw  # fmt: skip

    got = metrics().cohort_metrics([
        e("10:00", "refused", reason="definition not approved (pending)"),
        e("12:30", "decision", decision="approve"),
        e("13:00", "ran"),
        e("14:00", "ran"),
    ])  # fmt: skip
    assert got["runs_by_phenotype"] == {"p v1": 2}
    assert got["refusals_by_reason"] == {"definition not approved": 1}
    assert got["median_hours_to_approval"] == 2.5


def test_model_rows_read_committed_files() -> None:
    m = metrics()
    spec = json.loads((REPO / "config" / "service_metrics.json").read_text(encoding="utf-8"))
    rows = m.model_rows(spec)
    assert len(rows) == len(spec["model_performance"])
    assert all(r["low"] <= r["value"] <= r["high"] for r in rows)
    page = m.render(None, None, None, {"rows_by_match": {"exact": 1}}, rows)
    assert "Not read" in page and "| Model |" in page
