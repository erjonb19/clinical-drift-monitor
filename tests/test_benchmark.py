"""Backlog item 13: the committed serving benchmark says what ran and that it is not accuracy."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_benchmark_names_its_hardware_model_and_limits() -> None:
    d = json.loads((REPO / "results" / "phase3" / "serving" / "benchmark.json").read_text())
    assert d["bundle_version"] == "ci-test-bundle-untrained"
    assert "never accuracy" in d["caveat"] and "not accuracy" in d["what"]
    assert d["hardware"]["cpu"] and d["hardware"]["logical_cpus"] and d["where"]
    assert d["software"]["torch"] and d["conditions"]
    assert d["networks_in_bundle"] == 1  # written by the script; the champion runs three
    hand = d["annotations_by_hand"]
    assert "not script output" in hand["note"]
    assert {"not_the_champion", "differs_from_the_api_image", "unreconciled_with_phase_3"} <= set(
        hand
    )
    assert d["code_version"] and d["uncommitted_changes"] is False
    assert [lv["users"] for lv in d["levels"]] == [1, 5, 20]
    for lv in d["levels"]:
        assert lv["status_codes"] == {"200": lv["requests"]}  # every request scored, none limited
        assert lv["latency_ms"]["p50"] <= lv["latency_ms"]["p95"] <= lv["latency_ms"]["p99"]


def test_the_benchmark_workflow_runs_by_hand_only() -> None:
    text = (REPO / ".github" / "workflows" / "benchmark.yml").read_text("utf-8")
    assert "workflow_dispatch" in text and "push:" not in text and "pull_request" not in text
