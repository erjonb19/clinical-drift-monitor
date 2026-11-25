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


def test_the_bottleneck_is_named_from_measured_cpu() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("load_test", REPO / "scripts" / "load_test.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    busy = [
        {"users": 1, "server_cpu_cores_used": 1.0, "machine_cpu_busy_share": 0.5},
        {"users": 20, "server_cpu_cores_used": 3.6, "machine_cpu_busy_share": 0.97},
    ]
    idle = [{"users": 20, "server_cpu_cores_used": 1.2, "machine_cpu_busy_share": 0.4}]
    assert module.bottleneck(busy, 4)["named"] == "CPU compute"
    assert module.bottleneck(idle, 4)["named"].startswith("not CPU")
    assert "3.6 of 4 cores" in module.bottleneck(busy, 4)["evidence"]
    assert module.process_cpu_s(1) >= 0 and module.machine_cpu()[1] > 0


def test_the_champion_cpu_run_names_a_measured_bottleneck() -> None:
    d = json.loads((REPO / "results/phase3/serving/benchmark_champion_cpu.json").read_text())
    assert d["bundle_version"] == "v2-ensemble-cpu-d7191f4" and d["networks_in_bundle"] == 3
    assert d["code_version"] and d["uncommitted_changes"] is False and d["caveat"] == ""
    cores = d["hardware"]["logical_cpus"]
    for lv in d["levels"]:
        assert lv["status_codes"] == {"200": lv["requests"]}
        assert 0 < lv["server_cpu_cores_used"] <= cores + 0.1
        assert 0 < lv["machine_cpu_busy_share"] <= 1
    assert d["bottleneck"]["named"] == "CPU compute"
    assert f"of {cores} cores" in d["bottleneck"]["evidence"]
