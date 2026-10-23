"""The CI accuracy gate (Phase 3): the shipped model's committed metrics must stay above the floors
in config/eval_gate.json, and a promoted challenger must meet the same floors. Reads committed
results only, so it runs in CI without a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GATE = json.loads((REPO / "config" / "eval_gate.json").read_text(encoding="utf-8"))
FLOORS = GATE["floors"]


def serving_metrics(results: dict[str, Any]) -> dict[str, float]:
    test = results["test_look"]
    return {
        "client_test_balanced_accuracy": test["client_test"]["balanced_accuracy_all"],
        "buenos_aires_balanced_accuracy": test["buenos_aires"]["balanced_accuracy_all"],
        "pad_ufes_balanced_accuracy": test["pad_ufes"]["balanced_accuracy_all"],
        "validation_review_workload_max": results["validation"]["workload_share_above_threshold"],
    }


def below(metrics: dict[str, float]) -> list[str]:
    out = []
    for key, floor in FLOORS.items():
        value = metrics[key]
        bad = value > floor if key.endswith("_max") else value < floor
        if bad:
            out.append(f"{key} = {value:.4f}, floor {floor}")
    return out


def test_the_shipped_model_stays_above_the_floors() -> None:
    results = json.loads((REPO / GATE["serving_results"]).read_text(encoding="utf-8"))
    assert below(serving_metrics(results)) == []


def test_a_promoted_challenger_meets_the_floors_too() -> None:
    decision = REPO / "results" / "phase3" / "retrain" / "decision.json"
    if not decision.exists():
        return  # no retraining decision committed yet
    record = json.loads(decision.read_text(encoding="utf-8"))
    if record.get("promoted"):
        assert (
            record["challenger"]["held_out_balanced_accuracy"]
            >= FLOORS["buenos_aires_balanced_accuracy"]
        )


def test_the_gate_catches_a_regression() -> None:
    worse = {k: (v + 0.5 if k.endswith("_max") else v - 0.5) for k, v in FLOORS.items()}
    assert len(below(worse)) == len(FLOORS)
