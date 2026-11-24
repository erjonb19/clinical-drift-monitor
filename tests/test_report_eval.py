"""Phase 4 (item 8): scoring recorded report runs under the approved eval spec."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cdm.report_eval import (
    EIGHT_D,
    FINDINGS,
    MAX_SHORT,
    MIN_CORRECT,
    PASS_AT_LEAST,
    RUNS,
    correct,
    score,
    shape_errors,
)

REPO = Path(__file__).resolve().parents[1]
DATA = json.loads((REPO / "results/phase4/scenarios.json").read_text(encoding="utf-8"))
SPEC = json.loads((REPO / "results/phase4/eval_spec.json").read_text(encoding="utf-8"))
SCENARIOS: list[dict[str, Any]] = DATA["scenarios"]


def perfect(s: dict[str, Any]) -> dict[str, Any]:
    """A correct, grounded output: the known findings, and an 8D report only when due."""
    out: dict[str, Any] = copy.deepcopy(s["expected"])
    out["claims"] = [{"value": s["record"]["month"], "source": "month"}]
    out["eight_d"] = (
        {k: f"Month {s['record']['month']}: see the record." for k in EIGHT_D}
        if s["expected"]["eight_d_report"]
        else None
    )
    return out


def runs(outputs: dict[str, list[tuple[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Three ok, perfect runs per scenario unless ``outputs`` says otherwise."""
    rows = []
    for s in SCENARIOS:
        for n, (outcome, out) in enumerate(
            (outputs or {}).get(s["id"], [("ok", perfect(s))] * 3), 1
        ):
            row = {"scenario": s["id"], "run": n, "outcome": outcome, "output": out}
            if outcome == "outage":
                row["error"] = "503 after 3 retries"
            rows.append(row)
    return rows


def result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return score(rows, SCENARIOS, DATA["rules"], DATA["accuracy_floor"])


def test_the_constants_are_the_approved_thresholds() -> None:
    t = SPEC["thresholds"]
    assert f"at least {PASS_AT_LEAST} of 30" in t["pass"]
    assert f"more than {MAX_SHORT} scenarios" in t["inconclusive"]
    assert SPEC["scoring"]["runs_per_scenario"] == RUNS == 3 and len(SCENARIOS) == 30
    assert f"at least {MIN_CORRECT} of its {RUNS}" in SPEC["scoring"]["scenario_result"]
    assert all(set(FINDINGS) | {"caveats"} == set(s["expected"]) for s in SCENARIOS)


def test_perfect_runs_pass() -> None:
    r = result(runs())
    assert (r["verdict"], r["exit_code"], r["scenarios_passed"]) == ("pass", 0, 30)


def test_each_wrong_finding_or_report_shape_is_incorrect() -> None:
    alert = next(s for s in SCENARIOS if s["expected"]["eight_d_report"])
    calm = next(s for s in SCENARIOS if not s["expected"]["eight_d_report"])
    assert correct(perfect(alert), alert["expected"]) and correct(perfect(calm), calm["expected"])
    for field, wrong in (("severity", "warning"), ("overall_alert", 1), ("containment", "none")):
        out = perfect(alert) | {field: wrong}
        assert not correct(out, alert["expected"]), field
    assert not correct(perfect(alert) | {"eight_d": None}, alert["expected"])
    half = perfect(alert)
    half["eight_d"]["prevention"] = " "
    assert not correct(half, alert["expected"])
    assert not correct(perfect(calm) | {"eight_d": perfect(alert)["eight_d"]}, calm["expected"])
    with_caveat = next(s for s in SCENARIOS if s["expected"]["caveats"])
    assert not correct(perfect(with_caveat) | {"caveats": []}, with_caveat["expected"])
    assert shape_errors("not json") and shape_errors({})


def test_two_of_three_correct_runs_pass_a_scenario_and_one_does_not() -> None:
    s = SCENARIOS[0]
    wrong = perfect(s) | {"severity": "alert" if s["expected"]["severity"] != "alert" else "none"}
    two = result(runs({s["id"]: [("ok", perfect(s)), ("ok", perfect(s)), ("ok", wrong)]}))
    one = result(runs({s["id"]: [("ok", perfect(s)), ("ok", wrong), ("invalid", "garbled")]}))
    assert two["scenarios_passed"] == 30 and one["scenarios_passed"] == 29


def test_one_ungrounded_run_fails_the_whole_eval() -> None:
    s = SCENARIOS[0]
    bad = perfect(s) | {"claims": [{"value": 0.987654, "source": "drift.psi"}]}
    r = result(runs({s["id"]: [("ok", perfect(s)), ("ok", perfect(s)), ("ok", bad)]}))
    assert r["verdict"] == "fail" and r["ungrounded_runs"][0]["unmatched"] == ["0.987654"]


def test_outages_are_not_scored_and_too_many_make_it_inconclusive() -> None:
    short = {s["id"]: [("outage", None)] * 2 + [("ok", perfect(s))] for s in SCENARIOS[:4]}
    r = result(runs(short))
    assert (r["verdict"], r["exit_code"], r["scenarios_short_of_runs"]) == ("inconclusive", 2, 4)
    three = result(runs({k: short[k] for k in list(short)[:3]}))
    assert (
        three["verdict"] == "pass" and three["scenarios_passed"] == 27
    )  # short ones count as not passed


def test_four_failed_scenarios_fail() -> None:
    bad: dict[str, list[tuple[str, Any]]] = {s["id"]: [("invalid", {})] * 3 for s in SCENARIOS[:4]}
    r = result(runs(bad))
    assert (r["verdict"], r["scenarios_passed"]) == ("fail", 26)


def test_malformed_run_lists_are_refused() -> None:
    rows = runs()
    with pytest.raises(ValueError, match="repeated"):
        result([*rows, rows[0]])
    with pytest.raises(ValueError, match="unknown"):
        result([*rows[:-1], rows[-1] | {"outcome": "timeout"}])
    with pytest.raises(ValueError, match="out of range"):
        result([*rows[:-1], rows[-1] | {"run": 4}])


def test_a_missing_run_is_refused_not_treated_as_an_outage() -> None:
    """Review of #158: dropping a run (an ungrounded one, or whole scenarios) once turned a fail
    into a pass or an inconclusive."""
    rows = runs()
    for dropped in (rows[:-1], [r for r in rows if r["run"] < 3], rows[15:], []):
        with pytest.raises(ValueError, match="not recorded"):
            result(dropped)


def test_an_outage_must_carry_its_error_and_no_output() -> None:
    s = SCENARIOS[0]
    bad = perfect(s) | {"claims": [{"value": 0.987654, "source": "drift.psi"}]}
    rows = runs({s["id"]: [("ok", perfect(s)), ("ok", perfect(s)), ("outage", None)]})
    assert result(rows)["outages"] == 1
    hidden = [r | {"output": bad} if r["outcome"] == "outage" else r for r in rows]
    with pytest.raises(ValueError, match="outage"):
        result(hidden)
    no_error = [{k: v for k, v in r.items() if k != "error"} for r in rows]
    with pytest.raises(ValueError, match="outage"):
        result(no_error)
