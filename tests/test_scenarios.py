"""The Phase 4 eval's synthetic drift scenarios (staged) and their known findings.

These tests check the committed file, which is the eval's input. A rebuild is not compared in
CI: numpy does not promise the same random streams across versions (run
``python scripts/phase4_scenarios.py --check`` by hand).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cdm.eval import MIN_GROUP
from cdm.monitor import ALERT_P, ALERT_PSI, WARN_PSI
from cdm.scenarios import (
    FAMILIES,
    NEW_SITES,
    RULES,
    STAGED,
    _outcome,
    expected_findings,
    in_distribution_accuracy_floor,
)

REPO = Path(__file__).resolve().parents[1]
DOC = json.loads((REPO / "results" / "phase4" / "scenarios.json").read_text())
SCENARIOS: list[dict[str, Any]] = DOC["scenarios"]
REAL_SOURCES = {"buenos_aires", "pad_ufes"}


def test_twenty_to_thirty_scenarios_in_the_planned_families() -> None:
    assert 20 <= len(SCENARIOS) == DOC["count"] <= 30
    assert Counter(s["family"] for s in SCENARIOS) == {f.name: f.count for f in FAMILIES}
    assert len({s["id"] for s in SCENARIOS}) == len(SCENARIOS)


def test_the_record_does_not_give_away_the_answer() -> None:
    """The writer sees only the record and the rules: no ID, family, seed or finding in it."""
    for s in SCENARIOS:
        text = json.dumps(s["record"])
        assert s["id"] not in text  # "warning" alone is a real record field, so IDs are checked
        assert not any(f.name in text for f in FAMILIES if "_" in f.name)
        assert not {"scenario", "seed", "family", "expected"} & set(s["record"])


def test_every_scenario_is_labelled_staged_and_uses_no_real_source() -> None:
    assert DOC["staged"] == STAGED
    for s in SCENARIOS:
        assert s["record"]["staged"] == STAGED
        sources = set(s["record"]["by_source"])
        assert sources == {"client_test", *NEW_SITES} and not sources & REAL_SOURCES
        assert all(s["record"][k] == "synthetic" for k in ("run_utc", "job_run", "code_version"))


def test_records_have_the_real_month_record_shape() -> None:
    real = json.loads((REPO / "results" / "phase3" / "monthly" / "month_09.json").read_text())
    for s in SCENARIOS:
        assert set(s["record"]) - {"staged"} <= set(real)
        assert set(s["record"]["drift"]) == set(real["drift"])
        delayed = s["record"].get("delayed_accuracy_of_month")
        if delayed is not None:  # real records leave the key out until labels arrive
            assert set(delayed) == set(real["delayed_accuracy_of_month"])


def test_records_obey_the_pre_registered_alert_rule() -> None:
    for s in SCENARIOS:
        groups = [s["record"]["drift"]]
        groups += [g for g in s["record"]["by_skin_type"].values() if "psi" in g]
        for g in groups:
            assert g["alert"] == (g["ks_p"] < ALERT_P and g["psi"] >= ALERT_PSI)
            assert g["warning"] == (not g["alert"] and g["psi"] >= WARN_PSI)


def test_small_or_unrecorded_skin_type_groups_never_alert() -> None:
    for s in SCENARIOS:
        types = s["record"]["by_skin_type"]
        for name, g in types.items():
            if name == "not recorded" or g["images"] < MIN_GROUP:
                assert g["alert"] is False and "psi" not in g
        # all in-distribution images and some new-site images have no skin type
        assert types["not recorded"]["images"] >= s["record"]["by_source"]["client_test"]


def test_each_family_has_its_target_outcome() -> None:
    target = {f.name: f.target for f in FAMILIES}
    for s in SCENARIOS:
        assert _outcome(s["record"]) == target[s["family"]], s["id"]


def test_known_findings_are_derived_from_record_and_rules() -> None:
    assert DOC["accuracy_floor"] == pytest.approx(
        in_distribution_accuracy_floor(REPO / "results/phase3/monthly")
    )
    assert DOC["rules"] == json.loads(json.dumps(RULES))
    for s in SCENARIOS:
        assert s["expected"] == expected_findings(s["record"], DOC["accuracy_floor"])


def test_rules_quote_the_monitor_constants() -> None:
    assert (
        f"KS p < {ALERT_P}" in RULES["alert_rule"] and f"PSI >= {ALERT_PSI}" in RULES["alert_rule"]
    )
    assert f"PSI >= {WARN_PSI}" in RULES["alert_rule"]
    assert f"at least {MIN_GROUP} images" in RULES["skin_type_rule"]


def test_no_accuracy_answer_sits_near_the_floor() -> None:
    for s in SCENARIOS:
        delayed = s["record"].get("delayed_accuracy_of_month")
        if delayed is not None:
            assert abs(delayed["balanced_accuracy"] - DOC["accuracy_floor"]) > 0.02, s["id"]


def test_known_findings_cover_each_case() -> None:
    severities = Counter(s["expected"]["severity"] for s in SCENARIOS)
    caveats = Counter(c for s in SCENARIOS for c in s["expected"]["caveats"])
    drops = Counter(s["expected"]["accuracy_drop"] for s in SCENARIOS)
    assert set(severities) == {"none", "warning", "alert"}
    assert set(drops) == {True, False, None}
    assert set(caveats) == {
        "skin_type_recorded_only_at_new_sites",
        "new_images_counted_not_scored",
        "labels_not_yet_arrived",
    }
    for s in SCENARIOS:
        e = s["expected"]
        assert e["eight_d_report"] == (e["severity"] == "alert")
        assert (e["containment"] == "abstain_and_hold") == (e["severity"] == "alert")
        if s["family"] == "alert_skin_type_only":
            assert not e["overall_alert"] and e["alerting_skin_types"] == ["II"]
        if s["family"] == "no_drift":
            assert e["accuracy_drop"] is not True
