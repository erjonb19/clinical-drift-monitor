"""The Phase 4 eval's synthetic drift scenarios (staged) and their known findings."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cdm.monitor import ALERT_P, ALERT_PSI, WARN_PSI
from cdm.scenarios import FAMILIES, NEW_SITES, STAGED, _outcome, build, expected_findings

REPO = Path(__file__).resolve().parents[1]
COMMITTED = json.loads((REPO / "results" / "phase4" / "scenarios.json").read_text())
REAL_SOURCES = {"buenos_aires", "pad_ufes"}


@pytest.fixture(scope="module")
def scenarios() -> list[dict[str, Any]]:
    return build()


def _close(a: Any, b: Any) -> bool:
    """Equal structure and strings; floats equal to 1e-9 (scipy and numpy may differ in the last
    bits between platforms)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-300)
    return bool(a == b)


def test_committed_file_is_a_fresh_build(scenarios: list[dict[str, Any]]) -> None:
    assert COMMITTED["count"] == len(scenarios)
    assert _close(COMMITTED["scenarios"], json.loads(json.dumps(scenarios, sort_keys=True)))


def test_twenty_to_thirty_scenarios_in_the_planned_families(
    scenarios: list[dict[str, Any]],
) -> None:
    assert 20 <= len(scenarios) <= 30
    assert Counter(s["family"] for s in scenarios) == {f.name: f.count for f in FAMILIES}
    assert len({s["id"] for s in scenarios}) == len(scenarios)


def test_every_scenario_is_labelled_staged_and_uses_no_real_source(
    scenarios: list[dict[str, Any]],
) -> None:
    assert COMMITTED["staged"] == STAGED
    for s in scenarios:
        assert s["record"]["staged"] == STAGED
        sources = set(s["record"]["by_source"])
        assert sources == {"client_test", *NEW_SITES} and not sources & REAL_SOURCES


def test_records_obey_the_pre_registered_alert_rule(scenarios: list[dict[str, Any]]) -> None:
    for s in scenarios:
        groups = [s["record"]["drift"]]
        groups += [g for g in s["record"]["by_skin_type"].values() if "psi" in g]
        for g in groups:
            assert g["alert"] == (g["ks_p"] < ALERT_P and g["psi"] >= ALERT_PSI)
            assert g["warning"] == (not g["alert"] and g["psi"] >= WARN_PSI)


def test_small_or_unrecorded_skin_type_groups_never_alert(scenarios: list[dict[str, Any]]) -> None:
    for s in scenarios:
        types = s["record"]["by_skin_type"]
        for name, g in types.items():
            if name == "not recorded" or g["images"] < 100:
                assert g["alert"] is False and "psi" not in g
        # skin type is recorded only for new-site images, as at the real new hospitals
        assert types["not recorded"]["images"] == s["record"]["by_source"]["client_test"]


def test_each_family_has_its_target_outcome(scenarios: list[dict[str, Any]]) -> None:
    target = {f.name: f.target for f in FAMILIES}
    for s in scenarios:
        assert _outcome(s["record"]) == target[s["family"]], s["id"]


def test_known_findings_are_derived_not_typed(scenarios: list[dict[str, Any]]) -> None:
    for s in scenarios:
        assert s["expected"] == expected_findings(s["record"])


def test_known_findings_cover_each_case(scenarios: list[dict[str, Any]]) -> None:
    severities = Counter(s["expected"]["severity"] for s in scenarios)
    caveats = Counter(c for s in scenarios for c in s["expected"]["caveats"])
    assert set(severities) == {"none", "warning", "alert"}
    assert set(caveats) == {
        "skin_type_recorded_only_at_new_sites",
        "new_images_counted_not_scored",
        "labels_not_yet_arrived",
    }
    for s in scenarios:
        e = s["expected"]
        assert (e["containment"] == "abstain_and_hold") == (e["severity"] == "alert")
        if s["family"] == "alert_skin_type_only":
            assert not e["overall_alert"] and e["alerting_skin_types"] == ["II"]
