"""Phase 4 (item 8): the groundedness check reads numbers as the spec's rules say and fails
closed on anything it cannot match."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cdm.groundedness import check, numbers_in

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = json.loads((REPO / "results/phase4/scenarios.json").read_text(encoding="utf-8"))
RULES: dict[str, Any] = SCENARIOS["rules"]
FLOOR: float = SCENARIOS["accuracy_floor"]
RECORD = {
    "month": 8,
    "abstain_share": 0.15666666666666668,
    "drift": {"ks_p": 3.757624025488086e-07, "psi": 0.18403631233138318, "images": 300},
    "by_source": {"site_y": 148, "client_test": 152},
    "live": {"count": 553019, "new_since_last_run": 0},
    "by_skin_type": {
        "II": {"images": 37, "alert": False, "note": "counts only (under 100 images)"}
    },
}


def grounded(text: str) -> bool:
    return bool(check(text, RECORD, RULES, FLOOR)["grounded"])


def test_rounded_decimals_percents_and_scientific_forms_match() -> None:
    assert grounded("PSI 0.184 is under the 0.2 alert line; KS p = 3.76e-07 (below 0.01).")
    assert grounded("PSI was 0.18, at least 0.1, so a warning.")
    assert grounded("16% of 300 images were above the 95th percentile; 15.7% exactly.")
    assert grounded("Month 8 had 148 site_y images and 553,019 public images in total.")
    assert grounded(f"Accuracy floor {FLOOR:.3f}; 37 type II images, under 100: counts only.")


def test_a_wrong_or_derived_number_is_ungrounded() -> None:
    assert not grounded("PSI 0.19 crossed nothing.")  # 0.184 rounds to 0.18, not 0.19
    assert not grounded("PSI was 0.02 below the 0.2 line.")  # a derived difference
    assert not grounded("301 images were scored.")
    assert not grounded("Report written 2026-11-24.")
    assert not grounded("KS p = 3.76 x 10^-7")  # a form the check does not read
    result = check("PSI 0.19 on 301 images", RECORD, RULES, FLOOR)
    assert result["unmatched"] == ["0.19", "301"] and result["numbers"] == 2


def test_a_whole_number_is_never_a_rounding_of_a_fraction() -> None:
    assert not grounded("About 1 in 6 images abstained.")  # 1 and 6 are not in the record
    assert grounded("0.2")  # 0.2 written with a decimal matches the rule's 0.2


def test_tokens_joined_to_letters_are_not_numbers() -> None:
    assert [n.text for n in numbers_in("8D report for ISIC_0024306 at site_1, v2")] == []
    assert [n.value for n in numbers_in("the 95th percentile")] == [95.0]


def test_the_report_json_is_read_whole_and_numeric_leaves_count() -> None:
    report = {
        "severity": "warning",
        "claims": [{"value": 0.18403631233138318, "source": "drift.psi"}, {"value": 0.5}],
        "eight_d": None,
    }
    result = check(report, RECORD, RULES, FLOOR)
    assert result["unmatched"] == ["0.5"]


def test_every_committed_scenario_record_grounds_its_own_numbers() -> None:
    """A report that repeats each record value as stored is grounded against that record."""
    for s in SCENARIOS["scenarios"]:
        assert check(s["record"], s["record"], RULES, FLOOR)["grounded"], s["id"]
