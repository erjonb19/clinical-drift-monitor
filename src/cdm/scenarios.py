"""Phase 4 eval: synthetic drift scenarios with known findings (staged, never real data).

Each scenario is a month record shaped like the monthly job's ``gold_drift_monthly`` summary
(``results/phase3/monthly/month_<n>.json``), built from **synthetic** OOD scores: a seeded
standard-normal baseline and a month drawn from a mix of an in-distribution source and made-up
new sites (``site_x``, ``site_y``, ``site_z``). The drift numbers are computed by the monitoring
code itself (:func:`cdm.monitor.drift_test`, :func:`cdm.monitor.by_skin_type`), so every
scenario obeys the pre-registered alert rule. The known findings are then derived from the
record by :func:`expected_findings`, never typed by hand.

Each family has a target outcome; a scenario's seed is the first one, counting up from its own
fixed start (so scenarios do not depend on each other), whose computed record has that outcome.
Every scenario is labelled synthetic (staged). The scenario ID and seed are kept outside the
record, which is the only thing the report writer sees, together with :data:`RULES`.

The committed file, not a rebuild, is the eval's input: numpy does not promise the same random
streams across versions, so ``scripts/phase4_scenarios.py --check`` is run by hand, not in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cdm.eval import MIN_GROUP
from cdm.monitor import ALERT_P, ALERT_PSI, WARN_PSI, by_skin_type, drift_test

STAGED = "synthetic (staged) scenario: made-up scores, not real data"
BASELINE_N = 3210  # the size of the real validation baseline; the values here are synthetic
MONTH_N = 300
ABSTAIN_QUANTILE = 0.95  # the abstain threshold is the baseline's 95th percentile (Phase 3)
NEW_SITES = ("site_x", "site_y", "site_z")
SKIN_TYPES = ("I", "II", "III", "IV")
UNRECORDED_AT_NEW_SITE = 0.15  # share of new-site images with no skin type (real sites leave some)
# The repository checkout (editable install); not the working directory.
MONTHLY = Path(__file__).resolve().parents[2] / "results" / "phase3" / "monthly"


def in_distribution_accuracy_floor(monthly: Path = MONTHLY) -> float:
    """The lowest delayed balanced accuracy of the real in-distribution months (1 to 4 of the
    committed replay; months 2 to 4 report months 1 to 3, month 5 reports month 4)."""
    values = []
    for n in (2, 3, 4, 5):
        record = json.loads((monthly / f"month_{n:02d}.json").read_text())
        values.append(float(record["delayed_accuracy_of_month"]["balanced_accuracy"]))
    return min(values)


# The fixed context the report writer gets with each record: the job's rules and the words the
# findings use. Everything in the known findings follows from a record plus these rules.
RULES = {
    "alert_rule": f"alert when KS p < {ALERT_P} and PSI >= {ALERT_PSI}; warning when PSI >= {WARN_PSI} without an alert",  # noqa: E501
    "skin_type_rule": f"the same test per Fitzpatrick type with at least {MIN_GROUP} images that month; smaller or unrecorded groups get counts only and never alert",  # noqa: E501
    "skin_type_caveat": "skin type is recorded only at the new sites, so a skin-type alert detects the site, not anything about skin type",  # noqa: E501
    "month_alert": "the month is in alert when the overall test or any skin-type test alerts",
    "live_check": "new public images are counted each month, not downloaded or scored",
    "labels": "a month's accuracy is reported by the next month's run; when the key is missing, labels have not arrived",  # noqa: E501
    "abstain_share": "the share of the month's images whose OOD score is above the abstain threshold (the baseline's 95th percentile)",  # noqa: E501
    "containment_words": {"alert": "abstain_and_hold", "warning": "monitor", "none": "none"},
    "report_when": "an 8D report only when the month is in alert; for a warning or no drift, findings only and no 8D report",  # noqa: E501
    "accuracy_floor": "delayed balanced accuracy below the lowest in-distribution month is a drop",
}  # fmt: skip


@dataclass(frozen=True)
class Family:
    name: str
    count: int
    target: str  # the outcome a scenario of this family must have
    new_share: tuple[float, float]  # share of the month from the new site
    shift: tuple[float, float]  # mean shift of the new site's scores
    type_ii_share: float  # share of new-site images recorded as skin type II


FAMILIES = (
    Family("no_drift", 4, "none", (0.0, 0.0), (0.0, 0.0), 0.0),
    Family("warning", 5, "warning", (0.2, 0.5), (0.4, 1.0), 0.3),
    Family("alert_moderate", 6, "alert_overall", (0.5, 1.0), (0.6, 1.2), 0.3),
    Family("alert_strong", 6, "alert_overall", (0.8, 1.0), (1.5, 3.0), 0.3),
    Family("alert_skin_type_only", 4, "alert_skin_type_only", (0.34, 0.45), (0.8, 1.6), 0.95),
    Family("alert_overall_and_skin_type", 5, "alert_overall_and_skin_type", (0.6, 1.0), (0.8, 2.0), 0.7),  # noqa: E501
)  # fmt: skip


def _outcome(record: dict[str, Any]) -> str:
    overall = bool(record["drift"]["alert"])
    typed = any(v["alert"] for v in record["by_skin_type"].values())
    if overall and typed:
        return "alert_overall_and_skin_type"
    if overall:
        return "alert_overall"
    if typed:
        return "alert_skin_type_only"
    return "warning" if record["drift"]["warning"] else "none"


def expected_findings(record: dict[str, Any], floor: float | None = None) -> dict[str, Any]:
    """The findings a correct report states, derived from a record and :data:`RULES` only."""
    floor = in_distribution_accuracy_floor() if floor is None else floor
    typed = sorted(t for t, v in record["by_skin_type"].items() if v["alert"])
    overall = bool(record["drift"]["alert"])
    severity = (
        "alert" if overall or typed else ("warning" if record["drift"]["warning"] else "none")
    )
    delayed = record.get("delayed_accuracy_of_month")
    caveats = []
    if typed:
        caveats.append("skin_type_recorded_only_at_new_sites")
    if record.get("live", {}).get("new_since_last_run", 0) > 0:
        caveats.append("new_images_counted_not_scored")
    if delayed is None:
        caveats.append("labels_not_yet_arrived")
    new = sorted(s for s, n in record["by_source"].items() if s != "client_test" and n > 0)
    return {
        "severity": severity,
        "overall_alert": overall,
        "alerting_skin_types": typed,
        "new_sources": new,
        "containment": RULES["containment_words"][severity],  # type: ignore[index]
        "eight_d_report": severity == "alert",
        "delayed_accuracy_available": delayed is not None,
        "accuracy_drop": None if delayed is None else bool(delayed["balanced_accuracy"] < floor),
        "caveats": caveats,
    }


def _record(family: Family, index: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    baseline = rng.standard_normal(BASELINE_N)
    site = NEW_SITES[index % len(NEW_SITES)]
    share = float(rng.uniform(*family.new_share))
    n_new = int(round(share * MONTH_N))
    shift = float(rng.uniform(*family.shift))
    scores = np.concatenate([rng.standard_normal(MONTH_N - n_new), rng.normal(shift, 1.2, n_new)])
    # Skin type is recorded only at the new site, and not for every one of its images.
    other = [t for t in SKIN_TYPES if t != "II"]
    new_types: list[object] = []
    for _ in range(n_new):
        if rng.random() < UNRECORDED_AT_NEW_SITE:
            new_types.append(None)
        else:
            new_types.append(
                "II" if rng.random() < family.type_ii_share else str(rng.choice(other))
            )
    types: list[object] = [None] * (MONTH_N - n_new) + new_types
    # Last month's accuracy: in range unless last month had new-site images. Kept clear of the
    # floor (0.591) on both sides so the answer is never a coin toss.
    prev_new = family.name != "no_drift" and bool(rng.random() < 0.5)
    has_labels = bool(rng.random() < 0.7)
    delayed = None
    if has_labels:
        labelled, melanoma = int(rng.integers(260, 300)), int(rng.integers(30, 70))
        low, high = (0.35, 0.5) if prev_new else (0.62, 0.71)
        delayed = {
            "month": 7,
            "labelled": labelled,
            "melanoma_images": melanoma,
            "balanced_accuracy": round(float(rng.uniform(low, high)), 4),
        }
    new_images = int(rng.integers(1, 400)) if rng.random() < 0.25 else 0
    threshold = float(np.quantile(baseline, ABSTAIN_QUANTILE))
    record: dict[str, Any] = {
        "month": 8,
        "staged": STAGED,
        "run_utc": "synthetic",
        "job_run": "synthetic",
        "code_version": "synthetic",
        "bundle_version": "synthetic",
        "live": {"count": 553019 + new_images, "new_since_last_run": new_images},
    }
    if delayed is not None:  # real records leave the key out until labels arrive
        record["delayed_accuracy_of_month"] = delayed
    record |= {
        "by_source": {"client_test": MONTH_N - n_new}
        | {s: (n_new if s == site else 0) for s in NEW_SITES},
        "drift": drift_test(baseline, scores),
        "by_skin_type": by_skin_type(baseline, scores, types, MIN_GROUP),
        "abstain_share": float(np.mean(scores > threshold)),
    }
    return record


def build(start_seed: int = 20261103, max_tries: int = 1000) -> list[dict[str, Any]]:
    """Every scenario; each searches its own seed range for its family's target outcome."""
    floor = in_distribution_accuracy_floor()
    out: list[dict[str, Any]] = []
    for f_index, family in enumerate(FAMILIES):
        for i in range(family.count):
            first = start_seed + 100_000 * f_index + max_tries * i
            for seed in range(first, first + max_tries):
                record = _record(family, i, seed)
                if _outcome(record) == family.target:
                    break
            else:
                raise RuntimeError(f"{family.name}: no seed gave {family.target}")
            out.append({"id": f"{family.name}_{i:02d}", "family": family.name, "seed": seed,
                        "record": record, "expected": expected_findings(record, floor)})  # fmt: skip  # noqa: E501
    return out
