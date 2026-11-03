"""Phase 4 eval: synthetic drift scenarios with known findings (staged, never real data).

Each scenario is a month record shaped like the monthly job's ``gold_drift_monthly`` summary
(``results/phase3/monthly/month_<n>.json``), built from **synthetic** OOD scores: a seeded
standard-normal baseline and a month drawn from a mix of an in-distribution source and made-up
new sites (``site_x``, ``site_y``, ``site_z``). The drift numbers are computed by the monitoring
code itself (:func:`cdm.monitor.drift_test`, :func:`cdm.monitor.by_skin_type`), so every
scenario obeys the pre-registered alert rule. The known findings are then derived from the
record by :func:`expected_findings`, never typed by hand.

Each family has a target outcome; a scenario's seed is the first one, counting up from a fixed
start, whose computed record has that outcome. Every scenario is labelled synthetic (staged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cdm.eval import MIN_GROUP
from cdm.monitor import by_skin_type, drift_test

STAGED = "synthetic (staged) scenario: made-up scores, not real data"
BASELINE_N = 3210  # the size of the real validation baseline; the values here are synthetic
MONTH_N = 300
ABSTAIN_QUANTILE = 0.95  # the abstain threshold is the baseline's 95th percentile (Phase 3)
NEW_SITES = ("site_x", "site_y", "site_z")
SKIN_TYPES = ("I", "II", "III", "IV")


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


def expected_findings(record: dict[str, Any]) -> dict[str, Any]:
    """The findings a correct report states, derived from the record by the job's own rules."""
    typed = sorted(t for t, v in record["by_skin_type"].items() if v["alert"])
    overall = bool(record["drift"]["alert"])
    severity = (
        "alert" if overall or typed else ("warning" if record["drift"]["warning"] else "none")
    )
    caveats = []
    if typed:
        caveats.append("skin_type_recorded_only_at_new_sites")
    if record["live"]["new_since_last_run"] > 0:
        caveats.append("new_images_counted_not_scored")
    if record["delayed_accuracy_of_month"] is None:
        caveats.append("labels_not_yet_arrived")
    new = sorted(s for s, n in record["by_source"].items() if s != "client_test" and n > 0)
    containment = {"alert": "abstain_and_hold", "warning": "monitor", "none": "none"}
    return {
        "severity": severity,
        "overall_alert": overall,
        "alerting_skin_types": typed,
        "new_sources": new,
        "containment": containment[severity],
        "delayed_accuracy_available": record["delayed_accuracy_of_month"] is not None,
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
    # Skin type is recorded only for the new site's images, as at the real new hospitals.
    other = [t for t in SKIN_TYPES if t != "II"]
    types: list[object] = [None] * (MONTH_N - n_new) + [
        "II" if rng.random() < family.type_ii_share else str(rng.choice(other))
        for _ in range(n_new)
    ]
    prev_new = bool(rng.random() < 0.5)
    has_labels = bool(rng.random() < 0.7)
    delayed = None
    if has_labels:
        labelled, melanoma = int(rng.integers(260, 300)), int(rng.integers(30, 70))
        low, high = (0.35, 0.5) if prev_new else (0.58, 0.71)
        delayed = {
            "month": 7,
            "labelled": labelled,
            "melanoma_images": melanoma,
            "balanced_accuracy": round(float(rng.uniform(low, high)), 4),
        }
    new_images = int(rng.integers(1, 400)) if rng.random() < 0.25 else 0
    threshold = float(np.quantile(baseline, ABSTAIN_QUANTILE))
    return {
        "scenario": f"{family.name}_{index:02d}",
        "staged": STAGED,
        "month": 8,
        "seed": seed,
        "live": {"count": 553019 + new_images, "new_since_last_run": new_images},
        "delayed_accuracy_of_month": delayed,
        "by_source": {"client_test": MONTH_N - n_new}
        | {s: (n_new if s == site else 0) for s in NEW_SITES},
        "drift": drift_test(baseline, scores),
        "by_skin_type": by_skin_type(baseline, scores, types, MIN_GROUP),
        "abstain_share": float(np.mean(scores > threshold)),
    }


def build(start_seed: int = 20261103, max_tries: int = 500) -> list[dict[str, Any]]:
    """Every scenario, in family order, each from the first seed that gives its target."""
    out: list[dict[str, Any]] = []
    seed = start_seed
    for family in FAMILIES:
        for i in range(family.count):
            for _ in range(max_tries):
                record = _record(family, i, seed)
                seed += 1
                if _outcome(record) == family.target:
                    break
            else:
                raise RuntimeError(f"{family.name}: no seed gave {family.target}")
            out.append({"id": record["scenario"], "family": family.name, "record": record,
                        "expected": expected_findings(record)})  # fmt: skip
    return out
