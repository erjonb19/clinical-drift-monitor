"""Backlog items 4 and 5: model targets (SLOs) checked each month, and the alert path.

The targets and thresholds are in ``config/slo.json``. The melanoma sensitivity rule is the
owner's (2026-11-06, docs/BACKLOG.md item 4): a trailing window of labelled months grows back
from the latest month whose labels have arrived until it holds at least 100 melanoma images;
sensitivity is reported with a 95% Wilson score interval; if all months together hold fewer than
100 melanomas the target is neither met nor broken ("insufficient data").

The alert path: the monthly job checks every target after writing its results, then fails its
run with a message starting "ALERT" when any is broken or any drift alert fired. Databricks
emails the owner on a failed run (databricks/monthly_drift_job.json), so an alert reaches them
without a paid service. A failed run caused by an alert says so in its message.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CONFIG = Path(__file__).resolve().parents[2] / "config" / "slo.json"
MET, BROKEN, INSUFFICIENT = "met", "broken", "insufficient_data"
MIN_LABELLED = 100  # the repo's minimum for a reported metric (cdm.eval.MIN_GROUP)


class AlertRaised(RuntimeError):
    """Raised at the end of a monthly run when a target is broken or a drift alert fired."""


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return config


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson score interval for a proportion (95% by default)."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0: {successes}, {n}")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def melanoma_window(
    months: Sequence[dict[str, Any]], minimum: int, hits_key: str = "melanoma_referred"
) -> dict[str, Any]:
    """The trailing window: ``months`` are the labelled months' delayed-accuracy records
    (``month``, ``melanoma_images`` and the hits count), oldest first. Months without the hits
    count cannot join the window; if the latest one lacks it, the result says so."""
    window: list[int] = []
    melanomas = hits = 0
    for record in reversed(months):
        if hits_key not in record:
            if not window:
                return {
                    "status": INSUFFICIENT,
                    "reason": f"the latest labelled month has no {hits_key} count",
                }
            break
        window.append(int(record["month"]))
        melanomas += int(record["melanoma_images"])
        hits += int(record[hits_key])
        if melanomas >= minimum:
            break
    if melanomas < minimum:
        return {
            "status": INSUFFICIENT,
            "months": sorted(window),
            "melanoma_images": melanomas,
            "reason": f"fewer than {minimum} melanoma images in the labelled months",
        }
    low, high = wilson(hits, melanomas)
    return {
        "months": sorted(window),
        "melanoma_images": melanomas,
        hits_key: hits,
        "sensitivity": hits / melanomas,
        "wilson_95": [low, high],
    }


def judge_sensitivity(window: dict[str, Any], target: float) -> str:
    """Judged on the point estimate (one of the owner's two options, 2026-11-06): met when it
    reaches the target, broken when below; the Wilson interval is reported beside it.
    Insufficient data stays so."""
    if window.get("status") == INSUFFICIENT:
        return INSUFFICIENT
    return MET if window["sensitivity"] >= target else BROKEN


def evaluate(
    record: dict[str, Any], labelled: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Every target for one month. ``record`` is the month's results record; ``labelled`` the
    delayed-accuracy records of every labelled month so far, oldest first."""
    targets = config["targets"]
    out: dict[str, Any] = {}
    mel = targets["melanoma_referral_sensitivity"]
    window = melanoma_window(labelled, int(mel["min_melanoma_images"]))
    out["melanoma_referral_sensitivity"] = {
        **window,
        "target": mel["value"],
        "status": judge_sensitivity(window, float(mel["value"])),
    }
    bacc = targets["delayed_balanced_accuracy"]
    delayed = record.get("delayed_accuracy_of_month")
    if delayed is None:
        out["delayed_balanced_accuracy"] = {
            "status": INSUFFICIENT,
            "reason": "labels not yet arrived",
        }
    elif int(delayed.get("labelled", 0)) < MIN_LABELLED:
        out["delayed_balanced_accuracy"] = {
            "status": INSUFFICIENT,
            "reason": f"fewer than {MIN_LABELLED} labelled images",
        }
    else:
        value = float(delayed["balanced_accuracy"])
        out["delayed_balanced_accuracy"] = {
            "month": delayed["month"],
            "value": value,
            "target": bacc["value"],
            "status": MET if value >= float(bacc["value"]) else BROKEN,
        }
    abstain = targets["abstain_share"]
    if "abstain_share" not in record:
        out["abstain_share"] = {"status": INSUFFICIENT, "reason": "no images scored this month"}
    else:
        share = float(record["abstain_share"])
        out["abstain_share"] = {
            "value": share,
            "target": abstain["value"],
            "status": MET if share <= float(abstain["value"]) else BROKEN,
        }
    return out


def alerts(record: dict[str, Any], slo: dict[str, Any]) -> list[str]:
    """Why the month should alert: each broken target, and each drift alert in the record."""
    reasons = [f"target broken: {name}" for name, r in slo.items() if r["status"] == BROKEN]
    if record.get("drift", {}).get("alert"):
        reasons.append("drift alert (overall)")
    reasons += [
        f"drift alert (skin type {t})"
        for t, r in record.get("by_skin_type", {}).items()
        if r.get("alert")
    ]
    return reasons


def raise_if_alerts(by_month: dict[int, list[str]]) -> None:
    """The alert path: fail the run with a message the owner can tell from a job error. Called
    once at the end of a run, so a backfill of several months scores every month first."""
    parts = [f"month {m}: " + "; ".join(r) for m, r in sorted(by_month.items()) if r]
    if parts:
        raise AlertRaised("ALERT (not a job error), " + " | ".join(parts))
