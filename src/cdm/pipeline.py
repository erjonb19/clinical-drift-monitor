"""Backlog item 17: the retraining pipeline's two pure decisions (docs/specs/
item17-retraining-pipeline.md, approved by the owner on 2026-11-09).

- ``trigger_check``: the pipeline trains only when a month is in alert **and** that month's
  labels have arrived, so the challenger can learn from labelled images of the shifted month.
- ``verify_promotion``: the hand-run ``promote_challenger`` job is the human approval. It moves
  the ``champion`` alias only to a registered version whose decision file the owner names by
  SHA-256, whose rule said promote, and whose serving bundle was built and checked.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from cdm.slo import MIN_LABELLED


class PromotionRefused(ValueError):
    """The promotion is not allowed; the message says why."""


def month_in_alert(record: Mapping[str, Any]) -> bool:
    """The monthly table's alert: the overall drift alert or any skin-type alert."""
    drift = record.get("drift") or {}
    by_type = record.get("by_skin_type") or {}
    return bool(drift.get("alert")) or any(v.get("alert") for v in by_type.values())


def trigger_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``records`` are the monthly job's month records (the summary JSON), any order. A month
    triggers when it alerted and a later record reports its delayed accuracy from at least
    ``MIN_LABELLED`` labelled images."""
    labelled = {
        int(d["month"])
        for r in records
        if (d := r.get("delayed_accuracy_of_month")) and int(d.get("labelled", 0)) >= MIN_LABELLED
    }
    alerted = sorted(int(r["month"]) for r in records if month_in_alert(r))
    ready = [m for m in alerted if m in labelled]
    if ready:
        return {"status": "trigger", "months": ready}
    waiting = [m for m in alerted if m not in labelled]
    reason = (
        f"months {waiting} alerted but their labels have not arrived"
        if waiting
        else "no month in alert"
    )
    return {"status": "no_trigger", "reason": reason}


def verify_promotion(
    decision_bytes: bytes, expected_sha256: str, version_tags: Mapping[str, str]
) -> dict[str, Any]:
    """Refuse unless the decision file is the one the owner named, its rule promoted, and the
    registry version was registered from that same decision with a checked serving bundle."""
    got = hashlib.sha256(decision_bytes).hexdigest()
    if got != expected_sha256.lower():
        raise PromotionRefused(f"the decision file's SHA-256 is {got}, not the one given")
    decision: dict[str, Any] = json.loads(decision_bytes)
    if decision.get("promoted") is not True:
        raise PromotionRefused("the decision file says the rule did not promote")
    if version_tags.get("decision_sha256") != got:
        raise PromotionRefused("this registry version was not registered from that decision")
    if not version_tags.get("serving_bundle_sha256"):
        raise PromotionRefused("this registry version has no checked serving bundle")
    return decision
