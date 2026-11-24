"""Backlog item 4: backfill the melanoma referral count of replay months 1 to 12.

The melanoma-referral target (cdm.slo) pools labelled months until they hold 100 melanomas, and
needs each month's count of melanomas the service flagged for referral. The committed replay
records (results/phase3/monthly/month_*.json) predate that count, so this module rebuilds each
month's delayed-accuracy record from the rows the replay itself wrote to
``gold_prediction_audit`` (probabilities and the ``refer_melanoma`` flag) and the labels in
silver. Nothing is re-scored.

Two checks make a wrong backfill fail instead of reporting success:
- the stored referral flag must equal "melanoma probability >= the referral threshold" on every
  row (so the flag and the threshold are the ones the service used);
- each rebuilt record's labelled count, melanoma count and balanced accuracy must match the
  record the replay committed for that month (month 12 has none: its labels "arrived" after the
  replay, so it is marked unchecked).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cdm.monitor import delayed_accuracy
from cdm.splits import CLASSES

TOLERANCE = 1e-9  # balanced accuracy is recomputed from the same stored probabilities


class BackfillMismatch(RuntimeError):
    """The rebuilt records disagree with the stored flags or the committed records."""


def month_record(
    month: int,
    labels: NDArray[np.int64],
    probs: NDArray[np.float64],
    referred: NDArray[np.bool_],
    refer_threshold: float,
) -> dict[str, Any]:
    """One month's delayed-accuracy record, with ``melanoma_referred``. ``labels`` holds class
    indices, -1 for an unlabelled image; ``referred`` is the audit table's stored flag."""
    expected = probs[:, CLASSES.index("mel")] >= refer_threshold
    if referred.shape != expected.shape or not np.array_equal(referred, expected):
        bad = int((referred != expected).sum()) if referred.shape == expected.shape else -1
        raise BackfillMismatch(
            f"month {month}: the stored refer_melanoma flag disagrees with the threshold "
            f"{refer_threshold} on {bad} rows"
        )
    return {"month": month, **delayed_accuracy(labels, probs, refer_threshold=refer_threshold)}


def check_committed(
    records: Sequence[Mapping[str, Any]], committed: Mapping[int, Mapping[str, Any]]
) -> dict[int, str]:
    """Compare each rebuilt record with the replay's committed one for the same month; raise on
    any difference. Returns each month's check status."""
    status: dict[int, str] = {}
    for record in records:
        month = int(record["month"])
        old = committed.get(month)
        if old is None:
            status[month] = "unchecked: no committed delayed-accuracy record for this month"
            continue
        problems = [
            key for key in ("labelled", "melanoma_images") if int(old[key]) != int(record[key])
        ]
        if abs(float(old["balanced_accuracy"]) - float(record["balanced_accuracy"])) > TOLERANCE:
            problems.append("balanced_accuracy")
        if problems:
            raise BackfillMismatch(
                f"month {month}: {', '.join(problems)} differ from the committed record"
            )
        status[month] = "matches the committed record"
    return status


def committed_records(month_files: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """The delayed-accuracy records in the committed month files, keyed by the month they
    describe (month m's file reports month m - 1)."""
    out: dict[int, dict[str, Any]] = {}
    for record in month_files:
        delayed = record.get("delayed_accuracy_of_month")
        if delayed:
            out[int(delayed["month"])] = dict(delayed)
    return out


def merge_history(
    history: Sequence[Mapping[str, Any]],
    backfill: Sequence[Mapping[str, Any]],
    before_month: int | None = None,
) -> list[dict[str, Any]]:
    """The labelled months for cdm.slo, oldest first: a backfilled record replaces a history
    record that lacks ``melanoma_referred`` and fills a month the history lacks. A history
    record that has the count is kept. With ``before_month``, only months whose labels had
    arrived by then (month < before_month) are returned."""
    merged = {int(r["month"]): dict(r) for r in history}
    for record in backfill:
        month = int(record["month"])
        if "melanoma_referred" not in merged.get(month, {}):
            merged[month] = dict(record)
    return [merged[m] for m in sorted(merged) if before_month is None or m < before_month]
