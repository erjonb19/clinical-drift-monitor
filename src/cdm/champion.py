"""Checks that a retraining run scored the champion it was meant to (silent-failures #12).

A retraining run (job run 624636213399909) scored the champion ensemble at 68.45% balanced
accuracy on the clients' validation split, against 68.26% committed for the same checkpoints in
`results/phase3/serving/results.json`. It recorded neither the weight hashes it checked nor the
models file's SHA-256, so the decision alone could not show which weights it scored.

:func:`champion_record` builds the record a decision must carry, and :func:`check_champion_val`
refuses a run whose champion validation score is off the committed one by more than a tolerance,
before any challenger is trained.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Balanced accuracy points (0.25 = a quarter of a percentage point). Set 2026-11-03, before any
# run used it: the unexplained gap in silent-failures #12 was 0.19 points, so this tolerates a
# gap of that size (GPU arithmetic on near-tied predictions is a candidate cause) and refuses a
# larger one, which wrong or changed weights would be expected to produce.
TOLERANCE_POINTS = 0.25
COMMITTED = Path("results/phase3/serving/results.json")


class ChampionMismatch(RuntimeError):
    """The champion scored differently from its committed validation result."""


def committed_val_balanced_accuracy(path: Path = COMMITTED) -> float:
    """The shipped ensemble's balanced accuracy on the clients' validation split."""
    value = json.loads(path.read_text())["validation"]["balanced_accuracy_all"]
    return float(value)


def check_champion_val(
    observed: float, committed: float, tolerance_points: float = TOLERANCE_POINTS
) -> float:
    """The gap in points (observed minus committed); raises when it is past the tolerance."""
    if not (0.0 <= observed <= 1.0 and 0.0 <= committed <= 1.0):
        raise ValueError(f"balanced accuracy outside [0, 1]: {observed}, {committed}")
    gap = 100 * (observed - committed)
    if abs(gap) > tolerance_points:
        raise ChampionMismatch(
            f"champion validation balanced accuracy {100 * observed:.2f}% is {gap:+.2f} points "
            f"from the committed {100 * committed:.2f}% (tolerance {tolerance_points} points): "
            "not the champion that was evaluated, or not scored the same way; refusing to retrain"
        )
    return gap


def champion_record(
    models_file: Path, weights_sha256: dict[int, str], committed: float, observed: float
) -> dict[str, Any]:
    """What a retraining decision records about the champion it scored."""
    if not weights_sha256:
        raise ValueError("no champion weights were checked")
    return {
        "models_file": str(models_file),
        "models_file_sha256": hashlib.sha256(models_file.read_bytes()).hexdigest(),
        "weights_sha256": {f"seed{s}": h for s, h in sorted(weights_sha256.items())},
        "committed_val_balanced_accuracy": committed,
        "observed_val_balanced_accuracy": observed,
        "gap_points": 100 * (observed - committed),
        "tolerance_points": TOLERANCE_POINTS,
    }
