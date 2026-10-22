"""Research Phase C: performance and drift by age band, sex and body site.

Groups come from silver's ``age_approx`` (age bands) and from the OMOP tables (sex as
``gender_concept_id``, body site as ``anatomic_site_concept_id``), joined by ISIC ID. Concept
IDs are the group keys: no vocabulary names. Every rule of :func:`cdm.eval.subgroup_table`
applies (at least 100 images for a metric, and 100 melanoma images for melanoma sensitivity).
Intervals resample the group's patients, else lesions, else images, as everywhere else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cdm.eval import MIN_GROUP, detection_metrics, subgroup_table, summarize
from cdm.final import bootstrap_by_lesion

Array = NDArray[np.float64]
# Lower bound inclusive, upper bound exclusive; fixed in the spec before any result was seen.
AGE_BANDS = ((0, 30, "0-29"), (30, 45, "30-44"), (45, 60, "45-59"), (60, 75, "60-74"))
OLDEST = (75, "75+")


def age_band(age: object) -> str:
    """The band of an ``age_approx`` value; "unknown" when it is missing."""
    if age is None or (isinstance(age, float) and np.isnan(age)) or age == "":
        return "unknown"
    value = float(str(age))
    for low, high, name in AGE_BANDS:
        if low <= value < high:
            return name
    if value >= OLDEST[0]:
        return OLDEST[1]
    raise ValueError(f"age {value} is below every band")


def concept_group(concept_id: object) -> str:
    """A concept ID as a group key; 0 (unmapped) and missing both become "unknown"."""
    if concept_id is None or (isinstance(concept_id, float) and np.isnan(concept_id)):
        return "unknown"
    value = int(float(str(concept_id)))
    return "unknown" if value == 0 else f"concept:{value}"


def performance(
    labels: NDArray[np.int64],
    logits: Mapping[str, Array],
    units: Sequence[str],
    groups: Sequence[str],
    probe_seeds: Sequence[str],
    n_boot: int = 1000,
    min_n: int = MIN_GROUP,
) -> dict[str, Any]:
    """Each model's subgroup table, with bootstrap intervals for every group that gets a
    metric, and the probe's mean and range over seeds per group."""
    keys = np.asarray(groups, dtype=object)
    unit_arr = np.asarray(units, dtype=object)
    tables: dict[str, list[dict[str, object]]] = {}
    for model, z in logits.items():
        rows = subgroup_table(labels, z, list(keys), min_n)
        for row in rows:
            if "balanced_accuracy" in row:
                mask = keys == row["group"]
                row["bootstrap_95"] = bootstrap_by_lesion(
                    z[mask], labels[mask], list(unit_arr[mask]), n=n_boot
                )
        tables[model] = rows
    over_seeds: dict[str, dict[str, dict[str, float]]] = {}
    if probe_seeds:
        for i, row in enumerate(tables[probe_seeds[0]]):
            for metric in ("balanced_accuracy", "melanoma_sensitivity"):
                if metric in row:
                    values = [float(str(tables[s][i][metric])) for s in probe_seeds]
                    over_seeds.setdefault(str(row["group"]), {})[metric] = summarize(values)
    return {"models": tables, "probe_over_seeds": over_seeds}


def drift(
    reference: Array,
    scores: Array,
    groups: Sequence[str],
    min_n: int = MIN_GROUP,
) -> dict[str, Any]:
    """AUROC and FPR@95TPR of one detector, the whole reference set against each group of a
    shifted set. Groups under ``min_n`` images get their count only."""
    keys = np.asarray(groups, dtype=object)
    if len(keys) != len(scores):
        raise ValueError(f"{len(keys)} group values for {len(scores)} scores")
    out: dict[str, Any] = {}
    for group in sorted(set(keys.tolist()), key=str):
        mask = keys == group
        n = int(mask.sum())
        out[group] = (
            {"images": n, **detection_metrics(reference, scores[mask])}
            if n >= min_n
            else {"images": n, "warning": f"fewer than {min_n} images: count only"}
        )
    return out


def check_reproduced(name: str, got: float, committed: float, tol: float = 1e-9) -> None:
    """Regenerated predictions must give the committed overall number, or nothing is reported."""
    if abs(got - committed) > tol:
        raise RuntimeError(f"{name}: regenerated {got!r}, committed {committed!r}")
