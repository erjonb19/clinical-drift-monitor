"""Phase 1 data quality gates as plain functions over pandas frames.

The pipeline turns each result into a row of a gate table with an expect-or-fail constraint,
so a violation stops the pipeline update. Keeping the logic here means CI tests exactly the
checks that decide whether a pipeline run fails.

Frames use the silver column names: ``isic_id``, ``source``, ``lesion_id``, ``split``,
``label``, ``isic_label``, ``license``, ``attribution``, ``sha256``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from cdm.splits import SPLITS, split_fingerprint

NEEDS_ATTRIBUTION = ("CC-BY", "CC-BY-NC")


@dataclass(frozen=True)
class Gate:
    name: str
    violations: int
    detail: str


def _examples(values: Iterable[object], limit: int = 5) -> str:
    items = [str(v) for v in values]
    return ", ".join(items[:limit]) + (
        f" (+{len(items) - limit} more)" if len(items) > limit else ""
    )


def lesions_in_one_split(frame: pd.DataFrame) -> Gate:
    """No lesion may have images in two of train, val and test."""
    trained = frame[frame["split"].isin(SPLITS)]
    per_lesion = trained.groupby("lesion_id")["split"].nunique()
    leaked = per_lesion[per_lesion > 1].index
    return Gate("no_lesion_in_two_splits", len(leaked), _examples(leaked))


def split_matches_phase0(ham: pd.DataFrame, expected: str) -> Gate:
    """HAM10000's lesion-to-split assignment must be exactly Phase 0's."""
    got = split_fingerprint(ham.rename(columns={"label": "dx"}), ham["split"])
    return Gate(
        "split_fingerprint_matches_phase0", int(got != expected), f"got {got}, expected {expected}"
    )


def class_counts(ham: pd.DataFrame, expected: Mapping[str, int]) -> Gate:
    """HAM10000's image count per class must equal the published counts."""
    counts = ham["label"].value_counts().to_dict()
    wrong = {c: (counts.get(c, 0), n) for c, n in expected.items() if counts.get(c, 0) != n}
    extra = set(counts) - set(expected)
    detail = _examples(f"{c}: got {g}, expected {e}" for c, (g, e) in wrong.items())
    return Gate("class_counts_as_expected", len(wrong) + len(extra), detail or "")


def no_missing_labels(frame: pd.DataFrame) -> Gate:
    """Every image used for training, validation or testing has a label."""
    missing = frame[frame["split"].isin(SPLITS) & frame["label"].isna()]["isic_id"]
    return Gate("no_missing_labels", len(missing), _examples(missing))


def no_ham_in_sites(frame: pd.DataFrame) -> Gate:
    """No HAM10000 image may appear as a new-site image: that would be training data
    scored as new data."""
    ham_ids = set(frame.loc[frame["source"] == "ham10000", "isic_id"])
    leaked = frame[(frame["source"] != "ham10000") & frame["isic_id"].isin(ham_ids)]["isic_id"]
    return Gate("no_ham10000_image_in_a_site", len(leaked), _examples(leaked))


def license_policy(frame: pd.DataFrame, allowed: Iterable[str]) -> Gate:
    """Every license is allowed, and every CC-BY or CC-BY-NC image carries an attribution."""
    bad_license = ~frame["license"].isin(list(allowed))
    no_attribution = frame["license"].isin(NEEDS_ATTRIBUTION) & (
        frame["attribution"].isna() | (frame["attribution"].astype(str).str.strip() == "")
    )
    bad = frame[bad_license | no_attribution]["isic_id"]
    return Gate("license_and_attribution", len(bad), _examples(bad))


def mapping_reproduces_ham_labels(ham: pd.DataFrame) -> Gate:
    """The ISIC-to-HAM diagnosis mapping must give each HAM10000 image its HAM10000 label;
    otherwise site labels cannot be trusted either."""
    wrong = ham[ham["isic_label"] != ham["label"]]["isic_id"]
    return Gate("diagnosis_mapping_matches_ham10000", len(wrong), _examples(wrong))


def no_duplicate_files(frame: pd.DataFrame) -> Gate:
    """No two images may be the same file (same SHA-256) under different IDs."""
    dup = frame[frame.duplicated("sha256", keep=False)].sort_values("sha256")["isic_id"]
    return Gate("no_duplicate_files", len(dup), _examples(dup))


def run_all(frame: pd.DataFrame, config: Mapping[str, object]) -> list[Gate]:
    ham_cfg = config["ham10000"]
    assert isinstance(ham_cfg, Mapping)
    allowed = config["license_allowed"]
    assert isinstance(allowed, list)
    ham = frame[frame["source"] == "ham10000"]
    return [
        lesions_in_one_split(frame),
        split_matches_phase0(ham, str(ham_cfg["split_fingerprint"])),
        class_counts(ham, dict(ham_cfg["class_counts"])),
        no_missing_labels(frame),
        no_ham_in_sites(frame),
        license_policy(frame, allowed),
        mapping_reproduces_ham_labels(ham),
        no_duplicate_files(frame),
    ]
