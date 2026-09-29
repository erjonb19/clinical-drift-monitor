"""Phase 1 data quality gates as plain functions over pandas frames.

The pipeline turns each result into a row of a gate table with an expect-or-fail constraint,
so a violation stops the pipeline update. Keeping the logic here means CI tests exactly the
checks that decide whether a pipeline run fails.

Frames use the silver column names: ``isic_id``, ``source``, ``lesion_id``, ``split``,
``label``, ``isic_label``, ``license``, ``attribution``, ``sha256`` (computed in the
pipeline; null if no file landed) and ``manifest_sha256`` (recorded by the ingest task).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from cdm.splits import SPLITS, assign_splits, split_fingerprint

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


def latest_joined(
    metadata: pd.DataFrame, manifest: pd.DataFrame, landed: pd.DataFrame
) -> pd.DataFrame:
    """Latest metadata row per image, with the manifest's SHA-256 and the landed file's.

    The pipeline does this step in Spark; tests use this pandas version on the same inputs.
    ``landed`` has ``source``, ``isic_id`` and the ``sha256`` computed from the file in bronze.
    """
    key = ["source", "isic_id"]
    latest = metadata.sort_values("run_id").drop_duplicates(key, keep="last")
    recorded = (
        manifest.sort_values("run_id")
        .drop_duplicates(key, keep="last")[[*key, "sha256"]]
        .rename(columns={"sha256": "manifest_sha256"})
    )
    frame = latest.merge(recorded, on=key, how="left").merge(
        landed[[*key, "sha256"]].drop_duplicates(key), on=key, how="left"
    )
    return frame.reset_index(drop=True)


def with_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add the ``split`` column. Shared by the pipeline (inside applyInPandas) and tests."""
    frame = frame.reset_index(drop=True)
    return frame.assign(split=assign_splits(frame, seed))


def silver_frame(
    metadata: pd.DataFrame, manifest: pd.DataFrame, landed: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """The frame the gates check: every ingest run's metadata and manifest rows, plus the
    SHA-256 of each landed file, reduced to the latest row per image and split."""
    return with_splits(latest_joined(metadata, manifest, landed), seed)


def split_computed(frame: pd.DataFrame) -> Gate:
    """The lesion split could be computed (stratification needs enough lesions per class)."""
    bad = frame[frame["split"] == "unsplittable"]["isic_id"]
    return Gate("split_computed", len(bad), _examples(bad))


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
    """Every image of a trained source has a label: anything not marked ``score``."""
    missing = frame[(frame["split"] != "score") & frame["label"].isna()]["isic_id"]
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


def every_image_landed(frame: pd.DataFrame) -> Gate:
    """Every metadata row has a file in bronze. A file that fails to decode still counts as
    landed: it goes to quarantine, which is reported, not failed."""
    missing = frame[frame["sha256"].isna()]["isic_id"]
    return Gate("every_image_landed", len(missing), _examples(missing))


def files_match_manifest(frame: pd.DataFrame) -> Gate:
    """The SHA-256 computed in the pipeline equals the one the ingest task recorded."""
    landed = frame[frame["sha256"].notna()]
    wrong = landed[landed["sha256"] != landed["manifest_sha256"]]["isic_id"]
    return Gate("files_match_manifest", len(wrong), _examples(wrong))


def run_all(frame: pd.DataFrame, config: Mapping[str, object]) -> list[Gate]:
    ham_cfg = config["ham10000"]
    assert isinstance(ham_cfg, Mapping)
    allowed = config["license_allowed"]
    assert isinstance(allowed, list)
    ham = frame[frame["source"] == "ham10000"]
    return [
        split_computed(frame),
        lesions_in_one_split(frame),
        split_matches_phase0(ham, str(ham_cfg["split_fingerprint"])),
        class_counts(ham, dict(ham_cfg["class_counts"])),
        no_missing_labels(frame),
        no_ham_in_sites(frame),
        license_policy(frame, allowed),
        mapping_reproduces_ham_labels(ham),
        no_duplicate_files(frame[frame["sha256"].notna()]),
        every_image_landed(frame),
        files_match_manifest(frame),
    ]
