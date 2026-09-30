"""Lesion-level splits for HAM10000 and ISIC sites. No torch, so Spark pipelines can import it."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SPLITS = ("train", "val", "test")
HAM_IMAGES = 10_015
HAM_LESIONS = 7_470


class DataError(RuntimeError):
    """Input data is not what the pipeline was built for. Raised, never logged and skipped."""


def validate_ham(meta: pd.DataFrame, expected_images: int, expected_lesions: int) -> None:
    if len(meta) != expected_images or meta["image_id"].nunique() != expected_images:
        raise DataError(f"expected {expected_images} unique images, got {len(meta)} rows")
    if meta["lesion_id"].nunique() != expected_lesions:
        raise DataError(f"expected {expected_lesions} lesions, got {meta['lesion_id'].nunique()}")
    unknown = set(meta["dx"]) - set(CLASSES)
    if unknown or meta["dx"].isna().any():
        raise DataError(f"unknown or missing diagnoses: {sorted(map(str, unknown))}")
    mixed = meta.groupby("lesion_id")["dx"].nunique()
    if (mixed > 1).any():
        raise DataError(f"lesions with more than one diagnosis: {list(mixed[mixed > 1].index)}")


def lesion_split(
    meta: pd.DataFrame, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15
) -> pd.Series:
    """Assign every image to train, val or test by lesion, stratified by diagnosis.

    Splitting images instead of lesions leaks: HAM10000 has several images of one lesion, and
    a model that has seen one of them is being tested on a near copy.
    """
    lesions = meta.groupby("lesion_id")["dx"].first()
    train_ids, rest_ids = train_test_split(
        lesions.index,
        test_size=val_frac + test_frac,
        stratify=lesions.to_numpy(),
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        rest_ids,
        test_size=test_frac / (val_frac + test_frac),
        stratify=lesions[rest_ids].to_numpy(),
        random_state=seed,
    )
    assignment = {
        **dict.fromkeys(train_ids, "train"),
        **dict.fromkeys(val_ids, "val"),
        **dict.fromkeys(test_ids, "test"),
    }
    return meta["lesion_id"].map(assignment).rename("split")


def split_fingerprint(meta: pd.DataFrame, split: pd.Series) -> str:
    """A short hash of the lesion-to-split assignment, recorded with every result."""
    pairs = sorted(set(zip(meta["lesion_id"], split, strict=True)))
    text = "\n".join(f"{lesion},{s}" for lesion, s in pairs)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


HAM = "ham10000"


def site_labels(frame: pd.DataFrame, unlabelled_diagnoses: Iterable[str]) -> pd.Series:
    """Labels after the site label rule: at every source except HAM10000, images whose ISIC
    ``diagnosis_3`` is in ``unlabelled_diagnoses`` lose their label and become scoring-only.

    "Squamous cell carcinoma, NOS" is the case: ISIC uses it for HAM10000's intraepithelial
    carcinomas (akiec), but at other sites it can be invasive SCC, which akiec excludes.
    HAM10000 keeps its own labels, which come from HAM10000's metadata.
    """
    ambiguous = (frame["source"] != HAM) & frame.get(
        "diagnosis_3", pd.Series(None, index=frame.index)
    ).isin(list(unlabelled_diagnoses))
    return frame["label"].where(~ambiguous, None)


def _split_source(rows: pd.DataFrame, seed: int, unlabelled: str) -> pd.Series:
    split = pd.Series(unlabelled, index=rows.index)
    labelled = rows[rows["label"].notna()]
    if len(labelled):
        try:
            lesions = labelled[["lesion_id", "label"]].rename(columns={"label": "dx"})
            assigned = lesion_split(lesions, seed=seed)
        except ValueError:
            # Too few lesions of some class to stratify: the split_computed gate names it.
            assigned = pd.Series("unsplittable", index=labelled.index)
        split.loc[labelled.index] = assigned
    return split


def assign_splits(frame: pd.DataFrame, seed: int, clients: Iterable[str] = (HAM,)) -> pd.Series:
    """A lesion-level train, val and test split for each client source; ``score`` otherwise.

    Each client source is split on its own rows, so HAM10000's split is exactly Phase 0's
    whatever other sources are present. Unlabelled HAM10000 images get ``unlabelled`` (an
    error the missing-label gate names); unlabelled site images get ``score``, because sites
    have images outside HAM10000's classes by design. A split that cannot be stratified gives
    ``unsplittable``, which the split_computed gate names.
    """
    split = pd.Series("score", index=frame.index, name="split")
    for source in clients:
        rows = frame[frame["source"] == source]
        if len(rows):
            unlabelled = "unlabelled" if source == HAM else "score"
            split.loc[rows.index] = _split_source(rows, seed, unlabelled)
    return split
