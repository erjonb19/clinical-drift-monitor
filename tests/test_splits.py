"""The HAM10000 split must be by lesion: no lesion's images in two splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm.data import (
    CLASSES,
    HAM_IMAGES,
    HAM_LESIONS,
    SPLITS,
    DataError,
    data_root,
    lesion_split,
    split_fingerprint,
    validate_ham,
)


def synthetic_meta(seed: int = 0) -> pd.DataFrame:
    """Imbalanced like HAM10000, with 1 to 4 images per lesion."""
    rng = np.random.default_rng(seed)
    lesions_per_class = {
        "akiec": 30,
        "bcc": 40,
        "bkl": 60,
        "df": 20,
        "mel": 60,
        "nv": 400,
        "vasc": 20,
    }
    rows = []
    for dx, n in lesions_per_class.items():
        for k in range(n):
            lesion = f"{dx}_{k}"
            for j in range(int(rng.integers(1, 5))):
                rows.append({"lesion_id": lesion, "image_id": f"{lesion}_{j}", "dx": dx})
    return pd.DataFrame(rows)


def assert_leak_free(meta: pd.DataFrame, split: pd.Series) -> None:
    assert split.notna().all()
    assert set(split) == set(SPLITS)
    per_lesion = pd.DataFrame({"lesion": meta["lesion_id"], "split": split})
    assert (per_lesion.groupby("lesion")["split"].nunique() == 1).all()
    for s in SPLITS:
        assert set(meta.loc[split == s, "dx"]) == set(CLASSES), f"{s} is missing a class"


def test_split_never_shares_a_lesion() -> None:
    meta = synthetic_meta()
    assert_leak_free(meta, lesion_split(meta, seed=0))


def test_split_fractions_are_by_lesion() -> None:
    meta = synthetic_meta()
    split = lesion_split(meta, seed=0)
    lesion_split_counts = (
        pd.DataFrame({"lesion": meta["lesion_id"], "split": split})
        .drop_duplicates()["split"]
        .value_counts(normalize=True)
    )
    assert lesion_split_counts["train"] == pytest.approx(0.70, abs=0.02)
    assert lesion_split_counts["val"] == pytest.approx(0.15, abs=0.02)
    assert lesion_split_counts["test"] == pytest.approx(0.15, abs=0.02)


def test_split_is_fixed_by_seed() -> None:
    meta = synthetic_meta()
    a, b, c = (lesion_split(meta, seed=s) for s in (0, 0, 1))
    assert split_fingerprint(meta, a) == split_fingerprint(meta, b)
    assert split_fingerprint(meta, a) != split_fingerprint(meta, c)


def test_validation_rejects_a_lesion_with_two_diagnoses() -> None:
    meta = synthetic_meta()
    meta.loc[meta.index[0], "dx"] = "mel" if meta.loc[meta.index[0], "dx"] != "mel" else "nv"
    first_lesion = meta.loc[meta.index[0], "lesion_id"]
    if (meta["lesion_id"] == first_lesion).sum() == 1:
        meta = pd.concat([meta, meta.iloc[[1]].assign(lesion_id=first_lesion, image_id="dup")])
    with pytest.raises(DataError, match="more than one diagnosis"):
        validate_ham(meta, len(meta), meta["lesion_id"].nunique())


def test_validation_rejects_unknown_diagnosis() -> None:
    meta = synthetic_meta()
    meta.loc[meta.index[0], "dx"] = "unknown"
    with pytest.raises(DataError, match="unknown or missing"):
        validate_ham(meta, len(meta), meta["lesion_id"].nunique())


REAL_META = data_root() / "ham10000" / "HAM10000_metadata.csv"


@pytest.mark.skipif(not REAL_META.exists(), reason="HAM10000 metadata not downloaded")
def test_real_ham10000_split_is_leak_free() -> None:
    meta = pd.read_csv(REAL_META)
    validate_ham(meta, HAM_IMAGES, HAM_LESIONS)
    assert_leak_free(meta, lesion_split(meta, seed=0))
