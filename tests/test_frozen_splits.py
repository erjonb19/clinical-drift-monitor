"""v2 data: earlier (v1) lesions keep their split, only new lesions are assigned, and
HAM10000's Phase 0 split never moves."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cdm.gates import frozen_splits_kept, with_splits
from cdm.splits import HAM, SPLITS, DataError, assign_splits, load_frozen

REPO = Path(__file__).resolve().parents[1]
CLASSES = ("nv", "mel", "bcc")


def site_frame(lesions_per_class: int, prefix: str, source: str = "barcelona") -> pd.DataFrame:
    rows = [
        {"isic_id": f"ISIC_{prefix}{dx}{k}{j}", "lesion_id": f"{prefix}_{dx}{k}", "label": dx,
         "source": source}
        for dx in CLASSES
        for k in range(lesions_per_class)
        for j in range(2)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def ham_frame() -> pd.DataFrame:
    return site_frame(20, "H", source=HAM)


def frozen_from(frame: pd.DataFrame, split: pd.Series, source: str) -> dict[str, dict[str, str]]:
    rows = frame.assign(split=split)
    rows = rows[(rows["source"] == source) & rows["split"].isin(SPLITS)]
    return {source: dict(zip(rows["lesion_id"], rows["split"], strict=True))}


def test_frozen_lesions_keep_their_split_and_only_new_lesions_are_assigned() -> None:
    v1 = pd.concat([ham_frame(), site_frame(20, "OLD")], ignore_index=True)
    v1_split = assign_splits(v1, seed=0, clients=(HAM, "barcelona"))
    frozen = frozen_from(v1, v1_split, "barcelona")

    v2 = pd.concat([v1, site_frame(20, "NEW")], ignore_index=True)
    v2_split = assign_splits(v2, seed=0, clients=(HAM, "barcelona"), frozen=frozen)

    old = v2["lesion_id"].str.startswith("OLD")
    pd.testing.assert_series_equal(v2_split[old], v1_split[v1["lesion_id"].str.startswith("OLD")])
    new = v2["lesion_id"].str.startswith("NEW")
    assert set(v2_split[new]) == set(SPLITS)  # new lesions are split, among themselves
    per_lesion = v2.assign(split=v2_split).groupby("lesion_id")["split"].nunique()
    assert per_lesion.max() == 1
    # HAM10000 is untouched by the new site data.
    ham = v2["source"] == HAM
    pd.testing.assert_series_equal(v2_split[ham], v1_split[v1["source"] == HAM])


def test_without_new_lesions_the_split_is_exactly_v1() -> None:
    v1 = pd.concat([ham_frame(), site_frame(20, "OLD")], ignore_index=True)
    v1_split = assign_splits(v1, seed=0, clients=(HAM, "barcelona"))
    frozen = frozen_from(v1, v1_split, "barcelona")
    again = assign_splits(v1, seed=0, clients=(HAM, "barcelona"), frozen=frozen)
    pd.testing.assert_series_equal(again, v1_split)


def test_ham10000_can_never_be_frozen() -> None:
    with pytest.raises(DataError):
        assign_splits(ham_frame(), seed=0, frozen={HAM: {"H_nv0": "train"}})


def test_load_frozen_reads_the_csv_and_rejects_a_bad_split(tmp_path: Path) -> None:
    good = tmp_path / "ok.csv"
    good.write_text("source,lesion_id,split\nbarcelona,IL_1,train\nmsk,IL_2,test\n")
    assert load_frozen(good) == {"barcelona": {"IL_1": "train"}, "msk": {"IL_2": "test"}}
    bad = tmp_path / "bad.csv"
    bad.write_text("source,lesion_id,split\nbarcelona,IL_1,score\n")
    with pytest.raises(DataError):
        load_frozen(bad)


def test_gate_fails_when_a_frozen_lesion_moves_or_disappears() -> None:
    frame = site_frame(3, "OLD").assign(split="train")
    frozen = {"barcelona": {lesion: "train" for lesion in frame["lesion_id"].unique()}}
    assert frozen_splits_kept(frame, frozen).violations == 0
    moved = frame.copy()
    moved.loc[moved["lesion_id"] == "OLD_nv0", "split"] = "test"
    assert frozen_splits_kept(moved, frozen).violations == 1
    missing = frame[frame["lesion_id"] != "OLD_mel1"]
    assert frozen_splits_kept(missing, frozen).violations == 1
    assert frozen_splits_kept(frame, {}).violations == 0  # nothing frozen: nothing to check


def test_with_splits_reads_frozen_lesions_from_the_config() -> None:
    v1 = pd.concat([ham_frame(), site_frame(20, "OLD")], ignore_index=True)
    lesions = {"barcelona": {"OLD_nv0": "test"}}
    config = {
        "ham10000": {"split_seed": 0},
        "sites": {"barcelona": {"role": "client"}},
        "frozen_splits": {"file": "x.csv", "lesions": lesions},
    }
    out = with_splits(v1.assign(isic_label=v1["label"], diagnosis_3=None), config)
    assert set(out.loc[out["lesion_id"] == "OLD_nv0", "split"]) == {"test"}
    assert frozen_splits_kept(out, lesions).violations == 0


def test_committed_v1_splits_match_v1_and_the_config_points_to_them() -> None:
    config = json.loads((REPO / "config" / "sites.json").read_text(encoding="utf-8"))
    frozen = load_frozen(REPO / "config" / config["frozen_splits"]["file"])
    assert set(frozen) == {"barcelona", "msk"}  # only the ISIC client sites, never HAM10000
    assert {s: len(v) for s, v in frozen.items()} == {"barcelona": 1229, "msk": 2421}
    cap = config["sites"]["barcelona"]["cap"]
    assert (cap["max_images"], cap["fingerprint"]) == (10000, "4129a90a0e5f3aed")
    assert config["ham10000"]["split_fingerprint"] == "cc2b196cd5bf58ad"
