"""Phase 1 gates: clean data passes every gate; each kind of broken input fails its own gate."""

from __future__ import annotations

import io
from collections.abc import Callable

import pandas as pd
import pytest
from PIL import Image

from cdm.gates import run_all
from cdm.images import SIZE, DecodeError, decode_resize
from cdm.splits import assign_splits, lesion_split, split_fingerprint


def clean_frame() -> tuple[pd.DataFrame, dict[str, object]]:
    ham_rows = []
    for dx, n_lesions in {"nv": 12, "mel": 6, "bcc": 6}.items():
        for k in range(n_lesions):
            for j in range(2):
                ham_rows.append(
                    {"isic_id": f"ISIC_H{dx}{k}{j}", "lesion_id": f"HAM_{dx}{k}", "dx": dx}
                )
    ham = pd.DataFrame(ham_rows)
    split = lesion_split(ham, seed=0)
    ham = ham.assign(
        split=split,
        label=ham["dx"],
        isic_label=ham["dx"],
        source="ham10000",
        license="CC-BY-NC",
        attribution="MILK study team",
    ).drop(columns="dx")
    site = pd.DataFrame(
        {
            "isic_id": ["ISIC_S1", "ISIC_S2", "ISIC_S3"],
            "lesion_id": ["IL_1", "IL_1", "IL_2"],
            "split": "score",
            "label": ["nv", "nv", None],
            "isic_label": ["nv", "nv", None],
            "source": "barcelona",
            "license": ["CC-0", "CC-0", "CC-BY"],
            "attribution": ["Anonymous", "Anonymous", "Hospital"],
        }
    )
    frame = pd.concat([ham, site], ignore_index=True)
    frame["sha256"] = [f"{i:064x}" for i in range(len(frame))]
    frame["manifest_sha256"] = frame["sha256"]
    config: dict[str, object] = {
        "license_allowed": ["CC-0", "CC-BY", "CC-BY-NC"],
        "ham10000": {
            "split_fingerprint": split_fingerprint(ham, ham["split"]),
            "class_counts": ham["label"].value_counts().to_dict(),
        },
    }
    return frame, config


def failing(frame: pd.DataFrame, config: dict[str, object]) -> set[str]:
    return {g.name for g in run_all(frame, config) if g.violations}


def test_clean_data_passes_every_gate() -> None:
    frame, config = clean_frame()
    assert failing(frame, config) == set()


def _first_ham(frame: pd.DataFrame) -> int:
    return int(frame.index[frame["source"] == "ham10000"][0])


def leak_a_lesion(f: pd.DataFrame) -> None:
    i = _first_ham(f)
    other = next(s for s in ("train", "val", "test") if s != f.at[i, "split"])
    f.at[i, "split"] = other


def drop_a_label(f: pd.DataFrame) -> None:
    f.at[_first_ham(f), "label"] = None


def ham_image_in_a_site(f: pd.DataFrame) -> None:
    f.loc[len(f)] = {**f.loc[_first_ham(f)].to_dict(), "source": "barcelona", "split": "score",
                     "sha256": "f" * 64, "manifest_sha256": "f" * 64}  # fmt: skip


def unknown_license(f: pd.DataFrame) -> None:
    f.at[len(f) - 1, "license"] = "All rights reserved"


def cc_by_without_attribution(f: pd.DataFrame) -> None:
    f.at[len(f) - 1, "attribution"] = None


def duplicate_file(f: pd.DataFrame) -> None:
    """The same file landed twice under two IDs, faithfully recorded in the manifest."""
    f.at[len(f) - 1, "sha256"] = f.at[len(f) - 1, "manifest_sha256"] = f.at[0, "sha256"]


def file_never_landed(f: pd.DataFrame) -> None:
    f.at[len(f) - 1, "sha256"] = None


def file_changed_after_ingest(f: pd.DataFrame) -> None:
    f.at[len(f) - 1, "sha256"] = "e" * 64


def wrong_mapping(f: pd.DataFrame) -> None:
    f.at[_first_ham(f), "isic_label"] = "bkl"


@pytest.mark.parametrize(
    ("breakage", "expected"),
    [
        (leak_a_lesion, {"no_lesion_in_two_splits", "split_fingerprint_matches_phase0"}),
        (
            drop_a_label,
            {"no_missing_labels", "class_counts_as_expected", "diagnosis_mapping_matches_ham10000"},
        ),  # fmt: skip
        (ham_image_in_a_site, {"no_ham10000_image_in_a_site"}),
        (unknown_license, {"license_and_attribution"}),
        (cc_by_without_attribution, {"license_and_attribution"}),
        (duplicate_file, {"no_duplicate_files"}),
        (wrong_mapping, {"diagnosis_mapping_matches_ham10000"}),
        (file_never_landed, {"every_image_landed"}),
        (file_changed_after_ingest, {"files_match_manifest"}),
    ],
)
def test_each_breakage_fails_its_gate(
    breakage: Callable[[pd.DataFrame], None], expected: set[str]
) -> None:
    frame, config = clean_frame()
    breakage(frame)
    assert failing(frame, config) == expected


def jpeg(width: int, height: int, quality: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (120, 60, 30)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def test_decode_resize_centre_crops_and_records_compression() -> None:
    strong, light = decode_resize(jpeg(600, 450, 50)), decode_resize(jpeg(600, 450, 100))
    with Image.open(io.BytesIO(strong.png)) as img:
        assert img.size == (SIZE, SIZE) and img.format == "PNG"
    assert (strong.width, strong.height) == (600, 450)
    assert strong.jpeg_quant_mean is not None and light.jpeg_quant_mean is not None
    assert strong.jpeg_quant_mean > light.jpeg_quant_mean


@pytest.mark.parametrize("data", [b"not an image", jpeg(64, 64, 90)[:200]])
def test_undecodable_bytes_raise_for_quarantine(data: bytes) -> None:
    with pytest.raises(DecodeError):
        decode_resize(data)


def test_a_split_that_cannot_be_stratified_fails_a_named_gate() -> None:
    frame, config = clean_frame()
    ham = frame["source"] == "ham10000"
    keep = frame[~ham | (frame["label"] != "bcc")].index.tolist()
    one_bcc = frame[ham & (frame["label"] == "bcc")].index[:1].tolist()
    small = frame.loc[keep + one_bcc].reset_index(drop=True)
    small["split"] = assign_splits(small, seed=0)
    assert "split_computed" in failing(small, config)
