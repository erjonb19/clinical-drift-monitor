"""Backlog item 18: the input contract for images (cdm.contract)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from cdm import contract

CONFIG = contract.load_config()


def encode(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def textured(w: int = 400, h: int = 300, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(60, 200, (h, w, 3), dtype=np.uint8))


def test_a_good_image_passes_without_flags() -> None:
    v = contract.check(encode(textured()), CONFIG)
    assert v.ok and v.flags == [] and v.dhash is not None


def test_each_enforced_rule_quarantines_with_its_reason() -> None:
    assert "format: not a readable image" in contract.check(b"not an image", CONFIG).quarantined
    gif = encode(textured().convert("P"), "GIF")
    assert any(r.startswith("format: GIF") for r in contract.check(gif, CONFIG).quarantined)
    small = contract.check(encode(textured(200, 150)), CONFIG)
    assert any(r.startswith("resolution") for r in small.quarantined)
    wide = contract.check(encode(textured(1000, 300)), CONFIG)
    assert any(r.startswith("aspect ratio") for r in wide.quarantined)


def test_a_near_duplicate_is_quarantined_only_when_hashes_are_given() -> None:
    img = textured(600, 450)
    first = contract.check(encode(img), CONFIG)
    resized = encode(img.resize((480, 360)))
    assert contract.check(resized, CONFIG).ok  # the API passes no hashes
    dup = contract.check(resized, CONFIG, seen=[first.dhash or 0])
    assert any(r.startswith("duplicate") for r in dup.quarantined)
    other = contract.check(encode(textured(600, 450, seed=9)), CONFIG, seen=[first.dhash or 0])
    assert other.ok


def test_blur_and_exposure_only_flag() -> None:
    flat = Image.new("RGB", (400, 400), (128, 128, 128))
    v = contract.check(encode(flat), CONFIG)
    assert v.ok and any(f.startswith("blur") for f in v.flags)
    dark = contract.check(encode(Image.new("RGB", (400, 400), (3, 3, 3))), CONFIG)
    assert dark.ok and any(f.startswith("exposure") for f in dark.flags)


def test_month_counts_and_the_spike_rule() -> None:
    good = contract.check(encode(textured()), CONFIG)
    bad = contract.check(encode(textured(100, 100)), CONFIG)
    counts = contract.month_counts([good] * 8 + [bad] * 2, CONFIG)
    assert counts["quarantined"] == 2 and counts["by_reason"] == {"resolution": 2}
    assert counts["quarantine_share"] == 0.2 and counts["spike"] is True
    assert contract.month_counts([good] * 10, CONFIG)["spike"] is False
    assert contract.month_counts([], CONFIG) == {
        "images": 0,
        "quarantined": 0,
        "by_reason": {},
        "flagged": {},
        "quarantine_share": 0.0,
        "spike": False,
    }


def test_the_config_cannot_be_changed_through_a_copy() -> None:
    copy = contract.load_config()
    copy["enforced"]["min_side_px"] = 1
    assert contract.CONTRACT["enforced"]["min_side_px"] == 224
