"""Backlog item 18: the input contract for images (cdm.contract)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
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


def test_a_decompression_bomb_is_unreadable_not_a_crash() -> None:
    """Over twice the 50M pixel limit PIL refuses to decode; the contract reports unreadable."""
    bomb = encode(Image.new("L", (12000, 9000)), "PNG")  # 108M pixels, compresses to little
    v = contract.check(bomb, CONFIG)
    assert v.unreadable and not v.ok


def test_an_image_between_one_and_two_times_the_limit_is_unreadable() -> None:
    """Pillow only warns between 50M and 100M pixels; the contract refuses before decoding."""
    bomb = encode(Image.new("L", (10000, 6000)), "PNG")  # 60M pixels, about 58 KB
    assert len(bomb) < 100_000
    v = contract.check(bomb, CONFIG)
    assert v.unreadable and not v.ok
    at_limit = Image.new("L", (10000, 5000))  # exactly 50M pixels is allowed
    with contract.open_image(encode(at_limit, "PNG")) as img:
        assert img.size == (10000, 5000)


def test_the_api_mode_computes_no_hash() -> None:
    assert contract.check(encode(textured()), CONFIG, hash_image=False).dhash is None


def test_flat_images_are_never_duplicates_of_each_other() -> None:
    flat = encode(Image.new("RGB", (400, 400), (120, 120, 120)))
    first = contract.check(flat, CONFIG)
    assert first.dhash == 0
    assert contract.check(flat, CONFIG, seen=[0]).ok


@pytest.mark.parametrize("fmt", ["TIFF", "ICO", "BMP", "GIF"])
def test_only_the_jpeg_and_png_decoders_ever_run(fmt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backlog item 30: a TIFF, ICO, BMP or GIF upload is refused with its format named, and its
    own decoder is never called, in the contract check and in open_image (which serving uses;
    item 29's test checks that serving goes through it)."""
    Image.init()
    called: list[str] = []
    factory, accept = Image.OPEN[fmt]

    def spy(*args: Any, **kwargs: Any) -> Any:
        called.append(fmt)
        return factory(*args, **kwargs)

    monkeypatch.setitem(Image.OPEN, fmt, (spy, accept))
    image = textured(256, 256).convert("P") if fmt == "GIF" else textured(256, 256)
    data = encode(image, fmt)
    v = contract.check(data, CONFIG)
    assert v.quarantined == [f"format: {fmt} (allowed: JPEG, PNG)"] and not v.unreadable
    with pytest.raises(contract.RefusedFormat):
        contract.open_image(data)
    assert called == []


def test_jpeg_and_png_still_open() -> None:
    for fmt in ("JPEG", "PNG"):
        with contract.open_image(encode(textured(), fmt)) as img:
            assert img.format == fmt


def test_an_mpo_is_refused_before_its_pixels_are_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-picture JPEG opens through the JPEG plugin as MPO; it is quarantined from the
    header, and load() never runs."""
    buf = io.BytesIO()
    textured(256, 256).save(buf, format="MPO", save_all=True, append_images=[textured(256, 256)])
    loads: list[str] = []
    real = Image.Image.load

    def spy(self: Image.Image) -> Any:
        loads.append(str(self.format))
        return real(self)

    monkeypatch.setattr(Image.Image, "load", spy)
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile.ImageFile, "load", spy)
    v = contract.check(buf.getvalue(), CONFIG)
    assert v.quarantined == ["format: MPO (allowed: JPEG, PNG)"] and "MPO" not in loads
