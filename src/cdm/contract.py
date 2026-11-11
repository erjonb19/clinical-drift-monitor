"""Backlog item 18: an input contract for images, checked before scoring.

Each image is checked for format, resolution, aspect ratio, blur, exposure and (in the monthly
job only) near-duplicates. Checks with an objective rule are **enforced**: a failing image is
quarantined with its reason and not scored. Blur and exposure need thresholds calibrated on real
dermoscopic and clinical images, which has not been done, so they are **report-only** for now:
the image is scored and the flag is counted, never quarantined. The thresholds are
:data:`CONTRACT` below (in code, so the API image carries them), set by Claude on 2026-11-09
for the owner to confirm.

The public API checks format, resolution, aspect ratio, blur and exposure but never duplicates:
it keeps no hash of any upload (the owner's privacy condition, 2026-10-22).
"""

from __future__ import annotations

import copy
import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

# The pixel limit for every decoded image (the monthly job and the API alike). Pillow raises
# DecompressionBombError only above twice MAX_IMAGE_PIXELS and merely warns between one and two
# times it, so open_image checks the header's size against MAX_PIXELS itself, before any decode
# (silent failure #16).
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


# Only these decoders see bytes passed to open_image (backlog item 30): Image.open tries no
# other plugin, so a TIFF, PSD or ICO parser bug cannot be reached through the API's uploads.
# (A multi-picture JPEG is opened by the JPEG plugin as MPO, and refused before decoding.) The
# ISIC ingestion paths (cdm.images, silver, gold) still use plain Image.open.
DECODERS = ("JPEG", "PNG")


class RefusedFormat(UnidentifiedImageError):
    """The bytes look like an image in a format whose decoder is not allowed to run."""

    def __init__(self, fmt: str) -> None:
        super().__init__(f"{fmt} images are not decoded here")
        self.format = fmt


def sniff_format(data: bytes) -> str | None:
    """The format named by the first bytes, from each Pillow plugin's prefix check (``accept``).
    Those checks compare a few magic bytes and decode nothing; they only name what was refused."""
    Image.init()
    prefix = data[:16]
    for fmt, (_, accept) in Image.OPEN.items():
        try:
            if accept is not None and accept(prefix):
                return fmt
        except Exception:  # noqa: BLE001, S112 - a prefix check that fails just doesn't match
            continue
    return None


def open_image(data: bytes, formats: Iterable[str] = DECODERS) -> Image.Image:
    """Open image bytes lazily, with only the ``formats`` decoders, and refuse one over
    MAX_PIXELS before its pixels are decoded. Image.open reads only the header, so the size is
    known before ``load()``. Raises RefusedFormat for an image in another format,
    UnidentifiedImageError for bytes no check recognises, and Image.DecompressionBombError for
    an oversized image."""
    allowed = list(formats)
    try:
        img = Image.open(io.BytesIO(data), formats=allowed)
    except UnidentifiedImageError:
        fmt = sniff_format(data)
        if fmt is not None and fmt not in allowed:
            raise RefusedFormat(fmt) from None
        raise
    w, h = img.size
    if w * h > MAX_PIXELS:
        img.close()
        raise Image.DecompressionBombError(f"{w}x{h} is over the {MAX_PIXELS} pixel limit")
    return img


CONTRACT: dict[str, Any] = {
    "enforced": {
        # The formats the ISIC Archive serves; anything else is not what the model trained on.
        "formats": ["JPEG", "PNG"],
        # The model's input is 224 px (cdm.data.eval_transform resizes the shorter side to 224
        # and centre-crops); a smaller image would be upsampled, inventing detail.
        "min_side_px": 224,
        # The centre crop keeps the middle square; past 3:1 most of a frame is cut off.
        "max_aspect_ratio": 3.0,
        # 64-bit difference hashes within 4 bits are near-identical images (resized,
        # recompressed). Monthly job only; the API keeps no hashes.
        "duplicate_max_hamming": 4,
    },
    # Not calibrated: guesses until checked on the validation images, so they only flag.
    "report_only": {
        "analysis_long_side_px": 512,
        "min_laplacian_variance": 20.0,
        "min_mean_brightness": 20,
        "max_mean_brightness": 235,
        "max_clipped_share": 0.25,
    },
    # More than 10% of a month's images quarantined means the input changed: an alert.
    "spike_share": 0.1,
}


def load_config() -> dict[str, Any]:
    """A copy of :data:`CONTRACT`, so a caller cannot change the shared thresholds."""
    return copy.deepcopy(CONTRACT)


@dataclass
class Verdict:
    """What the contract found: ``quarantined`` reasons stop scoring; ``flags`` do not."""

    quarantined: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    dhash: int | None = None
    unreadable: bool = False

    @property
    def ok(self) -> bool:
        return not self.quarantined


def dhash(gray: Image.Image, size: int = 8) -> int:
    """A 64-bit difference hash: robust to resizing and small edits, for near-duplicates."""
    small = np.asarray(gray.resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian: low values mean little fine detail (a blurry image)."""
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def check(
    data: bytes, config: dict[str, Any], seen: Iterable[int] = (), hash_image: bool = True
) -> Verdict:
    """Check one image's bytes. ``seen`` holds earlier images' difference hashes (the monthly
    job passes them). The API passes ``hash_image=False``: it computes no hash of any upload.
    An image over the pixel limit (a decompression bomb) is unreadable, like a non-image. An
    image in a format outside the contract is quarantined with that reason alone, before its
    pixels are decoded."""
    v = Verdict()
    rules = config["enforced"]
    try:
        with open_image(data, rules["formats"]) as img:
            fmt = img.format
            if fmt not in rules["formats"]:  # e.g. MPO, opened by the JPEG plugin
                raise RefusedFormat(str(fmt))
            img.load()
            rgb = img.convert("RGB")
    except RefusedFormat as refused:
        v.quarantined.append(f"format: {refused.format} (allowed: {', '.join(rules['formats'])})")
        return v
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        v.quarantined.append("format: not a readable image")
        v.unreadable = True
        return v
    w, h = rgb.size
    if min(w, h) < rules["min_side_px"]:
        v.quarantined.append(f"resolution: {w}x{h}, shorter side under {rules['min_side_px']} px")
    if max(w, h) / max(1, min(w, h)) > rules["max_aspect_ratio"]:
        v.quarantined.append(f"aspect ratio: {w}x{h}, over {rules['max_aspect_ratio']}:1")
    gray = rgb.convert("L")
    if hash_image:
        v.dhash = dhash(gray)
        # A flat image hashes to all zeros (or all ones); such hashes say nothing about
        # identity, so they are never matched as duplicates.
        informative = v.dhash not in (0, (1 << 64) - 1)
        near = [s for s in seen if hamming(v.dhash, s) <= rules["duplicate_max_hamming"]]
        if informative and near:
            v.quarantined.append("duplicate: near-identical to an image already seen")
    report = config["report_only"]
    small = gray.copy()
    small.thumbnail((report["analysis_long_side_px"],) * 2)
    arr = np.asarray(small)
    if laplacian_variance(arr) < report["min_laplacian_variance"]:
        v.flags.append("blur: little fine detail")
    mean = float(arr.mean())
    clipped = float(np.mean((arr <= 5) | (arr >= 250)))
    if not report["min_mean_brightness"] <= mean <= report["max_mean_brightness"]:
        v.flags.append(f"exposure: mean brightness {mean:.0f}")
    if clipped > report["max_clipped_share"]:
        v.flags.append(f"exposure: {clipped:.0%} of pixels clipped")
    return v


def month_counts(verdicts: Iterable[Verdict], config: dict[str, Any]) -> dict[str, Any]:
    """Counts per reason for a month's results, and whether the quarantine share is a spike."""
    verdicts = list(verdicts)
    reasons: dict[str, int] = {}
    flags: dict[str, int] = {}
    for v in verdicts:
        for r in v.quarantined:
            key = r.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for f in v.flags:
            key = f.split(":")[0]
            flags[key] = flags.get(key, 0) + 1
    quarantined = sum(1 for v in verdicts if not v.ok)
    share = quarantined / len(verdicts) if verdicts else 0.0
    return {
        "images": len(verdicts),
        "quarantined": quarantined,
        "by_reason": reasons,
        "flagged": flags,
        "quarantine_share": share,
        "spike": bool(verdicts) and share > config["spike_share"],
    }
