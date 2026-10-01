"""Decode, check and resize images for the silver layer. Pillow only: no torch, no Spark."""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageStat, UnidentifiedImageError

SIZE = 224


class DecodeError(ValueError):
    """The bytes are not a readable image. Silver quarantines these rather than dropping them."""


@dataclass(frozen=True)
class Decoded:
    png: bytes  # SIZE x SIZE RGB, lossless so no second round of JPEG artifacts
    width: int
    height: int
    jpeg_quant_mean: float | None  # mean of the luminance quantisation table; None if not JPEG


def jpeg_quant_mean(img: Image.Image) -> float | None:
    """Mean of the first JPEG quantisation table: about 1 near-lossless, larger means stronger
    compression. Recorded per image because compression can differ by site and act as a
    hidden site signature for a drift detector."""
    tables = getattr(img, "quantization", None)
    if not tables:
        return None
    first = [int(v) for v in tables[0]]
    return round(float(sum(first)) / len(first), 2)


def decode_resize(data: bytes) -> Decoded:
    """Shorter side to SIZE, then centre crop to SIZE x SIZE, as in Phase 0's eval transform."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            quant = jpeg_quant_mean(img)
            width, height = img.size
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise DecodeError(f"cannot decode image: {exc}") from exc
    scale = SIZE / min(width, height)
    resized = rgb.resize(
        (max(SIZE, round(width * scale)), max(SIZE, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = (resized.width - SIZE) // 2
    top = (resized.height - SIZE) // 2
    cropped = resized.crop((left, top, left + SIZE, top + SIZE))
    out = io.BytesIO()
    cropped.save(out, format="PNG")
    return Decoded(out.getvalue(), width, height, quant)


def corner_brightness(png: bytes, patch: int = 16) -> float:
    """Mean grey level (0 to 255) of the four corner patches of an image.

    Dermoscopy images framed by a black circular vignette have dark corners; images filling
    the frame do not. Used to compare how sites frame their images.
    """
    with Image.open(io.BytesIO(png)) as img:
        grey = img.convert("L")
        w, h = grey.size
        boxes = [(0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)]
        means = [ImageStat.Stat(grey.crop((x, y, x + patch, y + patch))).mean[0] for x, y in boxes]
    return round(float(sum(means) / len(means)), 1)
