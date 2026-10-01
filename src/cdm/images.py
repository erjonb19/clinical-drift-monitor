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
    image: bytes  # the stored image (PNG or JPEG, per the silver settings)
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


def decode_resize(
    data: bytes,
    size: int = SIZE,
    center_crop: bool = True,
    fmt: str = "PNG",
    quality: int = 95,
) -> Decoded:
    """Shorter side to ``size``; then, by default, centre crop to ``size`` x ``size`` and store
    as PNG, as in Phase 0's eval transform. Without the crop the whole field of view is kept,
    so training crops can come from the full image (Phase 2 tuning rung 1a)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            quant = jpeg_quant_mean(img)
            width, height = img.size
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise DecodeError(f"cannot decode image: {exc}") from exc
    scale = size / min(width, height)
    resized = rgb.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    if center_crop:
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        resized = resized.crop((left, top, left + size, top + size))
    out = io.BytesIO()
    if fmt == "JPEG":
        resized.save(out, format="JPEG", quality=quality)
    else:
        resized.save(out, format="PNG")
    return Decoded(out.getvalue(), width, height, quant)


def corner_brightness(image: bytes, patch: int = 16) -> float:
    """Mean grey level (0 to 255) of the four corner patches of an image.

    Dermoscopy images framed by a black circular vignette have dark corners; images filling
    the frame do not. Used to compare how sites frame their images.
    """
    with Image.open(io.BytesIO(image)) as img:
        grey = img.convert("L")
        w, h = grey.size
        boxes = [(0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)]
        means = [ImageStat.Stat(grey.crop((x, y, x + patch, y + patch))).mean[0] for x, y in boxes]
    return round(float(sum(means) / len(means)), 1)
