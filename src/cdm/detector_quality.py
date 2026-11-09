"""Backlog item 15: how fast the monthly drift test detects a staged shift, and how often it
alarms when nothing changed (results/phase3/detector_quality/spec.json, committed first).

Every shift here is **staged**: real in-distribution test images altered in a known way (or
resampled by age). The scores come from the shipped bundle; this module only applies the
shifts to images and simulates 12-month sequences from per-image scores, so the costly part
(scoring) happens once per image and setting, and the simulation is cheap and tested.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter

from cdm.monitor import drift_test
from cdm.post_replay import label_free_signals, month_alert
from cdm.slo import wilson

Array = NDArray[np.float64]
INPUT = 224  # the model's input size; shifts are applied at it, as the spec says
MONTHS, START, PER_MONTH = 12, 4, 300
OLD_AGE = 75  # the repo's oldest age band


def to_input(image: Image.Image) -> Image.Image:
    """The geometric part of cdm.data.eval_transform (the same torchvision Resize and
    CenterCrop), so the shifts act on exactly the pixels the model would see."""
    from torchvision import transforms

    step = transforms.Compose([transforms.Resize(INPUT), transforms.CenterCrop(INPUT)])
    out: Image.Image = step(image.convert("RGB"))
    return out


def apply_shift(image: Image.Image, kind: str, size: float) -> Image.Image:
    """One staged shift on a 224 px input image."""
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(size)))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(size))
    if kind == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(size))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"unknown image shift {kind!r}")


def draw_month(
    rng: np.random.Generator,
    n_pool: int,
    old: NDArray[np.bool_] | None = None,
    old_share: float | None = None,
    n: int = PER_MONTH,
    known: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.int64], bool]:
    """Indices of one month's images, without replacement within the month. With ``old_share``
    (the age-mix shift) that share of the month comes from patients aged 75 or over and the rest
    from younger patients; images of unknown age (``known`` False) are left out of an age-mixed
    month. If either age group is too small for its share, that group is drawn with
    replacement, and the second value says so."""
    if old_share is None or old is None:
        return rng.choice(n_pool, size=n, replace=False), False
    known = np.ones(n_pool, dtype=bool) if known is None else known
    k = round(old_share * n)
    olds, youngs = np.flatnonzero(old & known), np.flatnonzero(~old & known)
    replaced = k > len(olds) or n - k > len(youngs)
    a = rng.choice(olds, size=k, replace=k > len(olds))
    b = rng.choice(youngs, size=n - k, replace=n - k > len(youngs))
    return np.concatenate([a, b]), replaced


def run_sequence(
    seed: Sequence[int],
    baseline: Array,
    unshifted: Array,
    shifted: Array | None = None,
    share: float = 0.0,
    old: NDArray[np.bool_] | None = None,
    old_share: float | None = None,
    known: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """One 12-month sequence: months 1 to 3 unshifted, the shift from month 4 on. ``shifted``
    holds each pool image's score after the image shift (``share`` of each shifted month's
    images use it); ``old_share`` resamples shifted months by age instead. With neither it is
    the no-shift control. The months are drawn from one generator and the shifted images
    picked from another, both from ``seed``, so the same seed without the shift is that
    sequence's paired control: the same images every month, none altered."""
    month_rng = np.random.default_rng([*seed, 0])
    pick_rng = np.random.default_rng([*seed, 1])
    alerts, warnings, replaced = [], [], False
    for month in range(1, MONTHS + 1):
        shifting = month >= START
        idx, rep = draw_month(
            month_rng, len(unshifted), old, old_share if shifting else None, known=known
        )
        replaced |= rep
        scores = unshifted[idx].copy()
        if shifting and shifted is not None and share > 0:
            pick = pick_rng.choice(len(idx), size=round(share * len(idx)), replace=False)
            scores[pick] = shifted[idx[pick]]
        test = drift_test(baseline, scores)
        alerts.append(bool(test["alert"]))
        warnings.append(bool(test["warning"]))
    return {"alerts": alerts, "warnings": warnings, "with_replacement": replaced}


def time_to_detect(alerts: Sequence[bool]) -> int | None:
    """Months from the shift's start to the first alert (1 = the first shifted month)."""
    for month in range(START, MONTHS + 1):
        if alerts[month - 1]:
            return month - (START - 1)
    return None


def summarise(
    sequences: Sequence[Mapping[str, Any]],
    control: bool = False,
    paired: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Time to detect over the repeats, and the alarm rate on the unshifted months (months 1 to
    3; every month for the control, which never shifts). ``paired`` are the cell's paired
    controls (the same seeds, no shift): how often they alert in months 4 to 12 is the chance
    level a detection is read against."""
    unshifted = slice(0, MONTHS) if control else slice(0, START - 1)
    out: dict[str, Any] = {"repeats": len(sequences)}
    for key in ("alerts", "warnings"):
        flags = [f for s in sequences for f in s[key][unshifted]]
        hits = int(sum(flags))
        low, high = wilson(hits, len(flags))
        out[f"unshifted_month_{key[:-1]}_rate"] = {
            "months": len(flags), "rate": hits / len(flags), "wilson_95": [low, high]}  # fmt: skip
    if not control:
        times = [time_to_detect(s["alerts"]) for s in sequences]
        found = [t for t in times if t is not None]
        out["share_detected"] = len(found) / len(times)
        out["months_to_detect"] = (
            {
                "median": float(np.median(found)),
                "p10": float(np.percentile(found, 10)),
                "p90": float(np.percentile(found, 90)),
            }
            if found
            else None
        )
    else:
        out["months_after_month_3_alerting"] = float(
            np.mean([a for s in sequences for a in s["alerts"][START - 1 :]])
        )
    if paired is not None:
        later = [a for s in paired for a in s["alerts"][START - 1 :]]
        low, high = wilson(int(sum(later)), len(later))
        out["paired_control_months_4_to_12_alert_rate"] = {
            "months": len(later),
            "rate": float(np.mean(later)),
            "wilson_95": [low, high],
        }
        chance = [time_to_detect(s["alerts"]) for s in paired]
        out["paired_control_share_alerting_by_month_12"] = sum(t is not None for t in chance) / len(
            chance
        )
    out["any_month_with_replacement"] = any(s.get("with_replacement") for s in sequences)
    return out


def post_replay_false_alarms(
    rng: np.random.Generator,
    baseline: Mapping[str, Any],
    pool: Mapping[str, Any],
    n_classes: int,
    months: int,
) -> dict[str, Any]:
    """How often the post-replay rule (any of four label-free signals) alarms on unshifted
    months drawn from the pool. ``baseline`` and ``pool`` hold ``ood``, ``features``,
    ``predicted`` and ``top_prob`` per image."""
    hits = 0
    by_signal = dict.fromkeys(("ood", "embedding", "prediction_mix", "confidence"), 0)
    for _ in range(months):
        idx = rng.choice(len(pool["ood"]), size=PER_MONTH, replace=False)
        month = {k: np.asarray(v)[idx] for k, v in pool.items()}
        signals = label_free_signals(dict(baseline), month, n_classes)
        hits += month_alert(signals)
        for k in by_signal:
            by_signal[k] += bool(signals[k]["alert"])
    low, high = wilson(hits, months)
    return {"months": months, "rate": hits / months, "wilson_95": [low, high],
            "by_signal": {k: v / months for k, v in by_signal.items()}}  # fmt: skip


SEED = 20261109


def cell_seed(index: int, seed: int = SEED) -> np.random.Generator:
    """The spec's seeding: numpy default_rng(20261109) combined with the cell's index."""
    return np.random.default_rng([seed, index])


def sequence_seed(cell: int, repeat: int) -> list[int]:
    """One sequence's seed: the spec's seed, the cell's index and the repeat."""
    return [SEED, cell, repeat]


def check_direction(baseline: Array, shifted_means: Mapping[str, float]) -> list[str]:
    """The shifted settings whose mean OOD score is *below* the baseline's. Scores should rise
    with distortion, so any listed here are reported for a look, as the honesty rules ask of a
    result that points the wrong way; the run does not fail on it."""
    base = float(np.mean(baseline))
    return [name for name, m in shifted_means.items() if math.isfinite(m) and m < base]
