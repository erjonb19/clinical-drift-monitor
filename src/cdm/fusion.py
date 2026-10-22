"""Research Phase E: does adding metadata to frozen image embeddings help?

The same head (:mod:`cdm.probe`) on the same frozen embeddings, with and without metadata:
image only, metadata only, and image plus metadata. Two embedding sets: BiomedCLIP
(Phase B) and the ImageNet-pretrained EfficientNet-B0 in ``gold_embeddings`` (not the shipped
model's own features). Metadata is age, sex and body site, with missing-value indicators;
the encoding is fit on the training split only. The image's source is never an input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score

from cdm.probe import GRID, ProbeConfig, train_probe

FEATURE_SETS = ("biomedclip", "imagenet_effb0")
VARIANTS = (
    "metadata_only",
    *(f"image_only__{f}" for f in FEATURE_SETS),
    *(f"fusion__{f}" for f in FEATURE_SETS),
)
UNKNOWN = "unknown"


def _concept(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN
    v = int(float(str(value)))
    return UNKNOWN if v == 0 else str(v)


@dataclass
class MetadataEncoder:
    """Age (standardized, 0 where missing, plus a missing flag) and one-hot sex and body-site
    concept IDs. Categories are those seen in training; anything else is "unknown"."""

    age_mean: float = 0.0
    age_std: float = 1.0
    sexes: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)

    def fit(self, meta: pd.DataFrame) -> MetadataEncoder:
        age = pd.to_numeric(meta["age_approx"], errors="coerce")
        self.age_mean, self.age_std = float(age.mean()), float(age.std(ddof=0)) or 1.0
        self.sexes = sorted({_concept(v) for v in meta["gender_concept_id"]} | {UNKNOWN})
        self.sites = sorted({_concept(v) for v in meta["anatomic_site_concept_id"]} | {UNKNOWN})
        return self

    @property
    def columns(self) -> list[str]:
        return (
            ["age", "age_missing"]
            + [f"sex:{s}" for s in self.sexes]
            + [f"site:{s}" for s in self.sites]
        )

    def transform(self, meta: pd.DataFrame) -> NDArray[np.float32]:
        if not self.sexes:
            raise RuntimeError("fit the encoder on training metadata first")
        age = pd.to_numeric(meta["age_approx"], errors="coerce").to_numpy(np.float64)
        missing = np.isnan(age)
        out = np.zeros((len(meta), len(self.columns)), np.float32)
        out[:, 0] = np.where(missing, 0.0, (age - self.age_mean) / self.age_std)
        out[:, 1] = missing
        for j, values, cats in (
            (2, meta["gender_concept_id"], self.sexes),
            (2 + len(self.sexes), meta["anatomic_site_concept_id"], self.sites),
        ):
            index = {c: i for i, c in enumerate(cats)}
            for row, v in enumerate(values):
                c = _concept(v)
                out[row, j + index.get(c, index[UNKNOWN])] = 1.0
        return out


def inputs(
    variant: str, images: dict[str, NDArray[np.float32]], metadata: NDArray[np.float32]
) -> NDArray[np.float32]:
    """The head's input for one variant: an embedding set, the metadata, or both side by side."""
    if variant == "metadata_only":
        return metadata
    kind, feature_set = variant.split("__")
    image = images[feature_set]
    if len(image) != len(metadata):
        raise ValueError(f"{len(image)} embeddings, {len(metadata)} metadata rows")
    return image if kind == "image_only" else np.concatenate([image, metadata], axis=1)


def check_order(ids: Sequence[str], other: Sequence[str], what: str) -> None:
    """Two tables must list the same images in the same order before they are joined by row."""
    if list(ids) != list(other):
        raise ValueError(f"{what}: the image order differs")


SHIPPED = "shipped_effb0"
SHIPPED_VARIANTS = (f"image_only__{SHIPPED}", f"fusion__{SHIPPED}")


def select_per_seed(
    x_train: Mapping[int, NDArray[np.float32]],
    y_train: NDArray[np.int64],
    x_val: Mapping[int, NDArray[np.float32]],
    y_val: NDArray[np.int64],
    n_classes: int,
    grid: Mapping[str, Sequence[float]] = GRID,
) -> tuple[ProbeConfig, list[dict[str, Any]]]:
    """Like :func:`cdm.probe.select_config`, for inputs that differ by seed: the shipped
    model's features come from three checkpoints, so probe seed ``s`` trains on checkpoint
    ``s``'s features. The grid point with the highest mean validation balanced accuracy over
    the seeds wins (ties: the earlier point)."""
    if set(x_train) != set(x_val):
        raise ValueError("train and validation inputs must cover the same seeds")
    table = []
    for lr, wd in product(grid["lr"], grid["weight_decay"]):
        cfg = ProbeConfig(lr=lr, weight_decay=wd)
        scores = [
            float(
                balanced_accuracy_score(
                    y_val,
                    train_probe(x_train[s], y_train, n_classes, s, cfg).logits(x_val[s]).argmax(1),
                )
            )
            for s in sorted(x_train)
        ]
        table.append(
            {**asdict(cfg), "val_balanced_accuracy": scores, "mean": float(np.mean(scores))}
        )
    best = max(range(len(table)), key=lambda i: table[i]["mean"])
    return ProbeConfig(**{k: table[best][k] for k in asdict(ProbeConfig())}), table
