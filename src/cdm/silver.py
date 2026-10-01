"""Training datasets from silver rows: stored images (PNG or JPEG) with label, split and client.

Phase 2 trains on Databricks from ``workspace.cdm.silver_images``; the notebook collects the
rows and this module turns them into datasets. No Spark here, so it is tested in CI.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from cdm.data import LabelledImages, eval_transform, train_transform
from cdm.splits import CLASSES, SPLITS
from cdm.train import TrainConfig


class EncodedImages(LabelledImages):
    """Images held as encoded bytes (PNG or JPEG) in memory, decoded on access."""

    def __init__(
        self,
        images: Sequence[bytes],
        labels: Sequence[int],
        transform: transforms.Compose,
        groups: Sequence[str] | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        self.images = list(images)
        self.labels = [int(x) for x in labels]
        self.transform = transform
        self.groups = list(groups) if groups is not None else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(io.BytesIO(self.images[idx])) as img:
            return self.transform(img.convert("RGB")), self.labels[idx]


def split_datasets(
    frame: pd.DataFrame, cfg: TrainConfig | None = None
) -> Mapping[str, EncodedImages]:
    """Train (augmented per ``cfg``), val and test datasets from labelled client rows.

    ``frame`` has ``image``, ``label`` and ``split``; rows outside train, val and test, or
    without a label, are refused rather than dropped.
    """
    cfg = cfg or TrainConfig()
    bad = frame[~frame["split"].isin(SPLITS) | frame["label"].isna()]
    if len(bad):
        raise ValueError(f"{len(bad)} rows are unlabelled or outside train/val/test")
    index = {c: i for i, c in enumerate(CLASSES)}
    out = {}
    for split in SPLITS:
        part = frame[frame["split"] == split]
        transform = (
            train_transform(
                cfg.image_size, cfg.rotate, cfg.color_jitter, cfg.view, cfg.color_constancy
            )
            if split == "train"
            else eval_transform(cfg.image_size, cfg.color_constancy)
        )
        groups = list(part["client"]) if "client" in part else None
        out[split] = EncodedImages(
            list(part["image"]), part["label"].map(index).tolist(), transform, groups
        )
    return out
