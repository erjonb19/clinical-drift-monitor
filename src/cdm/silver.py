"""Training datasets from silver rows: 224 x 224 PNGs with label, split and client.

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


class PngDataset(LabelledImages):
    """Images held as PNG bytes in memory, decoded on access (in DataLoader workers)."""

    def __init__(
        self, pngs: Sequence[bytes], labels: Sequence[int], transform: transforms.Compose
    ) -> None:
        if len(pngs) != len(labels):
            raise ValueError(f"{len(pngs)} images but {len(labels)} labels")
        self.pngs = list(pngs)
        self.labels = [int(x) for x in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pngs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(io.BytesIO(self.pngs[idx])) as img:
            return self.transform(img.convert("RGB")), self.labels[idx]


def split_datasets(frame: pd.DataFrame) -> Mapping[str, PngDataset]:
    """Train (augmented), val and test datasets from labelled client rows.

    ``frame`` has ``png``, ``label`` and ``split``; rows outside train, val and test, or
    without a label, are refused rather than dropped.
    """
    bad = frame[~frame["split"].isin(SPLITS) | frame["label"].isna()]
    if len(bad):
        raise ValueError(f"{len(bad)} rows are unlabelled or outside train/val/test")
    index = {c: i for i, c in enumerate(CLASSES)}
    out = {}
    for split in SPLITS:
        part = frame[frame["split"] == split]
        transform = train_transform() if split == "train" else eval_transform()
        out[split] = PngDataset(list(part["png"]), part["label"].map(index).tolist(), transform)
    return out
