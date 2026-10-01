"""Phase 0 data: HAM10000 in-distribution, PathMNIST (near) and CIFAR-10 (far) out-of-distribution.

Every download is checked against a published checksum, and HAM10000 is split by lesion so
no lesion's images land in two splits. Nothing here is committed: files live under a data
root outside the repository (``CDM_DATA``, default ``./data``).
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Re-exported: the split code lives in cdm.splits, which Spark pipelines import without torch.
from cdm.splits import (
    CLASSES as CLASSES,
)
from cdm.splits import (
    HAM_IMAGES as HAM_IMAGES,
)
from cdm.splits import (
    HAM_LESIONS as HAM_LESIONS,
)
from cdm.splits import (
    SPLITS as SPLITS,
)
from cdm.splits import (
    DataError as DataError,
)
from cdm.splits import (
    lesion_split as lesion_split,
)
from cdm.splits import (
    split_fingerprint as split_fingerprint,
)
from cdm.splits import (
    validate_ham as validate_ham,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    md5: str


_DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{}"

# Harvard Dataverse doi:10.7910/DVN/DBW86T, version 4.0. The metadata file is "ingested" by
# Dataverse, so without ?format=original it serves a re-exported, quoted TSV whose checksum
# does not match the published one (docs/silent-failures.md).
HAM_FILES = (
    RemoteFile(
        "HAM10000_metadata.csv",
        _DATAVERSE.format(4338392) + "?format=original",
        "8f85fb1aa29d80a2797247e434deb79d",
    ),
    RemoteFile(
        "HAM10000_images_part_1.zip",
        _DATAVERSE.format(3172585),
        "4639bfa73ab251610530a97c898e6e46",
    ),
    RemoteFile(
        "HAM10000_images_part_2.zip",
        _DATAVERSE.format(3172584),
        "da43d6cc50f6613013be07e8986b384b",
    ),
)


def data_root() -> Path:
    return Path(os.environ.get("CDM_DATA", "data"))


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(file: RemoteFile, dest: Path) -> Path:
    """Download ``file`` into ``dest`` unless present, then verify its MD5 either way."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / file.name
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(file.url, partial)
        partial.replace(path)
    actual = md5sum(path)
    if actual != file.md5:
        raise DataError(f"{path}: MD5 {actual}, expected {file.md5}. Delete it and rerun.")
    return path


def prepare_ham(root: Path) -> Path:
    """Download, verify and extract HAM10000. Returns the HAM10000 directory."""
    ham = root / "ham10000"
    images = ham / "images"
    marker = images / ".extracted"
    expected = "\n".join(f.md5 for f in HAM_FILES)
    if marker.exists() and marker.read_text() == expected:
        return ham
    for file in HAM_FILES:
        path = fetch(file, ham)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(images)
    marker.write_text(expected)
    return ham


def load_ham(ham: Path) -> pd.DataFrame:
    """Read and validate the metadata; add ``image_path`` and integer ``label``."""
    meta = pd.read_csv(ham / "HAM10000_metadata.csv")
    validate_ham(meta, expected_images=HAM_IMAGES, expected_lesions=HAM_LESIONS)
    meta["image_path"] = [str(ham / "images" / f"{i}.jpg") for i in meta["image_id"]]
    missing = [p for p in meta["image_path"] if not Path(p).exists()]
    if missing:
        raise DataError(f"{len(missing)} images listed in metadata are missing, e.g. {missing[0]}")
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return meta


def eval_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def inscribed_size(width: int, height: int, degrees: float) -> tuple[int, int]:
    """Largest axis-aligned rectangle inside a ``width`` x ``height`` image rotated by
    ``degrees``: cropping to it leaves no fill colour in the corners."""
    angle = math.radians(abs(degrees))
    if angle == 0:
        return width, height
    long_side, short_side = max(width, height), min(width, height)
    sin_a, cos_a = math.sin(angle), math.cos(angle)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        w, h = (x / sin_a, x / cos_a) if width >= height else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        w, h = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a
    return int(w), int(h)


class RandomRotation:
    """A random multiple of 90 degrees plus a small random angle, cropped so that no
    artificial black corners appear (they would mimic Barcelona's dark border).

    Uses torch's random number generator, which DataLoader workers seed.
    """

    def __init__(self, max_degrees: float = 15.0) -> None:
        self.max_degrees = max_degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        quarter = int(torch.randint(0, 4, (1,)))
        img = img.rotate(90 * quarter, expand=True)
        degrees = float(torch.empty(1).uniform_(-self.max_degrees, self.max_degrees))
        w, h = inscribed_size(img.width, img.height, degrees)
        rotated = img.rotate(degrees, resample=Image.Resampling.BILINEAR, expand=True)
        left, top = (rotated.width - w) // 2, (rotated.height - h) // 2
        return rotated.crop((left, top, left + w, top + h))


class CenterSquare:
    """Crop the centred square whose side is the image's shorter side.

    On silver stored uncropped (shorter side 384), this is the field of view that rung 0's
    224 px centre-cropped storage had, so ``view="center"`` reproduces rung 0's training view.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        return img.crop((left, top, left + side, top + side))


class MixedView:
    """Half the time the centred square, otherwise the full image (torch's seeded RNG)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        return CenterSquare()(img) if float(torch.rand(1)) < 0.5 else img


VIEWS = ("full", "center", "mix")


def train_transform(
    size: int = IMAGE_SIZE, rotate: bool = False, color_jitter: bool = False, view: str = "full"
) -> transforms.Compose:
    """Phase 0's augmentation by default; rotations, colour jitter and the field of view
    the random crops come from (``full``, ``center`` or a 50/50 ``mix``) are optional."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}: use one of {VIEWS}")
    views: dict[str, list[object]] = {"full": [], "center": [CenterSquare()], "mix": [MixedView()]}
    steps = views[view]
    if rotate:
        steps.append(RandomRotation())
    steps += [
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    if color_jitter:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class LabelledImages(Dataset[tuple[torch.Tensor, int]]):
    """A dataset of (image, label) pairs that also exposes every label, for class weights."""

    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)


class HamDataset(LabelledImages):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.paths = list(frame["image_path"])
        self.labels = [int(x) for x in frame["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            x = self.transform(img.convert("RGB"))
        return x, self.labels[idx]


class OODImages(Dataset[tuple[torch.Tensor, int]]):
    """A fixed subset of an out-of-distribution dataset, images only, label -1."""

    def __init__(self, base: Dataset[tuple[object, object]], indices: Sequence[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, _ = self.base[self.indices[idx]]
        assert isinstance(x, torch.Tensor)
        return x, -1


def fixed_subset(n_total: int, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=min(n, n_total), replace=False))


def cifar10_ood(root: Path, n: int, seed: int = 0) -> OODImages:
    base = CIFAR10(str(root / "cifar10"), train=False, download=True, transform=eval_transform())
    return OODImages(base, fixed_subset(len(base), n, seed))


def pathmnist_ood(root: Path, n: int, seed: int = 0) -> OODImages:
    """Native 224 px PathMNIST test split (medmnist verifies its MD5 on download)."""
    from medmnist import PathMNIST

    folder = root / "medmnist"
    folder.mkdir(parents=True, exist_ok=True)
    base = PathMNIST(
        split="test", size=IMAGE_SIZE, root=str(folder), download=True, transform=eval_transform()
    )
    return OODImages(base, fixed_subset(len(base), n, seed))
