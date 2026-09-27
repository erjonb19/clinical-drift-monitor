"""Phase 0 data: HAM10000 in-distribution, PathMNIST (near) and CIFAR-10 (far) out-of-distribution.

Every download is checked against a published checksum, and HAM10000 is split by lesion so
no lesion's images land in two splits. Nothing here is committed: files live under a data
root outside the repository (``CDM_DATA``, default ``./data``).
"""

from __future__ import annotations

import hashlib
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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SPLITS = ("train", "val", "test")
HAM_IMAGES = 10_015
HAM_LESIONS = 7_470

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


class DataError(RuntimeError):
    """Input data is not what the pipeline was built for. Raised, never logged and skipped."""


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


def validate_ham(meta: pd.DataFrame, expected_images: int, expected_lesions: int) -> None:
    if len(meta) != expected_images or meta["image_id"].nunique() != expected_images:
        raise DataError(f"expected {expected_images} unique images, got {len(meta)} rows")
    if meta["lesion_id"].nunique() != expected_lesions:
        raise DataError(f"expected {expected_lesions} lesions, got {meta['lesion_id'].nunique()}")
    unknown = set(meta["dx"]) - set(CLASSES)
    if unknown or meta["dx"].isna().any():
        raise DataError(f"unknown or missing diagnoses: {sorted(map(str, unknown))}")
    mixed = meta.groupby("lesion_id")["dx"].nunique()
    if (mixed > 1).any():
        raise DataError(f"lesions with more than one diagnosis: {list(mixed[mixed > 1].index)}")


def lesion_split(
    meta: pd.DataFrame, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15
) -> pd.Series:
    """Assign every image to train, val or test by lesion, stratified by diagnosis.

    Splitting images instead of lesions leaks: HAM10000 has several images of one lesion, and
    a model that has seen one of them is being tested on a near copy.
    """
    lesions = meta.groupby("lesion_id")["dx"].first()
    train_ids, rest_ids = train_test_split(
        lesions.index,
        test_size=val_frac + test_frac,
        stratify=lesions.to_numpy(),
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        rest_ids,
        test_size=test_frac / (val_frac + test_frac),
        stratify=lesions[rest_ids].to_numpy(),
        random_state=seed,
    )
    assignment = {
        **dict.fromkeys(train_ids, "train"),
        **dict.fromkeys(val_ids, "val"),
        **dict.fromkeys(test_ids, "test"),
    }
    return meta["lesion_id"].map(assignment).rename("split")


def split_fingerprint(meta: pd.DataFrame, split: pd.Series) -> str:
    """A short hash of the lesion-to-split assignment, recorded with every result."""
    pairs = sorted(set(zip(meta["lesion_id"], split, strict=True)))
    text = "\n".join(f"{lesion},{s}" for lesion, s in pairs)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HamDataset(Dataset[tuple[torch.Tensor, int]]):
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
