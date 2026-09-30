"""Silver-row datasets for Phase 2 training, and GPU-safe seeding."""

from __future__ import annotations

import io

import pandas as pd
import pytest
import torch
from PIL import Image

from cdm.silver import split_datasets
from cdm.splits import CLASSES
from cdm.train import seed_everything


def png(shade: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (224, 224), (shade, 100, 50)).save(out, format="PNG")
    return out.getvalue()


def frame() -> pd.DataFrame:
    rows = [
        {"png": png(i), "label": CLASSES[i % len(CLASSES)], "split": split}
        for i, split in enumerate(["train"] * 7 + ["val"] * 3 + ["test"] * 3)
    ]
    return pd.DataFrame(rows)


def test_split_datasets_decode_to_224_tensors_with_class_indices() -> None:
    sets = split_datasets(frame())
    assert [len(sets[s]) for s in ("train", "val", "test")] == [7, 3, 3]
    x, y = sets["test"][0]
    assert x.shape == (3, 224, 224) and y == CLASSES.index(frame().loc[10, "label"])
    assert sets["train"].labels == list(range(7))


@pytest.mark.parametrize("bad", [{"label": None}, {"split": "score"}])
def test_rows_that_cannot_train_are_refused(bad: dict[str, str | None]) -> None:
    f = frame()
    for column, value in bad.items():
        f.at[0, column] = value
    with pytest.raises(ValueError, match="unlabelled or outside"):
        split_datasets(f)


def test_seed_everything_is_strict_on_cpu_and_warn_only_on_gpu() -> None:
    seed_everything(0, 2, gpu=True)
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    seed_everything(0, 2)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.are_deterministic_algorithms_enabled()
