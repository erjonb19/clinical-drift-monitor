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


def test_confusion_rows_are_true_classes_in_class_order() -> None:
    import numpy as np

    from cdm.eval import confusion

    labels = np.array([0, 0, 1, 6])
    logits = np.zeros((4, len(CLASSES)))
    logits[[0, 1, 2, 3], [0, 1, 1, 6]] = 1.0
    matrix = confusion(labels, logits)
    assert len(matrix) == len(CLASSES) and matrix[0][:2] == [1, 1] and matrix[1][1] == 1
    assert matrix[6][6] == 1 and sum(map(sum, matrix)) == 4


def test_corner_brightness_separates_vignetted_from_full_frame_images() -> None:
    from PIL import ImageDraw

    from cdm.images import corner_brightness

    full = io.BytesIO()
    Image.new("RGB", (224, 224), (200, 160, 140)).save(full, format="PNG")
    vignette = Image.new("RGB", (224, 224), (0, 0, 0))
    ImageDraw.Draw(vignette).ellipse((0, 0, 223, 223), fill=(200, 160, 140))
    framed = io.BytesIO()
    vignette.save(framed, format="PNG")
    assert corner_brightness(framed.getvalue()) < 20 < corner_brightness(full.getvalue())
