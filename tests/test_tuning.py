"""Phase 2 tuning options: each defaults to the stage 2 setting and does what it claims."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cdm.data import RandomRotation, inscribed_size, train_transform
from cdm.eval import classification_metrics, macro_auroc
from cdm.images import decode_resize
from cdm.report import score_rows, support
from cdm.splits import CLASSES
from cdm.train import TrainConfig, build_model, lr_schedule


def test_rotation_never_leaves_fill_corners() -> None:
    torch.manual_seed(0)
    img = Image.new("RGB", (300, 200), (200, 150, 120))
    for _ in range(20):
        out = RandomRotation(15.0)(img)
        grey = np.asarray(out.convert("L"))
        assert grey.min() > 150, "a rotated crop exposed fill colour"


def test_inscribed_size_is_unchanged_without_rotation_and_shrinks_with_it() -> None:
    assert inscribed_size(300, 200, 0) == (300, 200)
    w, h = inscribed_size(300, 200, 15)
    assert w < 300 and h < 200


@pytest.mark.parametrize("rotate,jitter", [(False, False), (True, True)])
def test_train_transform_outputs_the_configured_size(rotate: bool, jitter: bool) -> None:
    img = Image.new("RGB", (512, 384), (120, 80, 60))
    assert train_transform(300, rotate, jitter)(img).shape == (3, 300, 300)


def test_default_train_transform_is_the_stage_2_one() -> None:
    names = [type(t).__name__ for t in train_transform().transforms]
    assert names == ["RandomResizedCrop", "RandomHorizontalFlip", "RandomVerticalFlip",
                     "ToTensor", "Normalize"]  # fmt: skip


def _lrs(cfg: TrainConfig, steps: int) -> list[float]:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = lr_schedule(opt, cfg, steps)
    out = []
    for _ in range(cfg.epochs * steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_schedule_defaults_to_plain_cosine_and_warms_up_when_asked() -> None:
    plain = _lrs(TrainConfig(epochs=4), 10)
    assert plain[0] == pytest.approx(1.0) and plain[-1] < 0.01
    warm = _lrs(TrainConfig(epochs=4, warmup_epochs=1), 10)
    assert warm[0] == pytest.approx(0.01) and max(warm) == pytest.approx(1.0, rel=0.02)
    assert warm.index(max(warm)) >= 9


def test_macro_auroc_and_melanoma_sensitivity() -> None:
    labels = np.array([0, 4, 4, 5, 5])
    logits = np.zeros((5, len(CLASSES)))
    logits[np.arange(5), labels] = 5.0
    assert macro_auroc(labels, logits) == pytest.approx(1.0)
    m = classification_metrics(labels, logits)
    assert m["melanoma_sensitivity"] == pytest.approx(1.0)
    no_mel = classification_metrics(np.array([0, 5]), logits[[0, 3]])
    assert no_mel["melanoma_sensitivity"] is None


def test_stored_images_keep_their_field_of_view_as_jpeg() -> None:
    src = io.BytesIO()
    Image.new("RGB", (600, 450), (90, 60, 40)).save(src, format="JPEG", quality=90)
    d = decode_resize(src.getvalue(), size=384, center_crop=False, fmt="JPEG", quality=95)
    with Image.open(io.BytesIO(d.image)) as img:
        assert img.format == "JPEG" and img.size == (512, 384)


def test_b3_gets_a_seven_class_head() -> None:
    model = build_model(pretrained=False, arch="b3").eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 300, 300)).shape == (1, len(CLASSES))
    with pytest.raises(ValueError, match="unknown arch"):
        build_model(pretrained=False, arch="b9")


def test_report_splits_barcelona_by_border_and_flags_small_test_classes() -> None:
    rows = []
    for i in range(12):
        rows.append({"client": "barcelona", "split": "val", "label": CLASSES[i % 2],
                     "lesion_id": f"L{i // 2}", "bordered": i < 4})  # fmt: skip
    for i in range(6):
        rows.append({"client": "ham_vienna", "split": "test", "label": "nv",
                     "lesion_id": f"H{i}", "bordered": False})  # fmt: skip
    frame = pd.DataFrame(rows)
    labels = np.asarray(frame["label"].map({c: i for i, c in enumerate(CLASSES)}), dtype=np.int64)
    logits = np.zeros((len(frame), len(CLASSES)))
    logits[np.arange(len(frame)), labels] = 1.0
    report = score_rows(frame, logits, labels)
    assert report["val"]["barcelona_bordered"]["images"] == 4
    assert report["val"]["barcelona_not_bordered"]["images"] == 8
    assert "barcelona_bordered" not in report["test"]
    sup = support(frame)
    assert sup["by_split_client_class"]["val/barcelona"]["akiec"] == {"images": 6, "lesions": 6}
    assert "nv" in sup["test_classes_under_20_images"]["pooled"]
