"""cdm.fusion: the metadata encoding is fit on train only, and each variant gets the right input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdm.fusion import VARIANTS, MetadataEncoder, check_order, inputs


def meta(ages: list[float | None], sexes: list[object], sites: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {"age_approx": ages, "gender_concept_id": sexes, "anatomic_site_concept_id": sites}
    )


def test_encoder_is_fit_on_train_and_sends_unseen_values_to_unknown() -> None:
    enc = MetadataEncoder().fit(meta([40.0, 60.0], [8507, 8532], [100, 0]))
    assert enc.columns == ["age", "age_missing", "sex:8507", "sex:8532", "sex:unknown",
                           "site:100", "site:unknown"]  # fmt: skip
    x = enc.transform(meta([50.0, None], [8532, None], [999, 100]))
    assert x[0, 0] == 0.0 and x[0, 1] == 0.0  # the training mean, not missing
    assert x[1, 0] == 0.0 and x[1, 1] == 1.0  # missing age
    assert x[0, 3] == 1 and x[1, 4] == 1  # 8532, then missing sex is unknown
    assert x[0, 6] == 1 and x[1, 5] == 1  # unseen site 999 is unknown; 100 is known
    assert np.all(x[:, 2:5].sum(1) == 1) and np.all(x[:, 5:].sum(1) == 1)


def test_transform_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        MetadataEncoder().transform(meta([1.0], [1], [1]))


def test_each_variant_gets_its_input() -> None:
    images = {
        "biomedclip": np.ones((2, 3), np.float32),
        "imagenet_effb0": np.zeros((2, 4), np.float32),
    }
    m = np.full((2, 5), 7, np.float32)
    shapes = {v: inputs(v, images, m).shape for v in VARIANTS}
    assert shapes == {
        "metadata_only": (2, 5),
        "image_only__biomedclip": (2, 3),
        "image_only__imagenet_effb0": (2, 4),
        "fusion__biomedclip": (2, 8),
        "fusion__imagenet_effb0": (2, 9),
    }
    with pytest.raises(ValueError, match="metadata rows"):
        inputs("fusion__biomedclip", images, m[:1])


def test_order_check() -> None:
    check_order(["a", "b"], ["a", "b"], "x")
    with pytest.raises(ValueError, match="order"):
        check_order(["a", "b"], ["b", "a"], "x")
