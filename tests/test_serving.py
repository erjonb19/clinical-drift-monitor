"""cdm.serving: the NumPy scorer matches the detector, risk-coverage, and the service end to end
on a bundle built from an untrained model."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cdm.ood import MahalanobisDetector
from cdm.serving import NOTICE, Service, detector_arrays, mahalanobis, risk_coverage
from cdm.splits import CLASSES
from cdm.train import build_model, weights_hash


def test_numpy_scorer_reproduces_the_detector() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 20))
    y = np.repeat(np.arange(3), 100)
    x[y == 1] += 2
    det = MahalanobisDetector(5).fit(x, y)
    q = rng.normal(size=(40, 20)) * 3
    assert np.allclose(mahalanobis(detector_arrays(det), q), det.score(q), rtol=1e-9)


def test_risk_coverage_keeps_the_most_certain_images() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.6, 0.4]])  # 2 right, 2 wrong
    uncertainty = np.array([0.1, 0.9, 0.1, 0.9])  # the wrong ones are the uncertain ones
    full, half = risk_coverage(labels, probs, uncertainty, [1.0, 0.5])
    assert full["accuracy"] == 0.5 and full["kept"] == 4
    assert half["accuracy"] == 1.0 and half["kept"] == 2 and half["balanced_accuracy"] == 1.0


def test_service_answers_from_a_bundle_and_refuses_a_changed_file(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(pretrained=False, arch="b0").eval()
    torch.save(model.state_dict(), tmp_path / "seed0.pt")
    rng = np.random.default_rng(0)
    det = MahalanobisDetector(4).fit(rng.normal(size=(70, 1280)), np.repeat(np.arange(7), 10))
    np.savez(tmp_path / "seed0_detector.npz", **detector_arrays(det))  # type: ignore[arg-type]
    files = ("seed0.pt", "seed0_detector.npz")
    manifest = {
        "version": "test",
        "abstain_threshold": 1e9,
        "refer_threshold": 0.0638,
        "seeds": [
            {
                "checkpoint": "seed0.pt",
                "detector": "seed0_detector.npz",
                "weights_sha256": weights_hash(model),
            }
        ],  # fmt: skip
        "sha256": {f: hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in files},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    service = Service.load(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (300, 260), (180, 120, 100)).save(buf, format="JPEG")
    out = service.predict(buf.getvalue())
    assert out["label"] in CLASSES and out["abstain"] is False and out["notice"] == NOTICE
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-4
    (tmp_path / "seed0_detector.npz").write_bytes(b"changed")
    try:
        Service.load(tmp_path)
    except ValueError as err:
        assert "manifest" in str(err)
    else:
        raise AssertionError("a changed bundle file was accepted")
