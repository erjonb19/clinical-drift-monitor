"""Phase 3 serving: the shipped ensemble on CPU with a Mahalanobis OOD score and an abstain flag.

The serving bundle (built by ``databricks/notebooks/serving_bundle.py`` under
``results/phase3/serving/spec.json``) holds, per seed, the checkpoint and the detector's arrays,
plus the abstain threshold (the 95th percentile of the validation OOD score) and the Phase 2
melanoma referral threshold. Scoring here is NumPy only, so the API needs no scikit-learn at
request time, and the bundle builder checks that it reproduces ``MahalanobisDetector.score``.

Research prototype, not for medical use. An uploaded image is held in memory for one request
and never written or logged.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from cdm.data import eval_transform
from cdm.ood import MahalanobisDetector
from cdm.splits import CLASSES
from cdm.train import build_model, features_and_logits, weights_hash

Array = NDArray[np.float64]
NOTICE = "Research demo, not for medical use."
# A decompression bomb under the 10 MB upload cap could need gigabytes; 50 megapixels is ample.
Image.MAX_IMAGE_PIXELS = 50_000_000


def detector_arrays(detector: MahalanobisDetector) -> dict[str, Array]:
    """The fitted detector as plain arrays (PCA mean and components, class means, precision)."""
    if detector.pca is None or detector.means is None or detector.precision is None:
        raise RuntimeError("fit the detector first")
    return {
        "pca_mean": np.asarray(detector.pca.mean_, dtype=np.float64),
        "pca_components": np.asarray(detector.pca.components_, dtype=np.float64),
        "class_means": np.asarray(detector.means, dtype=np.float64),
        "precision": np.asarray(detector.precision, dtype=np.float64),
    }


def mahalanobis(arrays: dict[str, Array], feats: Array) -> Array:
    """Distance to the nearest class mean, as ``MahalanobisDetector.score`` computes it."""
    z = (np.asarray(feats, dtype=np.float64) - arrays["pca_mean"]) @ arrays["pca_components"].T
    dists = np.stack(
        [
            np.einsum("ij,jk,ik->i", z - mu, arrays["precision"], z - mu)
            for mu in arrays["class_means"]
        ],
        axis=1,
    )
    return np.asarray(dists.min(axis=1), dtype=np.float64)


def balanced_accuracy(labels: NDArray[np.int64], predicted: NDArray[np.int64]) -> float | None:
    """Mean recall over the classes present; None for an empty set."""
    present = np.unique(labels)
    if len(present) == 0:
        return None
    return float(np.mean([np.mean(predicted[labels == c] == c) for c in present]))


def risk_coverage(
    labels: NDArray[np.int64],
    probs: Array,
    abstain_first: Array,
    grid: Sequence[float],
) -> list[dict[str, Any]]:
    """Accuracy and balanced accuracy on the images kept when the highest ``abstain_first``
    values are set aside, at each coverage in ``grid``."""
    order = np.argsort(-np.asarray(abstain_first), kind="stable")  # first to abstain first
    predicted = probs.argmax(axis=1)
    out = []
    for coverage in grid:
        n_keep = int(round(coverage * len(labels)))
        keep = np.sort(order[len(labels) - n_keep :]) if n_keep else np.array([], dtype=int)
        kept_labels, kept_pred = labels[keep], predicted[keep]
        out.append(
            {
                "coverage": float(coverage),
                "kept": int(n_keep),
                "accuracy": float(np.mean(kept_labels == kept_pred)) if n_keep else None,
                "balanced_accuracy": balanced_accuracy(kept_labels, kept_pred),
            }
        )
    return out


@dataclass
class Service:
    """The loaded bundle: three networks, their detectors and the two thresholds."""

    models: list[torch.nn.Module]
    detectors: list[dict[str, Array]]
    abstain_threshold: float
    refer_threshold: float
    version: str

    @classmethod
    def load(cls, folder: Path) -> Service:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{name} does not match the bundle manifest")
        models, detectors = [], []
        for seed in manifest["seeds"]:
            model = build_model(pretrained=False, arch="b0")
            model.load_state_dict(torch.load(folder / seed["checkpoint"], map_location="cpu"))
            if weights_hash(model) != seed["weights_sha256"]:
                raise ValueError(f"{seed['checkpoint']}: weights hash mismatch")
            models.append(model.eval())
            with np.load(folder / seed["detector"]) as z:
                detectors.append({k: z[k] for k in z.files})
        return cls(models, detectors, float(manifest["abstain_threshold"]),
                   float(manifest["refer_threshold"]), str(manifest["version"]))  # fmt: skip

    @torch.no_grad()
    def predict(self, image: bytes) -> dict[str, Any]:
        """One image in, the ensemble's answer out. The bytes are decoded in memory only."""
        with Image.open(io.BytesIO(image)) as img:
            x = eval_transform(224)(img.convert("RGB")).unsqueeze(0)
        probs, scores = [], []
        for model, arrays in zip(self.models, self.detectors, strict=True):
            feats, logits = features_and_logits(model, x)
            probs.append(torch.softmax(logits, dim=1).double().numpy()[0])
            scores.append(float(mahalanobis(arrays, feats.double().numpy())[0]))
        p = np.mean(probs, axis=0)
        ood = float(np.mean(scores))
        return {
            "label": CLASSES[int(p.argmax())],
            "probabilities": {c: round(float(v), 6) for c, v in zip(CLASSES, p, strict=True)},
            "ood_score": round(ood, 4),
            "abstain": ood > self.abstain_threshold,
            "refer_melanoma": bool(p[CLASSES.index("mel")] >= self.refer_threshold),
            "model_version": self.version,
            "notice": NOTICE,
        }
