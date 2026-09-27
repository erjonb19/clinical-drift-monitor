"""Out-of-distribution scores. Convention everywhere: a higher score means more likely OOD.

Every detector scores in-distribution and OOD inputs with the same function, from the same
fine-tuned model, using statistics fit on training features only. The ECE 570 notebook
broke each of these rules once; docs/phase0-note.md explains how that produced an AUROC
of 4.1%.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader

from cdm.train import features_and_logits

Array = NDArray[np.float64]


@torch.no_grad()
def extract(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[Array, Array, NDArray[np.int64]]:
    """Penultimate features, logits and labels for every item in ``loader``, in order."""
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        f, z = features_and_logits(model, x.to(device, non_blocking=True))
        feats.append(f.float().cpu())
        logits.append(z.float().cpu())
        labels.append(y)
    return (
        torch.cat(feats).double().numpy(),
        torch.cat(logits).double().numpy(),
        torch.cat(labels).long().numpy(),
    )


def msp_score(logits: Array) -> Array:
    """Negative maximum softmax probability (Hendrycks & Gimpel, 2017)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return np.asarray(-probs.max(axis=1), dtype=np.float64)


def energy_score(logits: Array, temperature: float = 1.0) -> Array:
    """Free energy -T * logsumexp(logits / T) (Liu et al., 2020). Lower for in-distribution."""
    scaled = logits / temperature
    top = scaled.max(axis=1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(scaled - top).sum(axis=1))
    return np.asarray(-temperature * lse, dtype=np.float64)


class MahalanobisDetector:
    """Distance to the nearest class mean under one shared covariance (Lee et al., 2018).

    PCA, class means and covariance are fit once, on training features. ``score`` takes no
    labels, so an image is always scored against its nearest class, never its true one,
    and in-distribution and OOD inputs go through exactly the same computation.
    """

    def __init__(self, n_components: int = 256) -> None:
        self.n_components = n_components
        self.pca: PCA | None = None
        self.means: Array | None = None
        self.precision: Array | None = None

    def fit(self, train_feats: Array, train_labels: NDArray[np.int64]) -> MahalanobisDetector:
        k = min(self.n_components, train_feats.shape[1], train_feats.shape[0] - 1)
        self.pca = PCA(n_components=k, random_state=0).fit(train_feats)
        z = self.pca.transform(train_feats)
        classes = np.unique(train_labels)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(f"expected labels 0..K-1, got {classes}")
        self.means = np.stack([z[train_labels == c].mean(axis=0) for c in classes])
        centered = z - self.means[train_labels]
        cov = centered.T @ centered / len(z)
        self.precision = np.linalg.pinv(cov, hermitian=True)
        return self

    def score(self, feats: Array) -> Array:
        if self.pca is None or self.means is None or self.precision is None:
            raise RuntimeError("fit the detector on training features before scoring")
        z = self.pca.transform(feats)
        dists = np.stack(
            [np.einsum("ij,jk,ik->i", z - mu, self.precision, z - mu) for mu in self.means],
            axis=1,
        )
        return np.asarray(dists.min(axis=1), dtype=np.float64)
