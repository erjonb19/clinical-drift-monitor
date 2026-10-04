"""Out-of-distribution scores. Convention everywhere: a higher score means more likely OOD.

Every detector scores in-distribution and OOD inputs with the same function, from the same
fine-tuned model, using statistics fit on training features only. The ECE 570 notebook
broke each of these rules once; docs/phase0-note.md explains how that produced an AUROC
of 4.1%.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

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


@torch.no_grad()
def logits_with_flips(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[Array, NDArray[np.int64]]:
    """Logits averaged over the original and its horizontal, vertical and double flips."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        logits.append(
            torch.stack([features_and_logits(model, v)[1].float() for v in views]).mean(0).cpu()
        )
        labels.append(y)
    return torch.cat(logits).double().numpy(), torch.cat(labels).long().numpy()


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


# ---------------------------------------- Gram matrices (Sastry & Oore, 2020)

GRAM_POWERS = tuple(range(1, 11))  # p = 1..10, as in the paper


def stage_maps(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Pooled features, logits and every stage output of EfficientNet's ``features``."""
    maps = []
    h = x
    for stage in cast(nn.Sequential, model.features):
        h = stage(h)
        maps.append(h)
    feats = torch.flatten(cast(nn.Module, model.avgpool)(h), 1)
    logits = cast(nn.Module, model.classifier)(feats)
    return feats, logits, maps


def gram_features(maps: Sequence[torch.Tensor], powers: Sequence[int] = GRAM_POWERS) -> list[Array]:
    """Per layer, (batch, len(powers) * channels): the row sums of each p-th order Gram matrix
    G_p = F^p (F^p)^T, then sign(G) |G|^(1/p). Float64: F^10 overflows float32."""
    out = []
    for m in maps:
        f = m.detach().double().flatten(2)  # batch, channels, positions
        per_power = []
        for p in powers:
            fp = f.pow(p)
            g = torch.bmm(fp, fp.transpose(1, 2)).sum(dim=2)
            per_power.append(g.sign() * g.abs().pow(1.0 / p))
        out.append(torch.cat(per_power, dim=1).cpu().numpy())
    return out


class GramDetector:
    """Gram-matrix deviations (Sastry & Oore, 2020).

    Fit: per predicted class, the minimum and maximum of every Gram feature over training
    images (streamed, batch by batch). Score: an image's deviation from its predicted class's
    range, summed per layer, each layer divided by its mean deviation on validation, then
    summed over layers. No labels are used at scoring, so in-distribution and OOD images go
    through the same computation. A class no training image was predicted as falls back to
    the range over all classes.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.mins: list[Array] = []
        self.maxs: list[Array] = []
        self.seen = np.zeros(n_classes, dtype=bool)
        self.norm: Array | None = None

    def update(self, layers: Sequence[Array], preds: NDArray[np.int64]) -> None:
        if not self.mins:
            self.mins = [np.full((self.n_classes, a.shape[1]), np.inf) for a in layers]
            self.maxs = [np.full((self.n_classes, a.shape[1]), -np.inf) for a in layers]
        for c in np.unique(preds):
            rows = preds == c
            self.seen[c] = True
            for i, a in enumerate(layers):
                self.mins[i][c] = np.minimum(self.mins[i][c], a[rows].min(axis=0))
                self.maxs[i][c] = np.maximum(self.maxs[i][c], a[rows].max(axis=0))

    def _bounds(self) -> tuple[list[Array], list[Array]]:
        mins, maxs = [m.copy() for m in self.mins], [m.copy() for m in self.maxs]
        for lo, hi in zip(mins, maxs, strict=True):
            lo[~self.seen] = lo[self.seen].min(axis=0)
            hi[~self.seen] = hi[self.seen].max(axis=0)
        return mins, maxs

    def deviations(self, layers: Sequence[Array], preds: NDArray[np.int64]) -> Array:
        """(images, layers): summed relative deviation outside the predicted class's range."""
        if not self.mins:
            raise RuntimeError("update the detector with training images before scoring")
        mins, maxs = self._bounds()
        devs = []
        for a, lo, hi in zip(layers, mins, maxs, strict=True):
            lo_c, hi_c = lo[preds], hi[preds]
            below = np.maximum(lo_c - a, 0) / (np.abs(lo_c) + 1e-6)
            above = np.maximum(a - hi_c, 0) / (np.abs(hi_c) + 1e-6)
            devs.append((below + above).sum(axis=1))
        return np.stack(devs, axis=1)

    def fit_normalizer(self, val_deviations: Array) -> GramDetector:
        self.norm = val_deviations.mean(axis=0) + 1e-12
        return self

    def score(self, deviations: Array) -> Array:
        if self.norm is None:
            raise RuntimeError("fit the normalizer on validation deviations before scoring")
        return np.asarray((deviations / self.norm).sum(axis=1), dtype=np.float64)


# ---------------------------------------------------------------- Mahalanobis PCA size

PCA_CANDIDATES = (16, 32, 64, 128, 256, 512)


def pca_size_for_variance(train_feats: Array, threshold: float = 0.95) -> int:
    """Method D: the smallest number of components explaining ``threshold`` of the training
    features' variance. Uses training features only."""
    pca = PCA(random_state=0).fit(train_feats)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def pca_size_by_selection(
    train_feats: Array,
    train_labels: NDArray[np.int64],
    val_feats: Array,
    selection: Mapping[str, Array],
    candidates: Sequence[int] = PCA_CANDIDATES,
) -> tuple[int, dict[int, dict[str, float]]]:
    """Method C (sensitivity only): the candidate size with the highest mean AUROC separating
    validation images from the selection draws of the benchmark sets."""
    from cdm.eval import auroc

    table: dict[int, dict[str, float]] = {}
    for k in candidates:
        det = MahalanobisDetector(k).fit(train_feats, train_labels)
        val = det.score(val_feats)
        table[k] = {name: auroc(val, det.score(f)) for name, f in selection.items()}
        table[k]["mean"] = float(np.mean([table[k][n] for n in selection]))
    best = max(candidates, key=lambda k: (table[k]["mean"], -k))
    return best, table
