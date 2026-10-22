"""Linear heads on frozen embeddings (research Phase B; Phase E reuses them for fusion).

A torch linear layer trained with AdamW on mini-batches, so the seed changes both the
initialization and the batch order and three seeds give three different heads. Features are
standardized with the training set's mean and standard deviation only. The loss is
class-weighted (inverse frequency on the training set), since balanced accuracy is the
headline metric. Hyperparameters are chosen on validation by mean balanced accuracy over the
seeds; the test split is never seen here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score
from torch import nn

Array = NDArray[np.float64]

SEEDS = (0, 1, 2)
GRID = {"lr": (1e-3, 3e-3, 1e-2), "weight_decay": (0.0, 1e-4, 1e-2)}


@dataclass(frozen=True)
class ProbeConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30
    batch_size: int = 256


@dataclass
class Probe:
    """A trained head with the standardization it was trained under."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    head: nn.Linear

    @torch.no_grad()
    def logits(self, x: NDArray[np.float32]) -> Array:
        z = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
        out: Array = self.head(z).double().numpy()
        return out


def class_weights(labels: NDArray[np.int64], n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"a class has no training examples: {counts.tolist()}")
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)


def train_probe(
    x: NDArray[np.float32],
    y: NDArray[np.int64],
    n_classes: int,
    seed: int,
    cfg: ProbeConfig | None = None,
) -> Probe:
    cfg = cfg or ProbeConfig()
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-6).astype(np.float32)
    data = torch.from_numpy(((x - mean) / std).astype(np.float32))
    target = torch.from_numpy(y.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = nn.Linear(x.shape[1], n_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, n_classes))
    for _ in range(cfg.epochs):
        order = torch.randperm(len(data), generator=generator)
        for start in range(0, len(data), cfg.batch_size):
            rows = order[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss_fn(head(data[rows]), target[rows]).backward()
            optimizer.step()
    return Probe(mean, std, head.eval())


def select_config(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
    n_classes: int,
    seeds: Sequence[int] = SEEDS,
    grid: Mapping[str, Sequence[float]] = GRID,
) -> tuple[ProbeConfig, list[dict[str, Any]]]:
    """The grid point with the highest mean validation balanced accuracy over ``seeds`` (ties:
    the earlier grid point), and every grid point's scores."""
    table = []
    for lr, wd in product(grid["lr"], grid["weight_decay"]):
        cfg = ProbeConfig(lr=lr, weight_decay=wd)
        scores = [
            float(
                balanced_accuracy_score(
                    y_val, train_probe(x_train, y_train, n_classes, s, cfg).logits(x_val).argmax(1)
                )
            )
            for s in seeds
        ]
        table.append(
            {**asdict(cfg), "val_balanced_accuracy": scores, "mean": float(np.mean(scores))}
        )
    best = max(range(len(table)), key=lambda i: table[i]["mean"])
    chosen = {k: table[best][k] for k in asdict(ProbeConfig())}
    return ProbeConfig(**chosen), table
