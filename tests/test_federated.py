"""Federated training: Flower's averaging, FedProx's pull towards the global weights, and
the round loop's best-round selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cdm.federated import (
    FedConfig,
    get_weights,
    local_train,
    make_strategy,
    round_lr,
    run_federated,
    set_weights,
    shared_keys,
)
from cdm.train import TrainConfig, balanced_accuracy


class Points(Dataset[tuple[torch.Tensor, int]]):
    """Separable 2-D points standing in for images; exposes labels like LabelledImages."""

    def __init__(self, n: int, seed: int, shift: float = 0.0) -> None:
        rng = np.random.default_rng(seed)
        self.labels = [int(v) for v in rng.integers(0, 7, n)]
        centres = np.stack([np.cos(np.arange(7)), np.sin(np.arange(7))], axis=1) * 3
        self.x = centres[self.labels] + rng.normal(0, 0.5, (n, 2)) + shift
        self.groups = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.x[i], dtype=torch.float32), self.labels[i]


def tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 7))


def test_flower_fedavg_is_the_image_weighted_mean() -> None:
    a, b = [np.ones(3, dtype=np.float32)], [np.full(3, 4.0, dtype=np.float32)]
    fits = [
        (None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(w), n, {}))
        for w, n in ((a, 100), (b, 50))
    ]
    out, _ = make_strategy(FedConfig("fedavg"), 2).aggregate_fit(1, fits, [])  # type: ignore[arg-type]
    assert out is not None
    np.testing.assert_allclose(parameters_to_ndarrays(out)[0], 2.0)  # (100*1 + 50*4) / 150


def test_strategy_choice_and_mu() -> None:
    assert make_strategy(FedConfig("fedprox", proximal_mu=0.01), 4).proximal_mu == 0.01  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        make_strategy(FedConfig("fedsgd"), 4)


def test_integer_buffers_are_not_averaged() -> None:
    model = tiny()
    keys = shared_keys(model)
    assert not any(k.endswith("num_batches_tracked") for k in keys)
    arrays = [np.zeros_like(a) for a in get_weights(model, keys)]
    set_weights(model, keys, arrays)
    assert all(float(v.abs().sum()) == 0 for k, v in model.state_dict().items() if k in keys)


def test_proximal_term_keeps_clients_near_the_global_weights() -> None:
    data = DataLoader(Points(256, seed=0, shift=2.0), batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    drift = {}
    for mu in (0.0, 10.0):
        torch.manual_seed(0)
        model = tiny()
        start = [p.detach().clone() for p in model.parameters()]
        local_train(model, data, criterion, 0.05, 0.0, 3, mu, torch.device("cpu"))
        drift[mu] = sum(
            float(((p - s) ** 2).sum()) for p, s in zip(model.parameters(), start, strict=True)
        )
    assert drift[10.0] < 0.5 * drift[0.0]


def test_round_lr_is_a_cosine_over_rounds() -> None:
    cfg, fcfg = TrainConfig(lr=1e-3), FedConfig(rounds=20)
    assert round_lr(cfg, fcfg, 1) == pytest.approx(1e-3)
    assert round_lr(cfg, fcfg, 11) == pytest.approx(5e-4)
    assert round_lr(cfg, fcfg, 20) < 1e-5


@pytest.mark.parametrize("method", ["fedavg", "fedprox"])
def test_federated_rounds_learn_and_keep_the_best_round(method: str) -> None:
    torch.manual_seed(0)
    clients = {f"c{i}": Points(200, seed=i, shift=0.3 * i) for i in range(3)}
    val = Points(300, seed=9)
    cfg = TrainConfig(batch_size=32, num_workers=0, lr=0.02)
    model, history = run_federated(
        tiny(),
        clients,  # type: ignore[arg-type]
        val,  # type: ignore[arg-type]
        cfg,
        FedConfig(method, rounds=5),
        seed=0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        log=lambda _: None,
    )
    assert [h["round"] for h in history] == [1, 2, 3, 4, 5]
    best = max(float(h["val_bacc"]) for h in history)  # type: ignore[arg-type]
    assert best > 0.6
    kept = balanced_accuracy(model, DataLoader(val, batch_size=64), torch.device("cpu"))
    assert kept == pytest.approx(best)
