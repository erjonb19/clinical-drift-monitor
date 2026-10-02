"""Federated training with Flower's FedAvg and FedProx strategies, one client per site.

The clients train one after another on the same GPU, in this process. Flower's strategy
objects do the server's work (weighted averaging of client weights by training images,
and FedProx's proximal coefficient); Flower's simulation engine (Ray) is not used, so
nothing here depends on starting a Ray cluster on serverless compute.

Fixed before any federated result:
- every round trains every client for ``local_epochs`` epochs from the global weights,
  with a fresh AdamW; the learning rate follows a cosine over rounds;
- FedProx adds (mu / 2) * ||w - w_global||^2 to each client's loss, mu = 0.01, not tuned;
- the class weights of the loss come from the pooled training class counts, a shared
  aggregate statistic, so a client missing a class still trains with the same weights;
- floating-point weights and BatchNorm statistics are averaged; integer buffers
  (BatchNorm's batch counters) are not, and keep the global model's values;
- the kept model is the round with the best pooled validation balanced accuracy, the
  same rule as centralized training's best epoch.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedProx
from torch import nn
from torch.utils.data import DataLoader

from cdm.data import LabelledImages
from cdm.train import TrainConfig, _worker_seed, balanced_accuracy, class_weights


@dataclass(frozen=True)
class FedConfig:
    method: str = "fedavg"  # or "fedprox"
    rounds: int = 20
    local_epochs: int = 1
    proximal_mu: float = 0.01


def make_strategy(cfg: FedConfig, clients: int) -> FedAvg:
    common = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 0.0,
        "min_fit_clients": clients,
        "min_available_clients": clients,
        "accept_failures": False,
    }
    if cfg.method == "fedavg":
        return FedAvg(**common)  # type: ignore[arg-type]
    if cfg.method == "fedprox":
        return FedProx(proximal_mu=cfg.proximal_mu, **common)  # type: ignore[arg-type]
    raise ValueError(f"unknown method {cfg.method!r}: use 'fedavg' or 'fedprox'")


def shared_keys(model: nn.Module) -> list[str]:
    """State-dict entries that are averaged: every floating-point tensor."""
    return [k for k, v in model.state_dict().items() if v.is_floating_point()]


def get_weights(model: nn.Module, keys: list[str]) -> list[np.ndarray]:
    state = model.state_dict()
    return [state[k].detach().cpu().float().numpy().copy() for k in keys]


def set_weights(model: nn.Module, keys: list[str], arrays: list[np.ndarray]) -> None:
    state = model.state_dict()
    for k, a in zip(keys, arrays, strict=True):
        state[k].copy_(torch.from_numpy(np.asarray(a)).to(state[k].dtype))


def round_lr(cfg: TrainConfig, fcfg: FedConfig, server_round: int) -> float:
    """Cosine over rounds: the full learning rate in round 1, near zero in the last."""
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * (server_round - 1) / fcfg.rounds))


def local_train(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    mu: float,
    device: torch.device,
) -> float:
    """Train ``model`` in place from its current (global) weights; returns the mean loss."""
    global_params = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    total, batches = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            if mu > 0:
                prox = sum(
                    ((p.float() - g.float()) ** 2).sum()
                    for p, g in zip(model.parameters(), global_params, strict=True)
                )
                loss = loss + (mu / 2.0) * cast(torch.Tensor, prox)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total, batches = total + float(loss.item()), batches + 1
    return total / max(batches, 1)


def run_federated(
    model: nn.Module,
    client_sets: Mapping[str, LabelledImages],
    val_set: LabelledImages,
    cfg: TrainConfig,
    fcfg: FedConfig,
    seed: int,
    device: torch.device,
    generator: torch.Generator,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, list[dict[str, object]]]:
    """Federated training; returns the model at its best validation round and the history."""
    names = sorted(client_sets)
    strategy = make_strategy(fcfg, len(names))
    mu = float(getattr(strategy, "proximal_mu", 0.0))
    pin = device.type == "cuda"
    loaders = {
        n: DataLoader(
            client_sets[n],
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=len(client_sets[n]) > cfg.batch_size,
            generator=generator,
            worker_init_fn=_worker_seed,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            persistent_workers=cfg.num_workers > 0,  # no worker restart every round
        )
        for n in names
    }
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    pooled_labels = [label for n in names for label in client_sets[n].labels]
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(pooled_labels).to(device), label_smoothing=cfg.label_smoothing
    )
    model.to(device)
    keys = shared_keys(model)
    global_weights = get_weights(model, keys)
    best_score, best_state = -1.0, copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = []
    for server_round in range(1, fcfg.rounds + 1):
        started = time.monotonic()
        lr = round_lr(cfg, fcfg, server_round)
        results, losses = [], {}
        for n in names:
            set_weights(model, keys, global_weights)
            losses[n] = local_train(
                model,
                loaders[n],
                criterion,
                lr,
                cfg.weight_decay,
                fcfg.local_epochs,
                mu,
                device,
            )
            fit = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(get_weights(model, keys)),
                num_examples=len(client_sets[n]),
                metrics={},
            )
            results.append((None, fit))
        aggregated, _ = strategy.aggregate_fit(server_round, results, [])  # type: ignore[arg-type]
        if aggregated is None:
            raise RuntimeError(f"round {server_round}: Flower returned no aggregate")
        global_weights = parameters_to_ndarrays(aggregated)
        set_weights(model, keys, global_weights)
        val_bacc = balanced_accuracy(model, val_loader, device)
        history.append({"round": server_round, "lr": lr, "client_loss": losses,
                        "val_bacc": val_bacc,
                        "seconds": round(time.monotonic() - started, 1)})  # fmt: skip
        log(
            f"{fcfg.method} seed {seed} round {server_round}/{fcfg.rounds}: "
            f"val bacc {val_bacc:.4f}, {time.monotonic() - started:.0f}s"
        )
        if val_bacc > best_score:
            best_score, best_state = val_bacc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history
