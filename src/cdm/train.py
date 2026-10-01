"""EfficientNet-B0 classifier for HAM10000: class-weighted fine-tuning, best epoch on validation."""

from __future__ import annotations

import copy
import hashlib
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

from cdm.data import CLASSES, LabelledImages


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    pretrained: bool = True
    # Fixed so results do not depend on how many cores a machine happens to have.
    num_threads: int = 8
    # Phase 2 tuning options; every default reproduces the Phase 0 and stage 2 setting.
    arch: str = "b0"
    image_size: int = 224
    rotate: bool = False
    color_jitter: bool = False
    warmup_epochs: int = 0
    label_smoothing: float = 0.0


def build_model(pretrained: bool = True, arch: str = "b0") -> nn.Module:
    """ImageNet EfficientNet (B0, or B3 for tuning rung 3) with a new 7-class head."""
    if arch == "b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = cast(nn.Module, efficientnet_b0(weights=weights))
    elif arch == "b3":
        weights_b3 = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = cast(nn.Module, efficientnet_b3(weights=weights_b3))
    else:
        raise ValueError(f"unknown arch {arch!r}: use 'b0' or 'b3'")
    classifier = cast(nn.Sequential, model.classifier)
    head = classifier[1]
    assert isinstance(head, nn.Linear)
    classifier[1] = nn.Linear(head.in_features, len(CLASSES))
    return model


def features_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 1280-d pooled features the classifier head sees, and the head's logits."""
    backbone, pool, head = (
        cast(nn.Module, getattr(model, n)) for n in ("features", "avgpool", "classifier")
    )
    feats = torch.flatten(pool(backbone(x)), 1)
    return feats, head(feats)


def weights_hash(model: nn.Module) -> str:
    """SHA-256 of the weight values, in state-dict order.

    Use this, not the checkpoint file's MD5, to ask "same model?": ``torch.save`` names the
    archive's inner folder after the file, so identical weights saved under two names get
    two different file hashes.
    """
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def seed_everything(seed: int, num_threads: int, gpu: bool = False) -> torch.Generator:
    """Make everything after this call repeatable: call it before building the model.

    Seeds every random number generator, fixes the CPU thread count and turns on PyTorch's
    deterministic algorithms. Returns the generator that orders training batches; DataLoader
    workers derive their own seeds from it (``_worker_seed``).

    On GPU the deterministic algorithms run in warn-only mode: EfficientNet's adaptive
    average pooling has no deterministic CUDA backward, and strict mode would stop training.
    Whether GPU runs repeat exactly is measured, not assumed (results/phase2).
    """
    # Required by deterministic cuBLAS on GPU; harmless on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=gpu)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds libraries that use the legacy global state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.Generator().manual_seed(seed)


def class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse class frequency, scaled so a balanced dataset gets weight 1 everywhere."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"training split is missing a class: counts {counts.tolist()}")
    return torch.tensor(len(labels) / (len(CLASSES) * counts), dtype=torch.float32)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


@torch.no_grad()
def balanced_accuracy(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> float:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    return float(balanced_accuracy_score(torch.cat(labels), torch.cat(preds)))


def lr_schedule(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay over all steps; with ``warmup_epochs``, a linear warmup first and cosine
    over the remaining steps."""
    total = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch
    if warmup == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total - warmup),
        ],
        milestones=[warmup],
    )


def train(
    model: nn.Module,
    train_set: LabelledImages,
    val_set: LabelledImages,
    cfg: TrainConfig,
    seed: int,
    device: torch.device,
    generator: torch.Generator,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Fine-tune ``model`` and return it at the epoch with the best validation balanced accuracy."""
    loader_args: dict[str, object] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=len(train_set) > cfg.batch_size,
        generator=generator,
        worker_init_fn=_worker_seed,
        **loader_args,  # type: ignore[arg-type]
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)  # type: ignore[arg-type]

    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_set.labels).to(device), label_smoothing=cfg.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_schedule(optimizer, cfg, len(train_loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, history = -1.0, copy.deepcopy(model.state_dict()), []
    for epoch in range(1, cfg.epochs + 1):
        started = time.monotonic()
        model.train()
        total, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total, batches = total + loss.item(), batches + 1
        val_bacc = balanced_accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / batches, "val_bacc": val_bacc})
        log(
            f"seed {seed} epoch {epoch}/{cfg.epochs}: loss {total / batches:.4f}, "
            f"val bacc {val_bacc:.4f}, {time.monotonic() - started:.0f}s"
        )
        if val_bacc > best_score:
            best_score, best_state = val_bacc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history
