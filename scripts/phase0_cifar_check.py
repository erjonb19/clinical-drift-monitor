"""Phase 0 follow-up: why does Mahalanobis do worse on CIFAR-10 than on PathMNIST?

Extracts features once (ImageNet EfficientNet-B0, or a fine-tuned checkpoint), saves them
under the data root, then reports for both OOD sets:
- AUROC and FPR@95TPR for Mahalanobis at several PCA sizes (fit on HAM10000 train only);
- which lesion class each image is nearest to, and how often;
- the median Mahalanobis score of each set, as a direction check.

Usage: python scripts/phase0_cifar_check.py [--checkpoint PATH] [--name NAME]
Writes results/phase0/checks/<name>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdm.data import (
    CLASSES,
    HamDataset,
    cifar10_ood,
    data_root,
    eval_transform,
    lesion_split,
    load_ham,
    pathmnist_ood,
    prepare_ham,
    split_fingerprint,
)
from cdm.eval import detection_metrics
from cdm.ood import MahalanobisDetector, extract
from cdm.reproduce import OOD_SUBSET_SEED, SPLIT_SEED, _code_version
from cdm.train import build_model

PCA_SIZES = (32, 64, 128, 256, 512, 1280)
DEFAULT_PCA = 256


def nearest_class(det: MahalanobisDetector, feats: np.ndarray) -> np.ndarray:
    assert det.pca is not None and det.means is not None and det.precision is not None
    z = det.pca.transform(feats)
    dists = np.stack(
        [np.einsum("ij,jk,ik->i", z - mu, det.precision, z - mu) for mu in det.means], axis=1
    )
    return np.asarray(dists.argmin(axis=1))


def class_shares(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels, minlength=len(CLASSES))
    return {c: round(float(n) / len(labels), 4) for c, n in zip(CLASSES, counts, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--name", default="pretrained")
    parser.add_argument("--n-ood", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = data_root()
    cache = root / "features" / f"{args.name}.npz"
    if cache.exists():
        saved = dict(np.load(cache))
        print(f"loaded features from {cache}")
    else:
        model = build_model(pretrained=True)
        if args.checkpoint is not None:
            model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        ham = prepare_ham(root)
        meta = load_ham(ham)
        split = lesion_split(meta, seed=SPLIT_SEED)
        sets = {
            "train": HamDataset(meta[split == "train"], eval_transform()),
            "test": HamDataset(meta[split == "test"], eval_transform()),
            "pathmnist": pathmnist_ood(root, args.n_ood, OOD_SUBSET_SEED),
            "cifar10": cifar10_ood(root, args.n_ood, OOD_SUBSET_SEED),
        }
        saved = {"split_fingerprint": np.array(split_fingerprint(meta, split))}
        for name, ds in sets.items():
            workers = args.workers if isinstance(ds, HamDataset) else 0
            loader = DataLoader(ds, batch_size=64, num_workers=workers)
            feats, _, labels = extract(model, loader, device)
            saved[f"{name}_feats"], saved[f"{name}_labels"] = feats, labels
            print(f"extracted {name}: {feats.shape}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **saved)

    train_x, train_y = saved["train_feats"], saved["train_labels"]
    test_x = saved["test_feats"]
    report: dict[str, object] = {
        "features": args.name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "code_version": _code_version(),
        "split_fingerprint": str(saved["split_fingerprint"]),
        "n": {
            k: int(saved[f"{k}_feats"].shape[0]) for k in ("train", "test", "pathmnist", "cifar10")
        },
        "by_pca_size": {},
    }
    by_pca: dict[str, object] = {}
    for k in PCA_SIZES:
        det = MahalanobisDetector(n_components=k).fit(train_x, train_y)
        id_scores = det.score(test_x)
        row: dict[str, object] = {}
        for ood in ("pathmnist", "cifar10"):
            ood_scores = det.score(saved[f"{ood}_feats"])
            row[ood] = {
                **{m: round(v, 4) for m, v in detection_metrics(id_scores, ood_scores).items()},
                "median_score_id": round(float(np.median(id_scores)), 2),
                "median_score_ood": round(float(np.median(ood_scores)), 2),
            }
        by_pca[str(k)] = row
        if k == DEFAULT_PCA:
            report["nearest_class_at_pca_256"] = {
                name: class_shares(nearest_class(det, saved[f"{name}_feats"]))
                for name in ("test", "pathmnist", "cifar10")
            }
    report["by_pca_size"] = by_pca

    out = Path("results/phase0/checks") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
