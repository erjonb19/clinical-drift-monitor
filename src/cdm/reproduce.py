"""Phase 0 in one command: ``python -m cdm.reproduce``.

Trains EfficientNet-B0 on the fixed lesion-level HAM10000 split once per seed, reports
test-split classification metrics, and scores three OOD detectors on PathMNIST (near) and
CIFAR-10 (far). Every number in the README comes from the files this writes.

Each finished seed is saved at once. A restart with the same code, data and settings skips
seeds already saved, so a crash loses at most the seed in progress.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cdm.data import (
    CLASSES,
    HamDataset,
    OODImages,
    cifar10_ood,
    data_root,
    eval_transform,
    lesion_split,
    load_ham,
    md5sum,
    pathmnist_ood,
    prepare_ham,
    split_fingerprint,
    train_transform,
)
from cdm.eval import classification_metrics, detection_metrics, summarize
from cdm.ood import MahalanobisDetector, energy_score, extract, msp_score
from cdm.train import TrainConfig, build_model, train

SPLIT_SEED = 0  # the lesion split is fixed; only the training seed varies
OOD_SUBSET_SEED = 0
DETECTORS = ("mahalanobis", "msp", "energy")
OOD_LABELS = {"pathmnist": "PathMNIST (near)", "cifar10": "CIFAR-10 (far)"}

ImageSet = Dataset[tuple[torch.Tensor, int]]
# Results are JSON-shaped: they are written to disk and read back by people, not code.
Result = dict[str, Any]


class ResumeMismatchError(RuntimeError):
    """A saved seed came from different code, data or settings than this run."""


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs; released when it exits."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def write_json(path: Path, data: object) -> None:
    """Write via a temporary file so a crash never leaves half a result on disk."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_key(provenance: Result) -> Result:
    """What must match for a saved seed to be reused: code, data, split and settings."""
    keys = ("code_version", "split_fingerprint", "metadata_md5", "images", "n_ood", "config")
    return {k: provenance[k] for k in keys}


def load_finished(path: Path, key: Result) -> Result | None:
    if not path.exists():
        return None
    saved: Result = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("run_key") != key:
        raise ResumeMismatchError(
            f"{path} was produced by a different run (code, data or settings changed). "
            "Move it aside or pass a different --out; results from two setups are never mixed."
        )
    return saved


def run_seed(
    seed: int,
    sets: Mapping[str, HamDataset],
    ood_sets: Mapping[str, OODImages],
    cfg: TrainConfig,
    device: torch.device,
) -> Result:
    """Train one model, then score the test split and every OOD set with that same model."""
    model = build_model(pretrained=cfg.pretrained)
    model, history = train(model, sets["train"], sets["val"], cfg, seed, device, log=log)

    def features(ds: ImageSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # OOD sets are in-memory arrays; worker processes would each get a pickled copy.
        workers = cfg.num_workers if isinstance(ds, HamDataset) else 0
        loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=workers)
        return extract(model, loader, device)

    # Detector statistics come from training images with the eval transform, never test.
    train_feats, _, train_labels = features(sets["train_eval"])
    test_feats, test_logits, test_labels = features(sets["test"])
    maha = MahalanobisDetector().fit(train_feats, train_labels)

    def scores(feats: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mahalanobis": maha.score(feats),
            "msp": msp_score(logits),
            "energy": energy_score(logits),
        }

    log(f"seed {seed}: scoring test split and OOD sets")
    id_scores = scores(test_feats, test_logits)
    ood: Result = {}
    for name, ds in ood_sets.items():
        feats, logits, _ = features(ds)
        ood_scores = scores(feats, logits)
        ood[name] = {d: detection_metrics(id_scores[d], ood_scores[d]) for d in DETECTORS}
        ood[name]["n"] = {"id": len(test_labels), "ood": len(ds)}

    best = max(history, key=lambda h: h["val_bacc"])
    return {
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "history": history,
        "test": classification_metrics(test_labels, test_logits),
        "ood": ood,
    }


def check_plausible(result: Result) -> list[str]:
    """An AUROC below 0.5 means the score points the wrong way: a bug, not a finding."""
    problems = []
    for name, per_detector in result["ood"].items():
        for detector in DETECTORS:
            value = per_detector[detector]["auroc"]
            if value < 0.5:
                problems.append(f"seed {result['seed']}: {detector} on {name} AUROC {value:.3f}")
    return problems


def aggregate(results: list[Result]) -> Result:
    tests = [r["test"] for r in results]
    oods = [r["ood"] for r in results]
    return {
        "seeds": [r["seed"] for r in results],
        "balanced_accuracy": summarize([t["balanced_accuracy"] for t in tests]),
        "recall": {c: summarize([t["recall"][c] for t in tests]) for c in CLASSES},
        "support": tests[0]["support"],
        "ood": {
            name: {
                d: {m: summarize([o[name][d][m] for o in oods]) for m in ("auroc", "fpr95")}
                for d in DETECTORS
            }
            for name in oods[0]
        },
    }


def _pct(s: Mapping[str, float]) -> str:
    return f"{100 * s['mean']:.1f} ({100 * s['min']:.1f}–{100 * s['max']:.1f})"


def results_markdown(summary: Result, provenance: Result) -> str:
    ood, recall, support = summary["ood"], summary["recall"], summary["support"]
    names = list(ood)
    lines = [
        "# Phase 0 results",
        "",
        f"Generated by `python -m cdm.reproduce` from code at `{provenance['code_version']}` "
        f"on {provenance['device']}. Seeds {summary['seeds']}; every cell is the mean with "
        "the min–max range over seeds, in percent. The lesion split is fixed "
        f"(fingerprint `{provenance['split_fingerprint']}`); only the training seed varies.",
        "",
        "## Classification, HAM10000 test split",
        "",
        f"Balanced accuracy: **{_pct(summary['balanced_accuracy'])}**",
        "",
        "| Class | Test images | Recall |",
        "| --- | ---: | ---: |",
        *(f"| {c} | {support[c]} | {_pct(recall[c])} |" for c in CLASSES),
        "",
        "## Out-of-distribution detection",
        "",
        "In-distribution is the HAM10000 test split. AUROC: higher is better. FPR@95TPR: the "
        "share of OOD images accepted when 95% of in-distribution images are kept; lower is "
        "better.",
        "",
        "| Detector | "
        + " | ".join(f"{OOD_LABELS.get(n, n)} AUROC | FPR@95TPR" for n in names)
        + " |",
        "| --- |" + " ---: | ---: |" * len(names),
        *(
            f"| {d} | "
            + " | ".join(f"{_pct(ood[n][d]['auroc'])} | {_pct(ood[n][d]['fpr95'])}" for n in names)
            + " |"
            for d in DETECTORS
        ),
        "",
    ]
    return "\n".join(lines)


# The files that decide what a run computes. Commits elsewhere (docs, committed results) do
# not change the code version, so they never block resuming a run.
CODE_PATHS = ("src", "pyproject.toml")


def _code_version() -> str:
    """Short hash of the last commit that touched the code, with -dirty for local edits."""
    try:
        sha = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", *CODE_PATHS],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *CODE_PATHS], capture_output=True, text=True
        ).stdout
        return (sha or "unknown") + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _limit(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(frame) <= n:
        return frame
    # Keep at least two images of every class so metrics and class weights stay defined.
    per_class = frame.sample(frac=1.0, random_state=seed).groupby("dx").head(2)
    rest = frame.drop(per_class.index).sample(max(n - len(per_class), 0), random_state=seed)
    return pd.concat([per_class, rest])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_root())
    parser.add_argument("--out", type=Path, default=Path("results/phase0"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--n-ood", type=int, default=2000, help="images per OOD set")
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke test: cap images per HAM split"
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.out == Path("results/phase0"):
        args.out = Path("results/smoke")  # never overwrite real results with a smoke run

    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers)

    ham = prepare_ham(args.data)
    meta = load_ham(ham)
    split = lesion_split(meta, seed=SPLIT_SEED)
    frames = {s: _limit(meta[split == s], args.limit, SPLIT_SEED) for s in ("train", "val", "test")}
    sets = {
        "train": HamDataset(frames["train"], train_transform()),
        "train_eval": HamDataset(frames["train"], eval_transform()),
        "val": HamDataset(frames["val"], eval_transform()),
        "test": HamDataset(frames["test"], eval_transform()),
    }
    ood_sets = {
        "pathmnist": pathmnist_ood(args.data, args.n_ood, OOD_SUBSET_SEED),
        "cifar10": cifar10_ood(args.data, args.n_ood, OOD_SUBSET_SEED),
    }

    provenance: Result = {
        "code_version": _code_version(),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else platform.processor() or "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "split_fingerprint": split_fingerprint(meta, split),
        "metadata_md5": md5sum(ham / "HAM10000_metadata.csv"),
        "images": {s: len(f) for s, f in frames.items()},
        "n_ood": {n: len(ds) for n, ds in ood_sets.items()},
        "config": {**vars(cfg), "limit": args.limit},
    }
    commit = provenance["code_version"]
    if args.limit is None and (commit == "unknown" or commit.endswith("-dirty")):
        log(f"refusing a full run from uncommitted code (code {commit})")
        return 1
    log("run settings:\n" + json.dumps(provenance, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    key = run_key(provenance)
    results: list[Result] = []
    problems: list[str] = []
    for seed in args.seeds:
        path = args.out / f"seed{seed}.json"
        # Smoke runs always start fresh; only full runs resume.
        result = load_finished(path, key) if args.limit is None else None
        if result is not None:
            log(f"seed {seed}: already finished, loaded {path}")
        else:
            log(f"seed {seed}: training ({cfg.epochs} epochs)")
            result = {"run_key": key, **run_seed(seed, sets, ood_sets, cfg, device)}
            result["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
            write_json(path, result)
            log(f"seed {seed}: finished, saved {path}")
        results.append(result)
        problems += check_plausible(result)

        summary = aggregate(results)
        write_json(args.out / "summary.json", {"provenance": provenance, **summary})
        (args.out / "results.md").write_text(
            results_markdown(summary, provenance), encoding="utf-8"
        )

    log("results:\n" + (args.out / "results.md").read_text(encoding="utf-8"))
    if problems:
        log("IMPLAUSIBLE RESULTS, investigate before reporting:\n  " + "\n  ".join(problems))
        return 2
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
