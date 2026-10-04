"""Build the benchmark OOD sets for the drift table from the local Phase 0 copies.

Evaluation sets are Phase 0's exact subsets: 2,000 images each from PathMNIST's test split
(native 224 px) and CIFAR-10's test split, chosen by ``cdm.data.fixed_subset`` with seed 0.
Selection sets, used only for the Mahalanobis PCA sensitivity row (method C), never overlap
them: 1,000 images from PathMNIST's validation split and 1,000 from CIFAR-10's training
split, seed 1. Images are stored losslessly as PNG with a SHA-256 per image, in one zip
with an ``index.csv``, plus a manifest with the zip's SHA-256. Only the manifest is committed.

PathMNIST's npz is 12.6 GB and deflate-compressed, so each split is streamed row by row
and only the chosen rows are kept.

Usage: python scripts/build_benchmarks.py --data C:/Users/Erjon/data/cdm --out dist/benchmarks
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib import format as npy_format
from PIL import Image

from cdm.data import fixed_subset

EVAL_N, EVAL_SEED = 2000, 0  # Phase 0's subsets (cdm.reproduce: n_ood 2000, seed 0)
SELECT_N, SELECT_SEED = 1000, 1
LICENSES = {
    "pathmnist": "CC BY 4.0 (MedMNIST v2; source NCT-CRC-HE-100K)",
    "cifar10": "CIFAR-10 (Krizhevsky, 2009); no explicit licence, used for research",
}


def npz_rows(npz: Path, member: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Stream one array of an npz and keep only the rows in ``wanted``."""
    with zipfile.ZipFile(npz) as z, z.open(member) as f:
        version = npy_format.read_magic(f)
        read_header = (
            npy_format.read_array_header_1_0
            if version == (1, 0)
            else npy_format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(f)
        assert not fortran and dtype == np.uint8, (shape, fortran, dtype)
        row = int(np.prod(shape[1:]))
        out = {}
        for i in range(shape[0]):
            buf = f.read(row)
            if i in wanted:
                out[i] = np.frombuffer(buf, dtype=np.uint8).reshape(shape[1:]).copy()
        return out


def cifar_rows(folder: Path, files: list[str], wanted: set[int]) -> dict[int, np.ndarray]:
    out, offset = {}, 0
    for name in files:
        with (folder / name).open("rb") as f:
            batch = pickle.load(f, encoding="bytes")  # noqa: S301 - the official CIFAR files
        data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        for i in range(len(data)):
            if offset + i in wanted:
                out[offset + i] = data[i].copy()
        offset += len(data)
    return out


def png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def rows(dataset: str, split: str, use: str, images: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    out = []
    for i in sorted(images):
        data = png(images[i])
        out.append({"dataset": dataset, "split": split, "use": use, "index": i, "png": data,
                    "sha256": hashlib.sha256(data).hexdigest()})  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("dist/benchmarks"))
    args = parser.parse_args()
    npz = args.data / "medmnist" / "pathmnist_224.npz"
    cifar = args.data / "cifar10" / "cifar-10-batches-py"
    plan = [
        ("pathmnist", "test", "eval", 7180, EVAL_N, EVAL_SEED),
        ("pathmnist", "val", "select", 10004, SELECT_N, SELECT_SEED),
        ("cifar10", "test", "eval", 10000, EVAL_N, EVAL_SEED),
        ("cifar10", "train", "select", 50000, SELECT_N, SELECT_SEED),
    ]
    records: list[dict[str, Any]] = []
    for dataset, split, use, total, n, seed in plan:
        wanted = set(fixed_subset(total, n, seed))
        if dataset == "pathmnist":
            images = npz_rows(npz, f"{split}_images.npy", wanted)
        else:
            files = ["test_batch"] if split == "test" else [f"data_batch_{i}" for i in range(1, 6)]
            images = cifar_rows(cifar, files, wanted)
        assert len(images) == len(wanted), (dataset, split, len(images))
        records += rows(dataset, split, use, images)
        print(dataset, split, use, len(images), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "benchmarks.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for r in records:
            z.writestr(f"{r['dataset']}/{r['split']}/{r['index']:05d}.png", r["png"])
        index = pd.DataFrame(records).drop(columns="png")
        index["path"] = [f"{r['dataset']}/{r['split']}/{r['index']:05d}.png" for r in records]
        z.writestr("index.csv", index.to_csv(index=False, lineterminator="\n"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.DataFrame(records)
    manifest = {
        "file": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "sets": [{"dataset": d, "split": s, "use": u, "seed": sd, "images": int(
            ((frame["dataset"] == d) & (frame["split"] == s)).sum()),
            "indices_sha256": hashlib.sha256(",".join(map(str, sorted(
                frame.loc[(frame["dataset"] == d) & (frame["split"] == s), "index"]))).encode()
            ).hexdigest()[:16]} for d, s, u, _, _, sd in plan],
        "licenses": LICENSES,
        "eval_overlaps_select": False,
    }  # fmt: skip
    Path("config/benchmarks_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
