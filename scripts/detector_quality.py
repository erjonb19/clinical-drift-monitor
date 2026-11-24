"""Backlog item 15: detector quality on staged shifts (results/phase3/detector_quality/spec.json).

Approved by the owner on 2026-11-09 to run on a GitHub Actions runner (dispatch only,
.github/workflows/detector-quality.yml), not on Databricks. Three steps:

  fetch     download, from the public ISIC Archive by ID, exactly the images silver holds for the
            clients' labelled validation and test splits (config/detector_quality_ids.csv,
            exported once by the owner from silver_images; IDs only, no data), keep each one's
            224 px model input as a lossless PNG, and record its license, attribution, age and
            SHA-256; refuse a license outside config/sites.json, and refuse any count other
            than silver's 3,210 and 3,262 (a wrong pool would otherwise pass unnoticed)
  score     score every image with the champion bundle on CPU: the validation baseline, the
            unshifted test pool and the pool under each staged image shift (cdm.detector_quality)
  simulate  12-month sequences from the scores, each with its paired control (the same months,
            no image altered); write results.json (aggregates only, staged)

Images and scores stay in the work folder (never committed). Unverified until the first run:
the ISIC endpoint's response fields, as in scripts/make_smoke_manifest.py.

Usage: python scripts/detector_quality.py {fetch,score,simulate} --work <folder> [--bundle <dir>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cdm import detector_quality as dq
from cdm.splits import CLASSES

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "results" / "phase3" / "detector_quality" / "spec.json"
OUT = REPO / "results" / "phase3" / "detector_quality" / "results.json"
IMAGE = "https://api.isic-archive.com/api/v2/images/{}/"
IDS = REPO / "config" / "detector_quality_ids.csv"
POOL_SIZES = {"val": 3210, "test": 3262}  # silver's labelled baseline and pool (the spec)
EXPORT_SQL = (
    "SELECT isic_id, split FROM workspace.cdm.silver_images WHERE role = 'client' "
    "AND split IN ('val', 'test') AND label IS NOT NULL ORDER BY split, isic_id"
)
FIELDS = ["isic_id", "split", "license", "attribution", "age_approx", "sha256"]


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data: bytes = response.read()
            return data
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def wanted_ids() -> list[tuple[str, str]]:
    """``(isic_id, split)`` for silver's labelled client validation and test images."""
    if not IDS.exists():
        sys.exit(
            f"{IDS.relative_to(REPO)} is missing: the owner exports it once from Databricks "
            f"(public ISIC IDs only) with: {EXPORT_SQL}"
        )
    with IDS.open(encoding="utf-8") as f:
        rows = [(r["isic_id"], r["split"]) for r in csv.DictReader(f)]
    counts = {s: sum(split == s for _, split in rows) for s in POOL_SIZES}
    if counts != POOL_SIZES or len({i for i, _ in rows}) != len(rows):
        sys.exit(f"{IDS.name}: counts {counts} or repeated IDs; silver's are {POOL_SIZES}")
    return rows


def fetch(work: Path) -> None:
    allowed = set(json.loads((REPO / "config" / "sites.json").read_text())["license_allowed"])
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = work / "manifest.csv"
    done = {r["isic_id"] for r in csv.DictReader(manifest.open())} if manifest.exists() else set()
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new_file:
            writer.writeheader()
        for isic_id, split in wanted_ids():
            if isic_id in done:
                continue
            image = json.loads(get(IMAGE.format(isic_id)))
            if image.get("isic_id") != isic_id:
                raise RuntimeError(f"{isic_id}: the archive returned {image.get('isic_id')}")
            license_ = image.get("copyright_license", "")
            if license_ not in allowed:
                raise RuntimeError(f"{isic_id}: license {license_!r} not allowed")
            data = get(image["files"]["full"]["url"])
            # Only the model's 224 px input is kept, losslessly: full-size files would not fit
            # on a runner's disk. The original's SHA-256 is recorded.
            with Image.open(io.BytesIO(data)) as original:
                dq.to_input(original).save(images / f"{isic_id}.png")
            clinical = image.get("metadata", {}).get("clinical", {})
            writer.writerow({
                "isic_id": isic_id, "split": split, "license": license_,
                "attribution": image.get("attribution", ""),
                "age_approx": clinical.get("age_approx", ""),
                "sha256": hashlib.sha256(data).hexdigest(),
            })  # fmt: skip
            done.add(isic_id)
    check_pool(manifest)


def read_manifest(work: Path) -> list[dict[str, str]]:
    with (work / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pool(manifest: Path) -> dict[str, int]:
    """Exactly silver's labelled images, each once: anything else is not the spec's pool."""
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    counts = {s: sum(r["split"] == s for r in rows) for s in POOL_SIZES}
    if counts != POOL_SIZES or len({r["isic_id"] for r in rows}) != len(rows):
        raise RuntimeError(
            f"fetched {counts} (or an image twice), silver has {POOL_SIZES}: not the spec's pool"
        )
    return counts


def score(work: Path, bundle: Path) -> None:
    import torch
    from torchvision import transforms

    from cdm.data import IMAGENET_MEAN, IMAGENET_STD
    from cdm.serving import Service, mahalanobis
    from cdm.train import features_and_logits

    spec = json.loads(SPEC_PATH.read_text())
    service = Service.load(bundle)
    rows = read_manifest(work)
    check_pool(work / "manifest.csv")
    tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    settings = [("none", 0.0)] + [
        (kind, float(size)) for kind in ("blur", "brightness", "jpeg")
        for size in spec["shifts"][kind]["sizes"]
    ]  # fmt: skip

    @torch.no_grad()
    def run(images: list[Image.Image]) -> dict[str, np.ndarray]:
        x = torch.stack([tensor(i) for i in images])
        probs, oods, feats = [], [], []
        for model, arrays in zip(service.models, service.detectors, strict=True):
            f, z = features_and_logits(model, x)
            probs.append(torch.softmax(z, dim=1).double().numpy())
            oods.append(mahalanobis(arrays, f.double().numpy()))
            feats.append(f.float().numpy())
        return {"ood": np.mean(oods, axis=0), "probs": np.mean(probs, axis=0),
                "features": np.concatenate(feats, axis=1)}  # fmt: skip

    out: dict[str, Any] = {}
    started = time.perf_counter()
    for split in ("val", "test"):
        ids = [r["isic_id"] for r in rows if r["split"] == split]
        per_setting: dict[str, list[dict[str, np.ndarray]]] = {}
        for start in range(0, len(ids), 32):
            chunk = ids[start : start + 32]
            base = [Image.open(work / "images" / f"{i}.png").convert("RGB") for i in chunk]
            for kind, size in settings if split == "test" else settings[:1]:
                batch = base if kind == "none" else [dq.apply_shift(b, kind, size) for b in base]
                per_setting.setdefault(f"{kind}:{size:g}", []).append(run(batch))
        for name, parts in per_setting.items():
            for key in ("ood", "probs", "features"):
                if key == "ood" or name == "none:0":
                    out[f"{split}/{name}/{key}"] = np.concatenate([p[key] for p in parts])
        out[f"{split}/ids"] = np.array(ids)
    ages = {r["isic_id"]: r["age_approx"] for r in rows}
    out["test/age"] = np.array([float(ages[i] or "nan") for i in out["test/ids"]])
    np.savez_compressed(work / "scores.npz", **out)
    (work / "score_meta.json").write_text(json.dumps({
        "bundle_version": service.version, "seconds": round(time.perf_counter() - started),
        "settings": [f"{k}:{s:g}" for k, s in settings],
        "torch": torch.__version__, "cpu": platform.processor() or platform.machine(),
    }, indent=2))  # fmt: skip


def simulate(work: Path, repeats: int | None = None) -> dict[str, Any]:
    """``repeats`` overrides the spec's 200 in tests only."""
    spec = json.loads(SPEC_PATH.read_text())
    z = np.load(work / "scores.npz")
    meta = json.loads((work / "score_meta.json").read_text())
    baseline, pool = z["val/none:0/ood"], z["test/none:0/ood"]
    repeats = repeats or int(str(spec["repeats"]).split()[0])
    cells: list[dict[str, Any]] = []
    index = 0

    def cell(**kwargs: Any) -> dict[str, Any]:
        """The cell's sequences and, with the same seeds, their paired controls."""
        nonlocal index
        seeds = [dq.sequence_seed(index, r) for r in range(repeats)]
        index += 1
        runs = [dq.run_sequence(s, baseline, pool, **kwargs) for s in seeds]
        paired = [dq.run_sequence(s, baseline, pool) for s in seeds]
        return dq.summarise(runs, paired=paired)

    seeds = [dq.sequence_seed(index, r) for r in range(repeats)]
    index += 1
    control = [dq.run_sequence(s, baseline, pool) for s in seeds]
    cells.append({"shift": "none (control)", **dq.summarise(control, control=True)})
    for kind in ("blur", "brightness", "jpeg"):
        for size in spec["shifts"][kind]["sizes"]:
            shifted = z[f"test/{kind}:{float(size):g}/ood"]
            for share in spec["share_shifted"]:
                summary = cell(shifted=shifted, share=float(share))
                cells.append({"shift": kind, "size": size, "share": share, **summary})
    ages = z["test/age"]
    known = np.isfinite(ages)
    old = np.nan_to_num(ages, nan=-1) >= dq.OLD_AGE
    for p in spec["shifts"]["age_mix"]["sizes"]:
        summary = cell(old=old, old_share=float(p), known=known)
        cells.append({"shift": "age_mix", "size": p, "share": None, **summary})
    first = {k: z[f"val/none:0/{k}"] for k in ("ood", "features")}
    base_set = {**first, "predicted": z["val/none:0/probs"].argmax(1),
                "top_prob": z["val/none:0/probs"].max(1)}  # fmt: skip
    pool_set = {"ood": pool, "features": z["test/none:0/features"],
                "predicted": z["test/none:0/probs"].argmax(1),
                "top_prob": z["test/none:0/probs"].max(1)}  # fmt: skip
    post = dq.post_replay_false_alarms(
        dq.cell_seed(index), base_set, pool_set, len(CLASSES), months=repeats * dq.MONTHS
    )
    means = {
        k.split("/")[1]: float(np.mean(z[k]))
        for k in z.files
        if k.startswith("test/") and k.endswith("/ood") and k != "test/none:0/ood"
    }
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    return {
        "staged": "every shift is staged (synthetic): real in-distribution test images altered "
        "in a known way, or resampled by age. Not the replay, not live drift",
        "spec": "results/phase3/detector_quality/spec.json",
        "spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "code_version": commit,
        "ran_on": {
            "where": "GitHub Actions runner (workflow detector-quality.yml)",
            "python": platform.python_version(),
            **meta,
        },  # fmt: skip
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool_images": int(len(pool)),
        "baseline_images": int(len(baseline)),
        "pool_source": "silver's labelled client images, by ID (config/detector_quality_ids.csv), "
        "fetched from the ISIC Archive at run time; the baseline is rescored on the runner",
        "pool_aged_75_or_over": int(old.sum()),
        "pool_age_unknown": int((~known).sum()),
        "mean_ood": {"baseline": float(np.mean(baseline)), **means},
        "lower_than_baseline_mean": dq.check_direction(baseline, means),
        "cells": cells,
        "post_replay_rule_false_alarms": post,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fetch", "score", "simulate"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.step == "fetch":
        fetch(args.work)
    elif args.step == "score":
        if args.bundle is None:
            sys.exit("score needs --bundle (the champion serving bundle)")
        score(args.work, args.bundle)
    else:
        result = simulate(args.work)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
