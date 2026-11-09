"""Backlog item 15: detector quality on staged shifts (results/phase3/detector_quality/spec.json).

Approved by the owner on 2026-11-09 to run on a GitHub Actions runner (dispatch only,
.github/workflows/detector-quality.yml), not on Databricks. Three steps, each resumable:

  fetch     download the clients' validation and test images from the public ISIC Archive and
            keep each one's 224 px model input as a lossless PNG
            (lesion-level splits in config/splits_v1.csv), recording each image's license,
            attribution, age and SHA-256; refuse a license outside config/sites.json, and refuse
            pool sizes other than the spec's (a wrong pool would otherwise pass unnoticed)
  score     score every image with the champion bundle on CPU: the validation baseline, the
            unshifted test pool and the pool under each staged image shift (cdm.detector_quality)
  simulate  12-month sequences from the scores; write results.json (aggregates only, staged)

Images and scores stay in the work folder (never committed). Unverified until the first run:
the ISIC search syntax (``lesion_id:<id>``), as in scripts/make_smoke_manifest.py.

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
import urllib.parse
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
SEARCH = "https://api.isic-archive.com/api/v2/images/search/?"
POOL_SIZES = {"val": 3210, "test": 3262}  # silver's labelled baseline and pool (the spec)
POOL_TOLERANCE = 0.10  # the spec's: the fetched pool may differ from silver's by at most 10%
FIELDS = ["isic_id", "lesion_id", "split", "license", "attribution", "age_approx", "sha256"]


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


def lesion_images(lesion_id: str) -> list[dict[str, Any]]:
    url: str | None = SEARCH + urllib.parse.urlencode(
        {"query": f"lesion_id:{lesion_id}", "limit": 100}
    )
    out: list[dict[str, Any]] = []
    while url:
        page = json.loads(get(url))
        out += page.get("results", [])
        url = page.get("next")
    return out


def fetch(work: Path) -> None:
    allowed = set(json.loads((REPO / "config" / "sites.json").read_text())["license_allowed"])
    with (REPO / "config" / "splits_v1.csv").open(encoding="utf-8") as f:
        lesions = [(r["lesion_id"], r["split"]) for r in csv.DictReader(f) if r["split"] != "train"]
    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = work / "manifest.csv"
    done = {r["isic_id"] for r in csv.DictReader(manifest.open())} if manifest.exists() else set()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if not done:
            writer.writeheader()
        for lesion, split in sorted(lesions):
            for image in lesion_images(lesion):
                if image["isic_id"] in done:
                    continue
                meta = image.get("metadata", {})
                clinical, acquisition = meta.get("clinical", {}), meta.get("acquisition", {})
                if acquisition.get("image_type", "dermoscopic") != "dermoscopic":
                    continue
                if clinical.get("lesion_id", lesion) != lesion:
                    raise RuntimeError(f"{image['isic_id']}: search returned another lesion")
                license_ = image.get("copyright_license", "")
                if license_ not in allowed:
                    raise RuntimeError(f"{image['isic_id']}: license {license_!r} not allowed")
                data = get(image["files"]["full"]["url"])
                # Only the model's 224 px input is kept, losslessly: full-size files would
                # not fit on a runner's disk. The original's SHA-256 is recorded.
                with Image.open(io.BytesIO(data)) as original:
                    dq.to_input(original).save(images / f"{image['isic_id']}.png")
                writer.writerow({
                    "isic_id": image["isic_id"], "lesion_id": lesion, "split": split,
                    "license": license_, "attribution": image.get("attribution", ""),
                    "age_approx": clinical.get("age_approx", ""),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })  # fmt: skip
                done.add(image["isic_id"])
    check_pool(manifest)


def read_manifest(work: Path) -> list[dict[str, str]]:
    with (work / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_pool(manifest: Path) -> dict[str, int]:
    """The fetched images are every dermoscopic image of the split's lesions, so they can differ
    a little from silver's labelled images; a large difference means a broken fetch."""
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    counts = {s: sum(r["split"] == s for r in rows) for s in POOL_SIZES}
    for split, n in counts.items():
        if abs(n - POOL_SIZES[split]) > POOL_TOLERANCE * POOL_SIZES[split]:
            raise RuntimeError(
                f"{split}: {n} images fetched, silver has {POOL_SIZES[split]}: more than "
                f"{POOL_TOLERANCE:.0%} apart, so the fetch is not the spec's pool; stop and check"
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

    def cell(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal index
        rng = dq.cell_seed(index)
        index += 1
        return [dq.run_sequence(rng, baseline, pool, **kwargs) for _ in range(repeats)]

    control = cell()
    cells.append({"shift": "none (control)", **dq.summarise(control, control=True)})
    for kind in ("blur", "brightness", "jpeg"):
        for size in spec["shifts"][kind]["sizes"]:
            shifted = z[f"test/{kind}:{float(size):g}/ood"]
            for share in spec["share_shifted"]:
                runs = cell(shifted=shifted, share=float(share))
                cells.append({"shift": kind, "size": size, "share": share, **dq.summarise(runs)})
    ages = z["test/age"]
    old = np.nan_to_num(ages, nan=-1) >= dq.OLD_AGE
    for p in spec["shifts"]["age_mix"]["sizes"]:
        runs = cell(old=old, old_share=float(p))
        cells.append({"shift": "age_mix", "size": p, "share": None, **dq.summarise(runs)})
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
        "silver_sizes": {
            "pool": POOL_SIZES["test"],
            "baseline": POOL_SIZES["val"],
            "note": "fetched from ISIC at run time: every dermoscopic image of the "
            "split's lesions, so the counts can differ from silver's labelled ones",
        },
        "pool_aged_75_or_over": int(old.sum()),
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
