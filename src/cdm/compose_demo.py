"""The one-command demo's client (backlog item 21): ``docker compose up`` runs it beside the API.

It fetches two fixed batches of public ISIC images, checks each file's SHA-256 against its
manifest, sends every image to the running API's ``/predict``, and runs the pre-registered
drift test (``cdm.monitor.drift_test``: KS p < 0.01 and PSI >= 0.2) on the API's OOD scores:

- the **reference** batch: the champion smoke test's 50 images of the clients' test split
  (``config/champion_smoke_images.csv``), in distribution;
- the **shifted** batch: 50 PAD-UFES-20 smartphone photos
  (``config/compose_demo_shift_images.csv``), a hospital and a camera the model never saw.
  Real images, not staged drift.

This compares two small scored batches, not a month against the monitoring job's validation
baseline (the bundle does not carry that baseline), and says so in its output. The thresholds
were set for 300 images a month against that baseline; on 50 against 50 the result is a
demonstration, not a measurement. Images are kept
only in a cache folder for the run; nothing is sent anywhere but the API.

Usage: python -m cdm.compose_demo --api http://api:7860 [--out result.json]
The result is printed; ``--out`` also writes it to a file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cdm.monitor import drift_test

# ISIC's file host, under both names its manifests use (the regional one in files written
# 2026-11-14).
ISIC_FILES = (
    "https://isic-archive.s3.amazonaws.com/images/",
    "https://isic-archive.s3.us-east-1.amazonaws.com/images/",
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
COMPARISON = (
    "shifted batch against the reference batch, both scored by the API now; not the monitoring "
    "job's comparison against the validation baseline"
)

Fetch = Callable[[str], bytes]
Post = Callable[[bytes], tuple[int, dict[str, Any]]]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no images")
    for r in rows:  # only ISIC's own file host, a plain ISIC ID, a full SHA-256
        if (
            not r["url"].startswith(ISIC_FILES)
            or not re.fullmatch(r"ISIC_\d+", r["isic_id"])
            or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
        ):
            raise ValueError(f"{path}: {r['isic_id']} has an unexpected URL, ID or hash")
    return rows


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - ISIC https only
        data: bytes = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{url}: larger than {MAX_IMAGE_BYTES} bytes")
    return data


def image_bytes(row: dict[str, str], cache: Path, fetch: Fetch) -> bytes:
    """The image, from the cache or ISIC, refused unless its SHA-256 matches the manifest."""
    path = cache / f"{row['isic_id']}.img"
    cached = path.exists()
    data = path.read_bytes() if cached else fetch(row["url"])
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"{row['isic_id']}: SHA-256 differs from the manifest")
    if not cached:
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def http_poster(api: str) -> Post:
    def post(data: bytes) -> tuple[int, dict[str, Any]]:
        kind = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        request = urllib.request.Request(  # noqa: S310 - the compose service's own address
            f"{api}/predict", data=data, headers={"Content-Type": kind}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return response.status, json.load(response)
        except urllib.error.HTTPError as err:
            return err.code, {}

    return post


def score(
    rows: list[dict[str, str]],
    cache: Path,
    fetch: Fetch,
    post: Post,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 20,
) -> list[dict[str, Any]]:
    """The API's answer for every image, waiting and retrying on its rate limit (429)."""
    answers = []
    for row in rows:
        data = image_bytes(row, cache, fetch)
        for _ in range(retries):
            status, body = post(data)
            if status != 429:
                break
            sleep(6.0)
        if status != 200:
            raise RuntimeError(f"{row['isic_id']}: the API answered {status}")
        answers.append(body)
    return answers


def summary(reference: list[dict[str, Any]], shifted: list[dict[str, Any]]) -> dict[str, Any]:
    ref = np.array([a["ood_score"] for a in reference], dtype=np.float64)
    new = np.array([a["ood_score"] for a in shifted], dtype=np.float64)
    versions = {a["model_version"] for a in reference + shifted}

    def share(answers: list[dict[str, Any]]) -> float:
        return float(np.mean([a["abstain"] for a in answers]))

    return {
        "what": "backlog item 21: the docker compose demo's drift check",
        "comparison": COMPARISON,
        "model_version": sorted(versions),
        "reference": {"images": len(ref), "abstain_share": share(reference)},
        "shifted": {"images": len(new), "abstain_share": share(shifted)},
        "drift": drift_test(ref, new),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://api:7860")
    parser.add_argument("--reference", type=Path, default=Path("config/champion_smoke_images.csv"))
    parser.add_argument(
        "--shifted", type=Path, default=Path("config/compose_demo_shift_images.csv")
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/cdm-compose-demo"))  # noqa: S108
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    post = http_poster(args.api.rstrip("/"))
    reference = score(read_manifest(args.reference), args.cache, http_get, post)
    shifted = score(read_manifest(args.shifted), args.cache, http_get, post)
    result = summary(reference, shifted)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    drift = result["drift"]
    print("DRIFT ALARM" if drift["alert"] else "no drift alarm",
          f"(KS p = {drift['ks_p']:.3g}, PSI = {drift['psi']:.2f})")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
