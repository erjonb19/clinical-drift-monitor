"""Download the champion serving bundle (backlog items 6 and 21) into dist/bundle.

Reads ``bundle_asset`` (URL and SHA-256) from config/champion_smoke.json, which is filled once
the owner has uploaded the bundle as a release asset (docs/OWNER_STEPS.md, S1). The archive must
come from this repository's GitHub releases and match the SHA-256, or nothing is written. It is
unpacked with tarfile's "data" filter (no links, devices or paths outside the folder), must hold
exactly one manifest.json, and is then checked by cdm.serving.Service.load's own hashes when the
API starts. An existing dist/bundle is never overwritten: move it away first.

Usage: python scripts/fetch_champion_bundle.py [--dest dist/bundle]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASES = "https://github.com/erjonb19/clinical-drift-monitor/releases/download/"
MAX_BYTES = 2 * 1024**3


def asset(config: Path) -> tuple[str, str]:
    found = json.loads(config.read_text(encoding="utf-8"))["bundle_asset"]
    url, sha = found.get("url"), found.get("sha256")
    if not url or not sha:
        sys.exit("bundle_asset is not filled in yet: see docs/OWNER_STEPS.md, S1")
    if not str(url).startswith(RELEASES) or len(str(sha)) != 64:
        sys.exit("bundle_asset must be this repository's release asset with a full SHA-256")
    return str(url), str(sha).lower()


def download(url: str, target: Path) -> str:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=600) as response, target.open("wb") as f:  # noqa: S310
        total = 0
        while chunk := response.read(1 << 20):
            total += len(chunk)
            if total > MAX_BYTES:
                sys.exit("the bundle archive is larger than 2 GB")
            digest.update(chunk)
            f.write(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=REPO / "dist" / "bundle")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "champion_smoke.json")
    args = parser.parse_args(argv)
    if args.dest.exists():
        sys.exit(f"{args.dest} exists: move it away first (it is never overwritten)")
    url, sha = asset(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        got = download(url, archive)
        if got != sha:
            sys.exit(f"SHA-256 {got} differs from config/champion_smoke.json ({sha})")
        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        roots = [p.parent for p in unpacked.rglob("manifest.json")]
        if len(roots) != 1:
            sys.exit(f"expected exactly one manifest.json in the archive, found {len(roots)}")
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(roots[0], args.dest)
    print(f"champion bundle ({sha[:12]}) unpacked to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
