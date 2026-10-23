"""Deploy the research API (app/api) to a free Hugging Face Space (Docker, CPU).

Run on a clean commit, after the owner has logged in (``hf auth login``). Stages exactly what
the Dockerfile needs (pyproject, src, app/api, the serving bundle) with a Space README that
carries the notice, licence and attribution, checks the bundle against its manifest, then
creates the Space if needed and uploads. The Space is public, so the model weights in the
bundle become downloadable there: non-commercial research use, with attribution.

Usage: python scripts/deploy_space.py <user>/<space-name> [--bundle dist/bundle] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Skin lesion classifier (research demo)
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
license: cc-by-nc-4.0
short_description: Research demo, not for medical use
---

# Skin lesion classifier with a drift score: research demo

**Research demo, not for medical use.** Do not use it to diagnose, screen or triage anyone.
Uploaded images are processed in memory for one request and never stored or logged; the
service keeps counts and timing only. Requests are rate limited.

The model is the shipped 3-seed EfficientNet-B0 ensemble of the Clinical Imaging Drift
Monitor, with a Mahalanobis out-of-distribution score and an abstain flag. Evaluation, limits
and source: https://github.com/erjonb19/clinical-drift-monitor

**Licence and attribution.** Non-commercial research use only. Trained on ISIC Archive images:
HAM10000 (CC BY-NC 4.0; Tschandl, Rosendahl and Kittler, Sci. Data 2018), BCN20000 from
Hospital Clinic de Barcelona (CC BY-NC) and MSK collections (CC0). Attribution: International
Skin Imaging Collaboration (ISIC) Archive.
"""


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def check_bundle(bundle: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
            sys.exit(f"{bundle / name} does not match the bundle manifest")
    return manifest


def stage(target: Path, bundle: Path) -> None:
    for rel in ("Dockerfile", ".dockerignore", "pyproject.toml"):
        shutil.copy2(REPO / rel, target / rel)
    (target / "README.md").write_text(SPACE_README, encoding="utf-8", newline="\n")
    shutil.copytree(
        REPO / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    shutil.copytree(
        REPO / "app" / "api", target / "app" / "api", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(bundle, target / "dist" / "bundle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", help="<user>/<space-name>")
    parser.add_argument("--bundle", type=Path, default=REPO / "dist" / "bundle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if git(
        "status", "--porcelain", "--", "src", "app", "Dockerfile", ".dockerignore", "pyproject.toml"
    ):
        sys.exit("uncommitted changes: deploy a commit")
    manifest = check_bundle(args.bundle)
    commit = git("rev-parse", "--short", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        stage(target, args.bundle)
        files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
        print(f"{len(files)} files, bundle {manifest['version']}, commit {commit}")
        if args.dry_run:
            print("\n".join(files))
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            folder_path=str(target), repo_id=args.space, repo_type="space",
            commit_message=f"Deploy {commit}",
        )  # fmt: skip
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
