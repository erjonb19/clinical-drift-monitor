"""Backlog item 40: .gitignore covers the secret and state files of SECURITY_STANDARDS §3.1."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IGNORED = [
    ".env",
    ".env.local",
    ".env.production",
    "x.pem",
    "x.key",
    "x.p12",
    "x.pfx",
    "x.tfstate",
    "x.tfstate.backup",
    ".terraform/plan",
    "secrets/a",
    "gcp-credentials.json",
    "service-account-1.json",
    ".databrickscfg",
]


def _ignored(path: str) -> bool:
    out = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO, check=False)
    return out.returncode == 0


def test_secret_and_state_files_are_ignored() -> None:
    missed = [p for p in IGNORED if not _ignored(p)]
    assert not missed, missed


def test_the_example_env_file_is_not_ignored() -> None:
    assert not _ignored(".env.example")
