"""Committed results files as MLflow runs, so every reported number is in the repo and in MLflow.

A results file (JSON) is logged once it is committed: its numeric leaves become metrics named
by their path, the file itself becomes an artifact, and its SHA-256 and commit become tags. The
index ``results/mlflow_runs.json`` maps each logged file to its run, and a test checks every
indexed file still has the hash that was logged, so a result edited after logging is caught.

Files written before this rule (Phase 0 to 2) are logged after the fact and tagged so; from the
research extension on, notebooks log their own runs and record them in the same index.
No MLflow import here: CI tests this module without MLflow installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

INDEX = Path("results/mlflow_runs.json")
MAX_KEY = 250  # MLflow's limit on metric names
_BAD = re.compile(r"[^A-Za-z0-9_\-. /]")


def metric_key(path: list[str]) -> str:
    """A path of JSON keys as an MLflow metric name: ``/``-joined, disallowed characters
    replaced by ``_``. Raises if the name would exceed MLflow's limit rather than truncating
    it, since two truncated names could collide."""
    key = "/".join(_BAD.sub("_", part) for part in path)
    if len(key) > MAX_KEY:
        raise ValueError(f"metric name longer than {MAX_KEY}: {key[:80]}...")
    return key


def numeric_leaves(data: Any, prefix: list[str] | None = None) -> dict[str, float]:
    """Every finite int or float in a JSON value, keyed by its path. Booleans, strings, nulls
    and non-finite numbers are skipped; list items are keyed by position."""
    path = prefix or []
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            out.update(numeric_leaves(v, [*path, str(k)]))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            out.update(numeric_leaves(v, [*path, str(i)]))
    elif isinstance(data, int | float) and not isinstance(data, bool) and math.isfinite(data):
        out[metric_key(path)] = float(data)
    return out


def sha256(path: Path) -> str:
    """SHA-256 of a file with line endings normalized to LF, as git stores it, so the hash is
    the same on a Windows checkout and in CI."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_index(repo: Path) -> dict[str, dict[str, Any]]:
    path = repo / INDEX
    if not path.exists():
        return {}
    index: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return index


def stale_entries(repo: Path) -> list[str]:
    """Indexed files that are missing or whose content changed after they were logged."""
    return [
        name
        for name, entry in load_index(repo).items()
        if not (repo / name).exists() or sha256(repo / name) != entry["file_sha256"]
    ]
