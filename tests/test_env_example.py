"""Backlog item 41: .env.example names every environment variable the code reads, with no
value filled in."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
READ = re.compile(
    r"""os\.(?:environ(?:\.get)?\(\s*|environ\[|getenv\(\s*)["']([A-Z][A-Z0-9_]*)["']"""
)
# Read through a tuple of names rather than a literal call (scripts/watchdog.py).
EXTRA = {"DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"}
# Set by a notebook for its own libraries, not read by this code.
SET_ONLY = {"HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}


def _example() -> dict[str, str]:
    lines = (REPO / ".env.example").read_text("utf-8").splitlines()
    pairs = [line.split("=", 1) for line in lines if line.strip() and not line.startswith("#")]
    return {name.strip(): value.strip() for name, value in pairs}


def test_every_variable_the_code_reads_is_listed() -> None:
    read: set[str] = set(EXTRA)
    for folder in ("src", "scripts", "app"):
        for path in (REPO / folder).rglob("*.py"):
            read |= set(READ.findall(path.read_text("utf-8")))
    assert read - SET_ONLY <= set(_example()), sorted(read - SET_ONLY - set(_example()))


def test_no_value_is_filled_in() -> None:
    filled = {name: value for name, value in _example().items() if value}
    assert not filled, sorted(filled)
