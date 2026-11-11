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


VARIABLE = re.compile(r"^(#\s)?([A-Z][A-Z0-9_]*)=(.*)$")


def _example(commented: bool = True) -> dict[str, str]:
    """Names and values in .env.example; ``commented`` includes the optional variables that
    are shown commented out with their defaults."""
    out = {}
    for line in (REPO / ".env.example").read_text("utf-8").splitlines():
        match = VARIABLE.match(line)
        if match and (commented or not match.group(1)):
            out[match.group(2)] = match.group(3).strip()
    return out


def test_every_variable_the_code_reads_is_listed() -> None:
    read: set[str] = set(EXTRA)
    for folder in ("src", "scripts", "app"):
        for path in (REPO / folder).rglob("*.py"):
            read |= set(READ.findall(path.read_text("utf-8")))
    assert read - SET_ONLY <= set(_example()), sorted(read - SET_ONLY - set(_example()))


def test_no_value_is_filled_in() -> None:
    """Live lines carry names only; the optional ones are commented out, so an env file built
    from this one cannot override a default with an empty value (review of PR #91)."""
    live = _example(commented=False)
    assert set(live) == {
        "ANTHROPIC_API_KEY",
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    }
    assert not any(live.values()), sorted(k for k, v in live.items() if v)
