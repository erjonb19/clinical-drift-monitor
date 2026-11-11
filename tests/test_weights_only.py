"""Every torch.load passes weights_only=True (security audit 2026-11-10, SEC.md §4.5, item 31).

Before torch 2.6, torch.load unpickles by default, which can run code from a swapped checkpoint.
The floor is torch>=2.6, and this test keeps every call explicit so a lower torch is safe too.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("src", "scripts", "tests", "app", "databricks/notebooks")
CALL = re.compile(r"\btorch\.load\(")


def _is_torch_load(func: ast.expr, aliases: set[str]) -> bool:
    if isinstance(func, ast.Attribute) and func.attr == "load":
        return isinstance(func.value, ast.Name) and func.value.id in aliases
    return False


def _weights_only_true(call: ast.Call) -> bool:
    return any(
        kw.arg == "weights_only" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def unsafe_calls(source: str, name: str) -> list[str]:
    """Return "name:line" for each torch.load call without a literal weights_only=True."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not valid Python: flag every torch.load( line that does not say weights_only=True.
        return [
            f"{name}:{i}"
            for i, line in enumerate(source.splitlines(), 1)
            if CALL.search(line) and "weights_only=True" not in line
        ]
    aliases = {"torch"}
    loads: set[str] = set()  # names bound by "from torch import load [as x]"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names if a.name == "torch" and a.asname}
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            loads |= {a.asname or a.name for a in node.names if a.name == "load"}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        direct = isinstance(node.func, ast.Name) and node.func.id in loads
        if (direct or _is_torch_load(node.func, aliases)) and not _weights_only_true(node):
            found.append(f"{name}:{node.lineno}")
    return found


def _python_files() -> list[Path]:
    files: list[Path] = []
    for folder in SCANNED:
        files += sorted((REPO / folder).rglob("*.py"))
    return files


def test_every_torch_load_passes_weights_only_true() -> None:
    files = _python_files()
    assert any("databricks/notebooks" in p.as_posix() for p in files), "notebooks not scanned"
    bad: list[str] = []
    for path in files:
        bad += unsafe_calls(path.read_text(encoding="utf-8"), str(path.relative_to(REPO)))
    assert not bad, "torch.load without weights_only=True at: " + ", ".join(bad)


def test_scanner_flags_planted_bad_calls() -> None:
    planted = (
        "# Databricks notebook source\n"
        "import torch\n"
        "import torch as T\n"
        "from torch import load as tl\n"
        'ok = torch.load("a.pt", map_location="cpu", weights_only=True)\n'
        'bad = torch.load("a.pt", map_location="cpu")\n'
        'bad2 = T.load("a.pt", weights_only=False)\n'
        'bad3 = tl("a.pt")\n'
        'bad4 = torch.load(dbutils.widgets.get("checkpoint"), **kw)\n'
    )
    assert unsafe_calls(planted, "nb.py") == ["nb.py:6", "nb.py:7", "nb.py:8", "nb.py:9"]
    # A file that does not parse falls back to a line scan.
    assert unsafe_calls('x = torch.load("a.pt")\n%run ./other\n', "nb.py") == ["nb.py:1"]
