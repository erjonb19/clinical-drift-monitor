"""No test file silently loses tests (docs/silent-failures.md #15)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDED = json.loads((REPO / "config" / "test_counts.json").read_text(encoding="utf-8"))["counts"]


def count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
        for n in ast.walk(tree)
    )


def test_no_test_file_lost_tests_or_disappeared() -> None:
    """Removing a test is allowed only by editing config/test_counts.json in the same change,
    where a reviewer sees it."""
    for name, recorded in RECORDED.items():
        path = REPO / "tests" / name
        assert path.exists(), f"{name} was deleted"
        assert count(path) >= recorded, f"{name}: {count(path)} tests, {recorded} recorded"
