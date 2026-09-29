"""Static guards for pipelines/lakehouse.py, which CI cannot run (no Spark).

Each guard pins a mistake that failed, or would fail, a real Databricks run. They inspect the
parsed code, not the text, so explanations in docstrings do not trip them.
"""

from __future__ import annotations

import ast
from pathlib import Path

TREE = ast.parse(
    (Path(__file__).resolve().parents[1] / "pipelines" / "lakehouse.py").read_text(encoding="utf-8")
)


def _method_calls(name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def _is_lit_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lit"
    )


def test_no_group_by_on_a_literal() -> None:
    """Spark reads an integer literal in groupBy as a column position: groupBy(F.lit(1))
    failed analysis on the first run (run 330838862474565)."""
    offending = [c.lineno for c in _method_calls("groupBy") if any(_is_lit_call(a) for a in c.args)]
    assert offending == []


def test_no_eager_actions_in_dataset_definitions() -> None:
    """Dataset functions run while Databricks plans an update; actions there would run
    before their inputs are refreshed."""
    for action in ("toPandas", "collect", "count"):
        assert _method_calls(action) == [], action
