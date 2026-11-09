"""Backlog item 10: the README's reviewer feedback table keeps its fixed format."""

from __future__ import annotations

import re
from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
HEADER = "| Date | Reviewer role | Comment | What changed | Link |"


def _section() -> str:
    start = README.index("## Reviewer feedback")
    end = README.index("\n## ", start + 1)
    return README[start:end]


def test_the_feedback_table_has_the_fixed_columns() -> None:
    assert HEADER in _section()


def test_every_entry_has_all_five_fields_and_a_real_date() -> None:
    rows = [line for line in _section().splitlines() if line.startswith("|")][2:]
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 5 and all(cells), row
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]), row
    if not rows:
        assert "No entries yet." in _section()
