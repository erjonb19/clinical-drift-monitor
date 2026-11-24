"""docs/BACKLOG.md keeps one entry per item: item numbers never change, and a merge that resolves
a conflict by keeping both sides can leave an item twice, once ticked and once not (2026-11-11,
caught in review). Each checkbox item number appears exactly once, and they run 1 to N."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_every_backlog_item_appears_once() -> None:
    text = (REPO / "docs" / "BACKLOG.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^(\d+)\. \[[ x]\]", text, flags=re.MULTILINE)]
    twice = sorted(n for n, k in Counter(numbers).items() if k > 1)
    assert not twice, f"items listed more than once: {twice}"
    assert sorted(numbers) == list(range(1, max(numbers) + 1)), "an item number is missing"
