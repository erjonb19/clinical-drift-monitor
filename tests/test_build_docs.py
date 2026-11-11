"""Backlog item 9: the model card, change-control plan and standards alignment are what
scripts/build_docs.py generates from committed evidence (nobody edits them by hand), and the
change-control plan fits on one page."""

from __future__ import annotations

import importlib.util
import re
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# One page: a US Letter or A4 page at 10.5 pt with 1 inch margins holds about 550 words, or 48
# lines of about 100 characters (blank lines included).
ONE_PAGE_WORDS, ONE_PAGE_LINES, WIDTH = 550, 48, 100


def rendered_lines(text: str) -> int:
    """Lines as printed: each paragraph, bullet, heading or quote is joined and wrapped at
    WIDTH characters (a source line, however long, is not one printed line); blank lines count."""
    blocks: list[list[str] | None] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            blocks.append(None)
            continue
        if current and re.match(r"(#|- |\d+\. |> )", line):
            blocks.append(current)
            current = []
        current.append(line.strip())
    if current:
        blocks.append(current)
    return sum(1 if b is None else max(1, len(textwrap.wrap(" ".join(b), WIDTH))) for b in blocks)


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)  # the script reads committed evidence by relative path
        spec = importlib.util.spec_from_file_location(
            "build_docs", REPO / "scripts" / "build_docs.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            "model-card.md": module.model_card(),
            "change-control-plan.md": module.change_control(),
            "standards-alignment.md": module.standards(),
        }


@pytest.mark.parametrize(
    "name", ["model-card.md", "change-control-plan.md", "standards-alignment.md"]
)
def test_committed_docs_match_what_the_evidence_generates(
    generated: dict[str, str], name: str
) -> None:
    committed = (REPO / "docs" / name).read_text(encoding="utf-8")
    assert committed == generated[name], (
        f"docs/{name} is stale or was edited by hand: run python scripts/build_docs.py"
    )


def test_the_change_control_plan_fits_on_one_page(generated: dict[str, str]) -> None:
    text = generated["change-control-plan.md"]
    assert len(text.split()) <= ONE_PAGE_WORDS, len(text.split())
    assert rendered_lines(text) <= ONE_PAGE_LINES, rendered_lines(text)


def test_the_change_control_plan_covers_serving_monitoring_and_retraining(
    generated: dict[str, str],
) -> None:
    text = generated["change-control-plan.md"]
    for part in ("**Serving.**", "**Monitoring.**", "Retraining after drift", "config/slo.json",
                 "rollback", "manifest", "skin type", "cdm.contract"):  # fmt: skip
        assert part.lower() in text.lower(), part
