"""scripts/stage_research_job.py refuses before uploading anything."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stage_research_job", REPO / "scripts" / "stage_research_job.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_needs_a_notebook_and_at_least_one_file() -> None:
    with pytest.raises(SystemExit):
        script().main(["subgroups"])


def test_refuses_a_dirty_tree_before_any_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    module = script()
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str, capture: bool = True) -> str:
        calls.append(args)
        return "?? src/cdm/new.py" if args[:2] == ("git", "status") else ""

    monkeypatch.setattr(module, "run", fake_run)
    with pytest.raises(SystemExit, match="uncommitted"):
        module.main(["subgroups", "results/x.json"])
    assert len(calls) == 1
    assert "pyproject.toml" in calls[0] and "databricks/notebooks/subgroups.py" in calls[0]
