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


def test_a_workspace_name_imports_the_notebook_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 1's rehearsal stages monthly_drift under another workspace name, so the real
    monthly job's notebook is never replaced."""
    module = script()
    calls: list[tuple[str, ...]] = []

    def fake_run(*args: str, capture: bool = True) -> str:
        calls.append(args)
        if args[:2] == ("git", "rev-parse"):
            return "abc1234"
        if args[:2] == ("databricks", "current-user"):
            return '{"userName": "someone"}'
        if args[1:3] == ("-m", "pip"):
            Path(args[-1], module.WHEEL).write_bytes(b"wheel")
        return ""

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b"x"})())
    module.main(["--workspace-name", "rehearsal_monthly_drift", "monthly_drift", "a.json"])
    imports = [c for c in calls if c[:3] == ("databricks", "workspace", "import")]
    assert imports and imports[0][3] == "/Users/someone/cdm/rehearsal_monthly_drift"
    with pytest.raises(SystemExit, match="letters"):
        module.main(["--workspace-name", "../x", "monthly_drift", "a.json"])
