"""The investigator quickstart notebook runs end to end (offline, on its synthetic stand-in)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_quickstart_notebook_runs_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDM_QUICKSTART_OFFLINE", "1")
    monkeypatch.chdir(REPO / "notebooks")
    notebook = json.loads((REPO / "notebooks" / "quickstart.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"], "commit the notebook without outputs"
            exec("".join(cell["source"]), namespace)  # noqa: S102 - the repo's own notebook
    gates = namespace["gates"]
    assert all(g.violations == 0 for g in gates)  # type: ignore[attr-defined]
    result = namespace["result"]
    assert result.attrition[0]["persons"] == "15"  # type: ignore[attr-defined]
    # The first run was refused (no approval yet), and every shown cell is safe.
    events = [e["event"] for e in namespace["service"].audit.entries()]  # type: ignore[attr-defined]
    assert events[:3] == ["refused", "decision", "ran"]
    for row in result.breakdown:  # type: ignore[attr-defined]
        assert (
            row["persons"] == "suppressed" or int(row["persons"]) == 0 or int(row["persons"]) >= 11
        )
