"""scripts/cost_table.py: SQL text values become numbers, and unnamed jobs are labelled."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_typed_rows() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("cost_table", REPO / "scripts" / "cost_table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.typed(
        [{"job_name": None, "dbus": "1.5", "runs": "2", "result_state": "SUCCEEDED"}]
    )
    assert rows == [{"job_name": "unnamed (one-off run or deleted job)", "dbus": 1.5, "runs": 2.0,
                     "result_state": "SUCCEEDED"}]  # fmt: skip
