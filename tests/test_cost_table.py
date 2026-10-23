"""scripts/cost_table.py: SQL text values become numbers, and unnamed jobs are labelled."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def script() -> ModuleType:
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("cost_table", REPO / "scripts" / "cost_table.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_typed_rows_make_numbers_and_integer_counts() -> None:
    rows = script().typed([{"name": "x", "dbus": "1.5", "runs": "2", "result_state": "SUCCEEDED"}])
    assert rows == [{"name": "x", "dbus": 1.5, "runs": 2, "result_state": "SUCCEEDED"}]


def test_an_unknown_sku_stops_the_build() -> None:
    module = script()
    assert module.compute_type("PREMIUM_MODEL_TRAINING_US_EAST_OHIO") == "gpu"
    with pytest.raises(ValueError, match="unknown SKU"):
        module.compute_type("PREMIUM_SOMETHING_NEW")


def test_wall_minutes_count_once_per_run_across_skus() -> None:
    rows = [
        {"name": "j", "compute": "cpu", "run_key": "1", "wall_minutes": 10.0, "dbus": 1.0,
         "list_price_usd": 0.4},
        {"name": "j", "compute": "cpu", "run_key": "1", "wall_minutes": 10.0, "dbus": 2.0,
         "list_price_usd": 0.6},
    ]  # fmt: skip
    (row,) = script().summarise(rows)
    assert row["runs"] == 1 and row["wall_minutes"] == 10.0 and row["dbus"] == 3.0
