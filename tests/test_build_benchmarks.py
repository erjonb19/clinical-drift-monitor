"""Backlog item 33: CIFAR-10 batch files are unpickled only after their published MD5 matches."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("bb", REPO / "scripts" / "build_benchmarks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_file_that_does_not_match_its_checksum_is_never_unpickled(tmp_path: Path) -> None:
    script = _script()
    ran: list[str] = []

    class Planted:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return (ran.append, ("code ran",))

    (tmp_path / "test_batch").write_bytes(pickle.dumps(Planted()))
    with pytest.raises(ValueError, match="MD5"):
        script.cifar_rows(tmp_path, ["test_batch"], {0})
    assert ran == []


def test_unknown_file_names_are_refused(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "other").write_bytes(b"x")
    with pytest.raises(ValueError, match="not an official"):
        script.checked_cifar_bytes(tmp_path / "other")
