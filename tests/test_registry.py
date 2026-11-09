"""Backlog item 2: moving the champion alias, and back (cdm.registry, scripts/move_champion.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cdm.registry import MoveRefused, plan_move

REPO = Path(__file__).resolve().parents[1]
VERSIONS = [
    {"version": 1, "tags": {"model_gen": "v2"}},
    {"version": 2, "tags": {"model_gen": "v3", "decision": "promoted_by_rule"}},
    {"version": 3, "tags": {"model_gen": "v4", "decision": "rejected"}},
    {"version": 4, "tags": {}},
]


def test_a_rollback_needs_no_decision_file() -> None:
    move = plan_move(VERSIONS, current=2, expect_current=2, target=1)
    assert (move.kind, move.from_version, move.to_version, move.model_gen) == (
        "rollback",
        2,
        1,
        "v2",
    )


def test_a_promotion_names_its_decision() -> None:
    with pytest.raises(MoveRefused, match="SHA-256"):
        plan_move(VERSIONS, current=1, expect_current=1, target=2)
    assert plan_move(VERSIONS, 1, 1, 2, decision_sha256="ab" * 32).kind == "promotion"


@pytest.mark.parametrize(
    ("current", "expect", "target", "why"),
    [
        (2, 1, 1, "someone moved it"),
        (2, 2, 9, "not in the registry"),
        (2, 2, 2, "already points"),
        (2, 2, 4, "no model_gen"),
        (2, 2, 3, "rejected"),
    ],
)
def test_unsafe_moves_are_refused(current: int, expect: int, target: int, why: str) -> None:
    with pytest.raises(MoveRefused, match=why):
        plan_move(VERSIONS, current, expect, target, decision_sha256="ab" * 32)


class FakeRegistry:
    """Stands in for mlflow.MlflowClient: the alias, the versions and the calls made."""

    def __init__(self, alias: int) -> None:
        self.alias = alias
        self.calls: list[tuple[object, ...]] = []

    def search_model_versions(self, _: str) -> list[Any]:
        return [SimpleNamespace(version=str(v["version"]), tags=v["tags"]) for v in VERSIONS]

    def get_model_version_by_alias(self, _: str, __: str) -> Any:
        return SimpleNamespace(version=str(self.alias))

    def set_registered_model_alias(self, _: str, __: str, version: int) -> None:
        self.calls.append(("alias", version))
        self.alias = version

    def set_model_version_tag(self, *args: str) -> None:
        self.calls.append(("tag", *args))


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("move", REPO / "scripts" / "move_champion.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_moves_and_logs_only_with_apply(tmp_path: Path) -> None:
    script, registry, log = _script(), FakeRegistry(alias=2), tmp_path / "moves.jsonl"
    assert script.main(["--expect-current", "2", "--to", "1"], registry, log) == 0
    assert registry.calls == [] and not log.exists()  # a dry run changes nothing
    assert script.main(["--expect-current", "2", "--to", "1", "--apply"], registry, log) == 0
    assert registry.calls == [("alias", 1)] and registry.alias == 1
    line = json.loads(log.read_text())
    assert (line["kind"], line["from_version"], line["to_version"]) == ("rollback", 2, 1)
    assert line["check_after"].startswith("pending")


def test_the_script_refuses_without_touching_the_alias(tmp_path: Path) -> None:
    script, registry = _script(), FakeRegistry(alias=2)
    argv = ["--expect-current", "1", "--to", "1", "--apply"]
    assert script.main(argv, registry, tmp_path / "m.jsonl") == 1
    assert registry.calls == []
