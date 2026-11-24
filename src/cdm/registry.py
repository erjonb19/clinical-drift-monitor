"""Backlog item 2: move the ``champion`` alias between registry versions, and back.

Only the ``champion`` alias decides what serves (owner, 2026-11-06). A move is planned here and
refused unless it is safe:
- the alias must currently point where the caller says (``expect_current``), so two people
  moving it at once cannot silently undo each other;
- the target version must exist, differ from the current one, carry a ``model_gen`` tag, and
  not be marked ``rejected`` (a challenger the rule turned down is registered but never
  promoted by hand by mistake: promotion goes through ``promote_challenger``, item 17);
- a rollback (to a version older than the current one) needs no decision file; a promotion
  (to a newer one) must name the decision file's SHA-256 that allowed it.

The MLflow calls live in ``scripts/move_champion.py``; this module is pure, so it is tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NAME = "workspace.cdm.skin_lesion_classifier"
ALIAS = "champion"


class MoveRefused(ValueError):
    """The planned alias move is not safe; the message says why."""


@dataclass(frozen=True)
class Move:
    from_version: int
    to_version: int
    kind: str  # "rollback" or "promotion"
    model_gen: str
    decision_sha256: str | None


def plan_move(
    versions: Sequence[Mapping[str, Any]],
    current: int,
    expect_current: int,
    target: int,
    decision_sha256: str | None = None,
) -> Move:
    """``versions`` are the registry's versions as ``{"version": int, "tags": {...}}``."""
    if current != expect_current:
        raise MoveRefused(
            f"the alias points at v{current}, not v{expect_current}: someone moved it"
        )
    by_version = {int(v["version"]): dict(v.get("tags") or {}) for v in versions}
    if target not in by_version:
        raise MoveRefused(f"v{target} is not in the registry ({sorted(by_version)})")
    if target == current:
        raise MoveRefused(f"the alias already points at v{target}")
    tags = by_version[target]
    if not tags.get("model_gen"):
        raise MoveRefused(f"v{target} has no model_gen tag; tag it first")
    if tags.get("decision") == "rejected":
        raise MoveRefused(f"v{target} was rejected by the promotion rule")
    kind = "rollback" if target < current else "promotion"
    if kind == "promotion" and not decision_sha256:
        raise MoveRefused("a promotion names the SHA-256 of the decision file that allowed it")
    return Move(current, target, kind, str(tags["model_gen"]), decision_sha256)


def move_record(move: Move, who: str, when_utc: str, check: str) -> dict[str, Any]:
    """One line of ``results/phase3/registry/alias_moves.jsonl``."""
    return {
        "model": NAME,
        "alias": ALIAS,
        "from_version": move.from_version,
        "to_version": move.to_version,
        "kind": move.kind,
        "model_gen": move.model_gen,
        "decision_sha256": move.decision_sha256,
        "by": who,
        "utc": when_utc,
        "check_after": check,
    }
