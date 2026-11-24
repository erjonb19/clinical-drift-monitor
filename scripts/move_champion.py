"""Move the registry's ``champion`` alias to another version, or back (backlog item 2).

Run by the owner from the laptop (cloud sessions have no workspace access). Prints the planned
move and changes nothing unless ``--apply`` is given. The move is refused unless the alias
points where ``--expect-current`` says, the target carries a ``model_gen`` tag and was not
rejected by the promotion rule, and a promotion names its decision file's SHA-256
(``cdm.registry``). After an applied move it appends one line to
``results/phase3/registry/alias_moves.jsonl`` (commit it), and the owner runs the
``check_registered_model`` job with the target version's serving bundle to show the served
answers match that version (docs/rollback.md).

Usage:
  python scripts/move_champion.py --expect-current 2 --to 1             # plan a rollback
  python scripts/move_champion.py --expect-current 2 --to 1 --apply     # do it
  python scripts/move_champion.py --expect-current 1 --to 2 --decision-sha256 <hex> --apply
  python scripts/move_champion.py --tag 1 model_gen v2 --apply          # tag a version
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdm.registry import ALIAS, NAME, MoveRefused, move_record, plan_move

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "results" / "phase3" / "registry" / "alias_moves.jsonl"


def client() -> Any:
    import mlflow  # the `evidence` extra; not installed in CI

    mlflow.set_registry_uri("databricks-uc")
    return mlflow.MlflowClient(registry_uri="databricks-uc")


def versions(mc: Any) -> list[dict[str, Any]]:
    """Unity Catalog's search results carry no tags (reading ``.tags`` raises), so the search
    lists version numbers only and each version's tags are read with ``get_model_version``."""
    numbers = sorted({int(v.version) for v in mc.search_model_versions(f"name='{NAME}'")})
    return [
        {"version": n, "tags": dict(mc.get_model_version(NAME, str(n)).tags or {})} for n in numbers
    ]


def main(argv: list[str] | None = None, mc: Any = None, log: Path = LOG) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-current", type=int)
    parser.add_argument("--to", type=int)
    parser.add_argument("--decision-sha256")
    parser.add_argument("--tag", nargs=3, metavar=("VERSION", "KEY", "VALUE"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    mc = mc if mc is not None else client()
    if args.tag:
        version, key, value = args.tag
        print(f"tag v{version} {key}={value}" + ("" if args.apply else " (dry run)"))
        if args.apply:
            mc.set_model_version_tag(NAME, version, key, value)
        return 0
    if args.expect_current is None or args.to is None:
        parser.error("--expect-current and --to are required for a move")
    current = int(mc.get_model_version_by_alias(NAME, ALIAS).version)
    try:
        move = plan_move(versions(mc), current, args.expect_current, args.to, args.decision_sha256)
    except MoveRefused as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(f"{move.kind}: {ALIAS} v{move.from_version} -> v{move.to_version} ({move.model_gen})")
    if not args.apply:
        print("dry run: nothing changed (add --apply)")
        return 0
    mc.set_registered_model_alias(NAME, ALIAS, move.to_version)
    after = int(mc.get_model_version_by_alias(NAME, ALIAS).version)
    if after != move.to_version:
        print(f"the alias reads v{after} after the move, not v{move.to_version}", file=sys.stderr)
        return 1
    record = move_record(
        move,
        who="owner",
        when_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        check="pending: run check_registered_model with this version's bundle (docs/rollback.md)",
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"moved; logged to {log.relative_to(REPO) if log.is_relative_to(REPO) else log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
