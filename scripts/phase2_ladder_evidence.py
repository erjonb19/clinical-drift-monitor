"""Save one tuning-ladder run's notebook output to results/phase2/ladder/<rung>.json.

Usage: python scripts/phase2_ladder_evidence.py --rung <name> --run <job run id>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_stage2_evidence import notebook_output  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    out = Path("results/phase2/ladder") / f"{args.rung}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **notebook_output(args.run),
    }
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
