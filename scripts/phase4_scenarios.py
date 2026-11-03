"""Write the Phase 4 eval's synthetic drift scenarios (staged) to results/phase4/scenarios.json.

Usage: python scripts/phase4_scenarios.py [--check]

Every scenario is built by `cdm.scenarios.build` from seeded synthetic scores and the monitoring
code's own drift tests, so rerunning reproduces the committed file exactly. ``--check`` fails
if the committed file differs from a fresh build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cdm.scenarios import STAGED, build

OUT = Path("results/phase4/scenarios.json")


def render() -> str:
    scenarios = build()
    doc = {"staged": STAGED, "built_by": "scripts/phase4_scenarios.py", "count": len(scenarios),
           "scenarios": scenarios}  # fmt: skip
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        same = OUT.exists() and OUT.read_text() == text
        print("scenarios.json matches a fresh build" if same else "scenarios.json differs")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
