"""Apply the v1 vs v2 data keep rule (cdm.ladder.keep_data_version) on validation, before
any v2 test look, and save it to results/phase2/v2/decision.json.

The rule uses the centralized models only: mean over seeds 0-2 on v1's exact validation
images. v1's scores come from the committed v1 runs (seed 0 from the tuning ladder's
control, seeds 1-2 from the overnight run); v2's from the v2 runs' ``val_v1_images``
report. Federated v1 vs v2 validation is reported beside it for information only.

Usage: python scripts/v2_decision.py --v2 results/phase2/v2/training
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cdm.ladder import VERSION_MAX_CLIENT_DROP, VERSION_MIN_GAIN, keep_data_version

CLIENTS = ("ham_vienna", "ham_queensland", "barcelona", "msk")
V1 = Path("results/phase2")
OUT = Path("results/phase2/v2/decision.json")


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def v1_val(kind: str, seed: int) -> dict[str, Any]:
    if kind == "final" and seed == 0:
        return dict(load(V1 / "ladder" / "control-center384.json")["output"]["scores"]["val"])
    name = f"{kind}-seed{seed}.json"
    path = V1 / "overnight" / name
    if not path.exists():  # FedProx seed 1 finished in the rerun
        path = V1 / "overnight" / "rerun-fedprox-seed1" / name
    return dict(load(path)["scores"]["val"])


def v2_val(folder: Path, kind: str, seed: int) -> dict[str, Any]:
    run = load(folder / f"v2-{kind}-seed{seed}.json")
    return {"v1_images": run["scores"]["val_v1_images"], "full_v2": run["scores"]["val"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True, help="folder with the v2 run results")
    args = parser.parse_args()
    seeds = (0, 1, 2)
    v1 = [v1_val("final", s) for s in seeds]
    v2 = [v2_val(args.v2, "final", s) for s in seeds]
    # Same images on both sides, or the comparison is not the one the rule describes.
    for group in ("pooled", *CLIENTS):
        sizes = {r[group]["images"] for r in v1} | {r["v1_images"][group]["images"] for r in v2}
        if len(sizes) != 1:
            raise SystemExit(f"{group}: v1 and v2 validation image counts differ: {sizes}")
    decision = keep_data_version(v1, [r["v1_images"] for r in v2], CLIENTS)
    info: dict[str, Any] = {}
    for kind in ("fedavg", "fedprox"):
        a = [v1_val(kind, s) for s in seeds]
        b = [v2_val(args.v2, kind, s)["v1_images"] for s in seeds]
        info[kind] = keep_data_version(a, b, CLIENTS)
        info[kind].pop("keep_v2")  # information only: the rule is applied to centralized
    full = {
        g: sum(r["full_v2"][g]["balanced_accuracy"] for r in v2) / len(v2)
        for g in ("pooled", *CLIENTS)
    }
    evidence = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rule": {"min_pooled_gain_points": VERSION_MIN_GAIN,
                 "max_client_drop_points": VERSION_MAX_CLIENT_DROP,
                 "models": "centralized, seeds 0-2, mean",
                 "images": "v1's exact validation images"},
        "centralized": decision,
        "decision": "v2" if decision["keep_v2"] else "v1",
        "federated_for_information": info,
        "v2_centralized_mean_on_full_v2_validation": full,
        "v1_validation_images": v1[0]["pooled"]["images"],
        "v2_validation_images_v1_subset": v2[0]["v1_images"]["pooled"]["images"],
    }  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ("decision", "centralized")}, indent=2))


if __name__ == "__main__":
    main()
