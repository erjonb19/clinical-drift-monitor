"""Build the Phase 2 drift table from the per-model detection results and apply the ship
rule fixed before any Phase 2 result: ship the detector with the highest mean AUROC on the
two real held-out sites, FPR@95TPR breaking ties; benchmark sets are secondary. The rule is
applied to the centralized models (the ones served), with Mahalanobis at method D's PCA size;
method C's PCA size is a separately labelled sensitivity row and never takes part in the rule.

Usage: python scripts/drift_table.py --results results/phase2/drift/v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SETUPS = {"final": "centralized", "fedavg": "FedAvg", "fedprox": "FedProx"}
DETECTORS = ("mahalanobis", "gram", "max_softmax", "energy")
SENSITIVITY = "mahalanobis_pca_c"
SHIFTS = ("buenos_aires", "pad_ufes", "pathmnist_eval", "cifar10_eval")
REAL = ("buenos_aires", "pad_ufes")


def spread(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "min": float(np.min(values)),
            "max": float(np.max(values)), "n": len(values)}  # fmt: skip


def build(results: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for kind, setup in SETUPS.items():
        runs = [r for r in results if r["kind"] == kind]
        table[setup] = {
            det: {
                shift: {m: spread([r["metrics"][det][shift][m] for r in runs])
                        for m in ("auroc", "fpr95")}
                for shift in SHIFTS
            }
            for det in (*DETECTORS, SENSITIVITY)
        }  # fmt: skip
        table[setup]["seeds"] = sorted(r["seed"] for r in runs)
    central = table["centralized"]
    ranking = sorted(
        DETECTORS,
        key=lambda d: (-np.mean([central[d][s]["auroc"]["mean"] for s in REAL]),
                       np.mean([central[d][s]["fpr95"]["mean"] for s in REAL])),
    )  # fmt: skip
    rule = {
        d: {"real_sites_mean_auroc": float(np.mean([central[d][s]["auroc"]["mean"] for s in REAL])),
            "real_sites_mean_fpr95": float(np.mean([central[d][s]["fpr95"]["mean"] for s in REAL]))}
        for d in DETECTORS
    }  # fmt: skip
    return {"table": table, "ship_rule": rule, "ranking": ranking, "ship": ranking[0]}


def markdown(out: dict[str, Any]) -> str:
    def cell(x: dict[str, float]) -> str:
        return f"{100 * x['mean']:.1f} ({100 * x['min']:.1f}–{100 * x['max']:.1f})"

    lines = []
    for metric, title in (("auroc", "AUROC"), ("fpr95", "FPR@95TPR")):
        lines += [f"### {title}, % (mean and range over 3 seeds)", "",
                  "| Setup | Detector | Buenos Aires | PAD-UFES-20 | PathMNIST | CIFAR-10 |",
                  "| --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        for setup, rows in out["table"].items():
            for det in (*DETECTORS, SENSITIVITY):
                label = "Mahalanobis, PCA by method C (sensitivity)" if det == SENSITIVITY else det
                lines.append(
                    f"| {setup} | {label} | "
                    + " | ".join(cell(rows[det][s][metric]) for s in SHIFTS)
                    + " |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(args.results.glob("*-seed*.json"))]  # fmt: skip
    if len(results) != 9:
        raise SystemExit(f"expected 9 model results, found {len(results)}")
    out = build(results)
    out["pca_components"] = {r["model"]: r["pca"] for r in results}
    (args.results / "table.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (args.results / "table.md").write_text(markdown(out) + "\n", encoding="utf-8")
    print(markdown(out))
    print("ship rule (centralized, real sites):", json.dumps(out["ship_rule"], indent=1))
    print("ship:", out["ship"])


if __name__ == "__main__":
    main()
