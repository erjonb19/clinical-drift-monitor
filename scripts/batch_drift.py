"""Run the batch-level drift spec (results/phase2/batch_drift/spec.json) from saved scores.

Reads each v2 centralized model's test and shift scores from the drift run and its
validation reference scores from the reference job (downloaded from the volume into a
temporary folder), runs the KS simulation for Mahalanobis and Gram, and writes
results/phase2/batch_drift/results.json with per-seed rates and their mean and range.

Usage: python scripts/batch_drift.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cdm.batch_drift import simulate

SPEC = Path("results/phase2/batch_drift/spec.json")
OUT = Path("results/phase2/batch_drift/results.json")
VOLUME = "dbfs:/Volumes/workspace/cdm/raw/results/phase2"
MODELS = ("v2-final-seed0", "v2-final-seed1", "v2-final-seed2")
DETECTORS = ("mahalanobis",)  # Gram dropped by spec amendment 2
SITES = ("buenos_aires", "pad_ufes")


def fetch(remote: str, local: Path) -> Path:
    subprocess.run(["databricks", "fs", "cp", remote, str(local), "--overwrite"], check=True,
                   capture_output=True)  # fmt: skip
    return local


def spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    check = None
    per_model: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        check = {}
        for key in MODELS:
            check[key] = json.loads(fetch(f"{VOLUME}/batch_drift/{key}_check.json",
                                          folder / f"{key}_check.json").read_text())  # fmt: skip
            assert check[key]["passed"], f"{key}: the reference check did not pass"
            shifts = np.load(fetch(f"{VOLUME}/drift/v2/{key}_scores.npz", folder / f"{key}.npz"))
            ref = np.load(fetch(f"{VOLUME}/batch_drift/{key}_val_scores.npz",
                                folder / f"{key}_val.npz"))  # fmt: skip
            per_model[key] = {
                det: simulate(
                    reference=ref[det], in_dist=shifts[f"test__{det}"],
                    sites={s: shifts[f"{s}__{det}"] for s in SITES},
                    sizes=spec["batch_sizes"], shares=spec["new_site_shares"],
                    batches=spec["batches_per_cell"], alpha=spec["alpha"], seed=spec["random_seed"],
                )
                for det in DETECTORS
            }  # fmt: skip
            print(key, "done", flush=True)
    summary: dict[str, Any] = {}
    for det in DETECTORS:
        runs = [per_model[k][det] for k in MODELS]
        summary[det] = {
            "false_alarm_rate": {size: spread([r["false_alarm_rate"][size] for r in runs])
                                 for size in runs[0]["false_alarm_rate"]},
            "detection_rate": {site: {size: {share: spread([r["detection_rate"][site][size][share]
                                                            for r in runs])
                                             for share in runs[0]["detection_rate"][site][size]}
                                      for size in runs[0]["detection_rate"][site]}
                               for site in SITES},
        }  # fmt: skip
    # Spec's false-alarm rule: more than 2% of 0% batches flagged, for any model and
    # detector, is a finding (validation baseline vs in-distribution test); no new baseline.
    findings = [
        {"model": k, "detector": det, "batch_size": size, "false_alarm_rate": rate}
        for k in MODELS for det in DETECTORS
        for size, rate in per_model[k][det]["false_alarm_rate"].items() if rate > 0.02
    ]  # fmt: skip
    result = {"spec": spec, "reference_check": check, "per_model": per_model,
              "over_seeds": summary, "false_alarm_findings": findings,
              "gram": "dropped by spec amendment 2 (CPU refit missed the GPU normalizer by 11.8%)",
              "setup_tested": "reference scores were recomputed on CPU from a CPU refit of the "
              "detector and checked against the GPU scores, so these results also test "
              "computing the shipped detector on CPU"}  # fmt: skip
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("false-alarm findings (>2%):", findings or "none")
    for det in DETECTORS:
        s = summary[det]
        print(
            f"\n{det}: false alarm (0%) by size",
            {k: round(100 * v["mean"], 1) for k, v in s["false_alarm_rate"].items()},
        )
        for site in SITES:
            for size, row in s["detection_rate"][site].items():
                print(f"  {site:13s} n={size:>3}", {sh: round(100 * v["mean"], 1)
                                                     for sh, v in row.items()})  # fmt: skip


if __name__ == "__main__":
    main()
