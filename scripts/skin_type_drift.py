"""Drift by skin type from the drift run's saved GPU scores (spec part A:
results/phase2/analysis_look/spec.json). No new look at any image.

Matches each saved Buenos Aires and PAD-UFES-20 score to its image by the drift notebook's
order (silver_images ordered by source, isic_id), checks the counts, and reports Mahalanobis
AUROC and FPR@95TPR (test vs the group) within each site for the skin types with at least 100
images; types IV to VI as counts only.

Usage: python scripts/skin_type_drift.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql  # noqa: E402

from cdm.eval import detection_metrics

VOLUME = "dbfs:/Volumes/workspace/cdm/raw/results/phase2/drift/v2"
MODELS = ("v2-final-seed0", "v2-final-seed1", "v2-final-seed2")
SITES = ("buenos_aires", "pad_ufes")
MIN_IMAGES = 100
OUT = Path("results/phase2/analysis_look/skin_type_drift.json")


def main() -> None:
    meta = Sql()(
        """SELECT s.source, s.isic_id, m.fitzpatrick_skin_type AS skin_type
           FROM workspace.cdm.silver_images s
           LEFT JOIN workspace.cdm.silver_metadata m USING (source, isic_id)
           WHERE s.source IN ('buenos_aires', 'pad_ufes') ORDER BY s.source, s.isic_id"""
    )
    types = {s: meta.loc[meta["source"] == s, "skin_type"].fillna("not recorded").to_numpy()
             for s in SITES}  # fmt: skip
    per_model: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key in MODELS:
            local = Path(tmp) / f"{key}.npz"
            subprocess.run(["databricks", "fs", "cp", f"{VOLUME}/{key}_scores.npz", str(local),
                            "--overwrite"], check=True, capture_output=True)  # fmt: skip
            with np.load(local) as z:
                test = z["test__mahalanobis"]
                per_model[key] = {}
                for site in SITES:
                    scores = z[f"{site}__mahalanobis"]
                    assert len(scores) == len(types[site]), (site, len(scores), len(types[site]))
                    per_model[key][site] = {
                        t: detection_metrics(test, scores[types[site] == t])
                        for t in sorted(set(types[site]))
                        if (types[site] == t).sum() >= MIN_IMAGES and t != "not recorded"
                    }
    groups: dict[str, Any] = {}
    for site in SITES:
        counts = {t: int((types[site] == t).sum()) for t in sorted(set(types[site]))}
        reported = sorted(per_model[MODELS[0]][site])
        groups[site] = {
            "images_by_skin_type": counts,
            "reported_types": reported,
            "counts_only_gap": {t: n for t, n in counts.items() if t in ("IV", "V", "VI")},
            "mahalanobis": {
                t: {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                    for m in ("auroc", "fpr95")
                    for v in [[per_model[k][site][t][m] for k in MODELS]]}
                for t in reported
            },
        }  # fmt: skip
    result = {
        "spec": "results/phase2/analysis_look/spec.json, part A",
        "source_scores": "results/phase2/drift/v2 (drift run 457781950298490), Mahalanobis (D)",
        "note": "within each site only, never pooled; skin type is mixed up with site and imaging "
        "type, so no skin-type effect is claimed; types IV to VI are counts only (a gap)",
        "by_site": groups,
        "per_model": per_model,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for site in SITES:
        print(site, groups[site]["images_by_skin_type"])
        for t, v in groups[site]["mahalanobis"].items():
            print(
                f"  type {t}: AUROC {100 * v['auroc']['mean']:.1f} ({100 * v['auroc']['min']:.1f}-"
                f"{100 * v['auroc']['max']:.1f}), FPR@95 {100 * v['fpr95']['mean']:.1f}"
            )


if __name__ == "__main__":
    main()
