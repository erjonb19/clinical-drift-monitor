"""Phase 1 source checks, run from a laptop against the live ISIC API.

1. Profile each configured site (images, lesions, labels, licenses, skin type) with the same
   code the ingest task uses, including the pinned Barcelona selection.
2. Check that the diagnosis mapping reproduces HAM10000's labels on HAM10000's images.
3. Compare ISIC's HAM10000 files with the Dataverse files from Phase 0 for a seeded sample:
   size, JPEG quantisation, mean absolute pixel difference. Needs Phase 0's data under
   CDM_DATA.

Usage: CDM_DATA=... python scripts/phase1_isic_checks.py
Writes results/phase1/isic_checks.json.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cdm.data import data_root
from cdm.images import jpeg_quant_mean
from cdm.sources import fingerprint, ham_rows, http_bytes, load_sites, site_rows

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sites.json"
SAMPLE = 20


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"sites": {}}

    ham_dir = data_root() / "ham10000"
    meta = pd.read_csv(ham_dir / config["ham10000"]["metadata_file"])
    ham = pd.DataFrame(ham_rows(meta, config["ham10000"]["collection"]))
    report["ham10000"] = {
        "images": len(ham),
        "mapping_matches_ham_label": int((ham["isic_label"] == ham["label"]).sum()),
        "licenses": ham["license"].value_counts().to_dict(),
    }

    for site in load_sites(config):
        rows = pd.DataFrame(site_rows(site))
        lesions = rows["isic_lesion_id"].fillna(rows["isic_id"])
        report["sites"][site.name] = {
            "images": len(rows),
            "lesions": int(lesions.nunique()),
            "images_without_lesion_id": int(rows["isic_lesion_id"].isna().sum()),
            "images_in_ham10000": int(rows["isic_id"].isin(set(ham["isic_id"])).sum()),
            "labels": rows["isic_label"].fillna("unmapped").value_counts().to_dict(),
            "with_skin_type": int(rows["fitzpatrick_skin_type"].notna().sum()),
            "licenses": rows["license"].value_counts().to_dict(),
            "missing_attribution": int(rows["attribution"].isna().sum()),
            "image_types": rows["image_type"].value_counts().to_dict(),
            "selection_fingerprint": fingerprint(rows["isic_id"]) if site.cap else None,
        }
        print(site.name, report["sites"][site.name]["images"], flush=True)

    rng = random.Random(0)
    comparisons = []
    for _, row in ham.sample(n=SAMPLE, random_state=rng.randint(0, 2**31)).iterrows():
        remote = http_bytes(row["url"])
        local = (ham_dir / "images" / f"{row['isic_id']}.jpg").read_bytes()
        with Image.open(io.BytesIO(remote)) as a, Image.open(io.BytesIO(local)) as b:
            same_size = a.size == b.size
            diff = (
                float(
                    np.abs(
                        np.asarray(a.convert("RGB"), dtype=np.float64)
                        - np.asarray(b.convert("RGB"), dtype=np.float64)
                    ).mean()
                )
                if same_size
                else None
            )
            comparisons.append(
                {
                    "isic_id": row["isic_id"],
                    "isic_bytes": len(remote),
                    "dataverse_bytes": len(local),
                    "identical_bytes": remote == local,
                    "same_pixel_size": same_size,
                    "isic_jpeg_quant_mean": jpeg_quant_mean(a),
                    "dataverse_jpeg_quant_mean": jpeg_quant_mean(b),
                    "mean_abs_pixel_diff": round(diff, 3) if diff is not None else None,
                }
            )
    diffs = [c["mean_abs_pixel_diff"] for c in comparisons if c["mean_abs_pixel_diff"] is not None]
    report["ham10000_isic_vs_dataverse"] = {
        "sample": SAMPLE,
        "identical_bytes": sum(c["identical_bytes"] for c in comparisons),
        "same_pixel_size": sum(c["same_pixel_size"] for c in comparisons),
        "mean_abs_pixel_diff_min": min(diffs),
        "mean_abs_pixel_diff_max": max(diffs),
        "images": comparisons,
    }

    out = Path("results/phase1/isic_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in report.items() if k != "ham10000_isic_vs_dataverse"}, indent=2)
    )
    print({k: v for k, v in report["ham10000_isic_vs_dataverse"].items() if k != "images"})


if __name__ == "__main__":
    main()
