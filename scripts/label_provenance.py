"""Label provenance: how each diagnosis in silver was confirmed, per site and class.

ISIC sites: ISIC's ``clinical.diagnosis_confirm_type`` (for example ``histopathology``), read
from the public metadata API in pages of 100 records per collection (no per-image calls). The
raw pages are cached, gzipped, in results/phase2/label_provenance/raw/ and reused on a rerun.
HAM10000: its own ``dx_type`` column from HAM10000_metadata.csv (histo, follow_up, consensus,
confocal), with ISIC's field for the same images reported beside it as a cross-check.
Labels, roles and splits come from the current silver table, so the counts describe exactly
the images the models trained and were scored on.

Usage: python scripts/label_provenance.py --ham-csv <HAM10000_metadata.csv>
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_export_evidence import Sql  # noqa: E402

from cdm.sources import API, http_json

OUT = Path("results/phase2/label_provenance")
RAW = OUT / "raw"
HAM_DX_TYPE = {"histo": "histopathology", "follow_up": "serial imaging (follow-up)",
               "consensus": "expert consensus", "confocal": "confocal microscopy"}  # fmt: skip


def pages(collection: int) -> list[dict[str, Any]]:
    """Every page of a collection's records, from the cache if present."""
    cache = RAW / f"collection_{collection}.json.gz"
    if cache.exists():
        return list(json.loads(gzip.decompress(cache.read_bytes())))
    url: str | None = API + "?" + urllib.parse.urlencode({"collections": collection, "limit": 100})
    got = []
    while url:
        page = http_json(url)
        got.append(page)
        url = page.get("next")
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(got).encode()))
    return got


def confirm_types(collection: int) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for page in pages(collection):
        for r in page["results"]:
            clinical = r.get("metadata", {}).get("clinical") or {}
            out[r["isic_id"]] = clinical.get("diagnosis_confirm_type")
    return out


def table(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (source, label), g in frame.groupby(["source", "label"], dropna=False):
        counts = Counter(g[column].fillna("not recorded"))
        n = len(g)
        out.setdefault(source, {})[str(label) if pd.notna(label) else "unlabelled"] = {
            "images": n,
            "by_method": dict(counts.most_common()),
            "histopathology_share": round(counts.get("histopathology", 0) / n, 4),
        }
    for source, g in frame.groupby("source"):
        counts = Counter(g[column].fillna("not recorded"))
        out[source]["all"] = {"images": len(g), "by_method": dict(counts.most_common()),
                              "histopathology_share": round(counts.get("histopathology", 0)
                                                            / len(g), 4)}  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ham-csv", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path("config/sites.json").read_text(encoding="utf-8"))
    silver = Sql()("SELECT isic_id, source, role, split, label FROM workspace.cdm.silver_images")
    isic: dict[str, str | None] = {}
    collections = {name: spec["collections"] for name, spec in config["sites"].items()}
    collections["ham10000"] = [config["ham10000"]["collection"]]
    for name, cols in collections.items():
        for c in cols:
            isic.update(confirm_types(c))
        print(name, "fetched", flush=True)
    silver["isic_confirm_type"] = silver["isic_id"].map(isic)
    ham = pd.read_csv(args.ham_csv).set_index("image_id")["dx_type"].map(HAM_DX_TYPE)
    silver["method"] = silver["isic_confirm_type"]
    is_ham = silver["source"] == "ham10000"
    silver.loc[is_ham, "method"] = silver.loc[is_ham, "isic_id"].map(ham)
    missing = int(silver["isic_id"].map(lambda i: i not in isic).sum())
    result = {
        "sources": {"isic_sites": "ISIC metadata clinical.diagnosis_confirm_type",
                    "ham10000": "HAM10000_metadata.csv dx_type"},
        "silver_images": len(silver),
        "images_missing_from_isic_metadata": missing,
        "by_site_and_class": table(silver, "method"),
        "ham10000_isic_field_cross_check": table(silver[is_ham], "isic_confirm_type"),
        "raw_cache": sorted(p.name for p in RAW.glob("*.json.gz")),
    }  # fmt: skip
    (OUT / "provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: v["all"] for s, v in result["by_site_and_class"].items()}, indent=1))


if __name__ == "__main__":
    main()
