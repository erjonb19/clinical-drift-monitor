"""Write config/champion_smoke_images.csv for the champion smoke test (backlog item 6).

Picks up to 50 public ISIC images of the clients' **test** split (lesion-level, from
config/splits_v1.csv, so none was seen in training), records each image's license and
attribution and the SHA-256 of its file, and writes the manifest. The images themselves are
never saved. Needs the ISIC API (api.isic-archive.com); run it where that is reachable, then
commit the CSV.

Usage: python scripts/make_smoke_manifest.py [--n 50]

Every returned image's own lesion ID must equal the one asked for, or the run stops.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SEARCH = "https://api.isic-archive.com/api/v2/images/search/?"
OUT = REPO / "config" / "champion_smoke_images.csv"
# diagnosis is ISIC's coarse diagnosis_1 (Benign, Malignant, Indeterminate), for reading only.
FIELDS = ["isic_id", "lesion_id", "diagnosis", "license", "attribution", "url", "sha256"]
# The licences this project may use (CLAUDE.md); an image under any other is skipped.
ALLOWED_LICENSES = {"CC-0", "CC-BY", "CC-BY-NC"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    return data


def image_lesion(image: dict[str, Any]) -> str | None:
    """The lesion an ISIC image belongs to. ISIC's API files every clinical field, lesion_id
    included, under metadata.clinical (ImageSerializer.get_metadata in ImageMarkup/isic)."""
    lesion = ((image.get("metadata") or {}).get("clinical") or {}).get("lesion_id")
    return str(lesion) if lesion else None


def first_image(lesion_id: str) -> dict[str, Any] | None:
    """The first image of the lesion, refused unless the API says it belongs to that lesion
    (so a query the API reads differently cannot slip in an image from another split)."""
    query = urllib.parse.urlencode({"query": f"lesion_id:{lesion_id}", "limit": 1})
    results = json.loads(get(SEARCH + query)).get("results", [])
    if not results:
        return None
    if image_lesion(results[0]) != lesion_id:
        raise SystemExit(
            f"{lesion_id}: the API returned {results[0].get('isic_id')} of lesion "
            f"{image_lesion(results[0])}; the query syntax is not doing what we expect"
        )
    image: dict[str, Any] = results[0]
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    with (REPO / "config" / "splits_v1.csv").open(encoding="utf-8") as f:
        lesions = sorted(r["lesion_id"] for r in csv.DictReader(f) if r["split"] == "test")
    rows: list[dict[str, str]] = []
    for lesion in lesions:
        if len(rows) == args.n:
            break
        image = first_image(lesion)
        url = ((image or {}).get("files") or {}).get("full", {}).get("url")
        if image is None or not url or image.get("copyright_license") not in ALLOWED_LICENSES:
            continue
        meta = image.get("metadata", {}).get("clinical", {})
        rows.append(
            {
                "isic_id": image["isic_id"],
                "lesion_id": lesion,
                "diagnosis": meta.get("diagnosis_1") or meta.get("diagnosis") or "",
                "license": image.get("copyright_license", ""),
                "attribution": image.get("attribution", ""),
                "url": url,
                "sha256": hashlib.sha256(get(url)).hexdigest(),
            }
        )
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images to {OUT}")


if __name__ == "__main__":
    main()
