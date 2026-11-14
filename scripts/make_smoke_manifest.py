"""Write config/champion_smoke_images.csv for the champion smoke test (backlog item 6).

Picks up to 50 public ISIC images of the clients' **test** split (lesion-level, from
config/splits_v1.csv, so none was seen in training), records each image's license and
attribution and the SHA-256 of its file, and writes the manifest. The images themselves are
never saved. Needs the ISIC API (api.isic-archive.com); run it where that is reachable, then
commit the CSV.

Usage: python scripts/make_smoke_manifest.py [--n 50]
       python scripts/make_smoke_manifest.py --collection 406 \
           --out config/compose_demo_shift_images.csv

Every returned image's own lesion ID must equal the one asked for, or the run stops.

``--collection`` (backlog item 21) instead takes the first images, one per lesion, of one ISIC
collection that the model never trained on: PAD-UFES-20 (406), smartphone photos, the compose
demo's shifted batch. Every image must be a clinical (non-dermoscopic) photo, or the run stops,
so a query the API reads differently cannot slip in dermoscopic images.
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
IMAGES = "https://api.isic-archive.com/api/v2/images/?"
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


def row(image: dict[str, Any], lesion: str, url: str) -> dict[str, str]:
    meta = image.get("metadata", {}).get("clinical", {})
    return {
        "isic_id": image["isic_id"],
        "lesion_id": lesion,
        "diagnosis": meta.get("diagnosis_1") or meta.get("diagnosis") or "",
        "license": image.get("copyright_license", ""),
        "attribution": image.get("attribution", ""),
        "url": url,
        "sha256": hashlib.sha256(get(url)).hexdigest(),
    }


def collection_rows(collection: int, n: int) -> list[dict[str, str]]:
    """Up to ``n`` images of one collection, one per lesion, allowed licences only. Pages are
    followed only over https on the API's own host; every image must be a clinical photo."""
    host = urllib.parse.urlsplit(IMAGES).netloc
    url: str | None = IMAGES + urllib.parse.urlencode({"collections": collection, "limit": 100})
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    while url and len(rows) < n:
        page = json.loads(get(url))
        for image in page.get("results", []):
            kind = ((image.get("metadata") or {}).get("acquisition") or {}).get("image_type")
            if not kind or "dermoscopic" in str(kind):
                raise SystemExit(
                    f"{image.get('isic_id')}: image_type {kind!r}, not a clinical photo"
                )
            lesion = image_lesion(image) or image["isic_id"]
            file_url = ((image.get("files") or {}).get("full") or {}).get("url")
            if lesion in seen or not file_url:
                continue
            if image.get("copyright_license") not in ALLOWED_LICENSES:
                continue
            seen.add(lesion)
            rows.append(row(image, lesion, file_url))
            if len(rows) == n:
                break
        url = page.get("next")
        if url:
            parts = urllib.parse.urlsplit(str(url))
            if parts.scheme != "https" or parts.netloc != host:
                raise SystemExit(f"refusing to follow a pagination link off {host}")
    return rows


def write(rows: list[dict[str, str]], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--collection", type=int, help="take images from this ISIC collection")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.collection is not None:
        write(collection_rows(args.collection, args.n), args.out)
        return
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
        rows.append(row(image, lesion, url))
    write(rows, args.out)


if __name__ == "__main__":
    main()
