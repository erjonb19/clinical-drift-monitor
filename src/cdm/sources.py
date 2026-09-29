"""ISIC Archive sources for Phase 1: API records, sites, diagnosis mapping, selection, landing.

Standard library and pandas only, so Databricks job tasks and pipelines can import it without
torch. Network access goes through an injectable ``fetch`` or ``opener`` so tests never touch
the network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdm.splits import DataError

API = "https://api.isic-archive.com/api/v2/images/search/"

Record = dict[str, Any]
Row = dict[str, Any]
Fetch = Callable[[str], dict[str, Any]]
Opener = Callable[[str], bytes]


def http_json(url: str, attempts: int = 4) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data: dict[str, Any] = json.load(response)
                return data
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def http_bytes(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                body: bytes = response.read()
                return body
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def collection_records(collection: int, fetch: Fetch = http_json) -> list[Record]:
    """Every public image record in one ISIC collection, following the API's pagination."""
    url: str | None = API + "?" + urllib.parse.urlencode({"collections": collection, "limit": 100})
    records: list[Record] = []
    while url:
        page = fetch(url)
        records.extend(page["results"])
        url = page.get("next")
    return records


# ISIC's diagnosis hierarchy mapped onto HAM10000's seven classes. The rules reproduce every
# HAM10000 label from ISIC's own labels for those images (checked by a pipeline gate).
# Anything else (scars, warts, collisions, metastases, unlabelled images) maps to None.
_BY_DIAGNOSIS_3 = {
    "Nevus": "nv",
    "Melanoma, NOS": "mel",
    "Melanoma in situ": "mel",
    "Melanoma Invasive": "mel",
    "Basal cell carcinoma": "bcc",
    "Solar or actinic keratosis": "akiec",
    "Squamous cell carcinoma in situ": "akiec",
    # ISIC labels HAM10000's intraepithelial carcinomas (akiec) as "SCC, NOS".
    "Squamous cell carcinoma, NOS": "akiec",
    "Pigmented benign keratosis": "bkl",
    "Seborrheic keratosis": "bkl",
    "Solar lentigo": "bkl",
    "Lichen planus like keratosis": "bkl",
    "Dermatofibroma": "df",
}
_VASCULAR_DIAGNOSIS_2 = {"Benign soft tissue proliferations - Vascular", "Hemorrhagic lesions"}


def map_diagnosis(diagnosis_2: str | None, diagnosis_3: str | None) -> str | None:
    if diagnosis_2 in _VASCULAR_DIAGNOSIS_2:
        return "vasc"
    return _BY_DIAGNOSIS_3.get(diagnosis_3) if diagnosis_3 else None


def flatten(record: Record, source: str, collection: int) -> Row:
    """One row per image with the fields the lakehouse uses. Missing metadata becomes None."""
    meta = record.get("metadata", {})
    clinical, acquisition = meta.get("clinical", {}), meta.get("acquisition", {})
    return {
        "isic_id": record["isic_id"],
        "source": source,
        "collection": collection,
        "license": record.get("copyright_license"),
        "attribution": record.get("attribution"),
        "image_type": acquisition.get("image_type"),
        "pixels_x": acquisition.get("pixels_x"),
        "pixels_y": acquisition.get("pixels_y"),
        "isic_lesion_id": clinical.get("lesion_id"),
        "patient_id": clinical.get("patient_id"),
        "fitzpatrick_skin_type": clinical.get("fitzpatrick_skin_type"),
        "sex": clinical.get("sex"),
        "age_approx": clinical.get("age_approx"),
        "anatom_site_1": clinical.get("anatom_site_1"),
        "diagnosis_1": clinical.get("diagnosis_1"),
        "diagnosis_2": clinical.get("diagnosis_2"),
        "diagnosis_3": clinical.get("diagnosis_3"),
        "isic_label": map_diagnosis(clinical.get("diagnosis_2"), clinical.get("diagnosis_3")),
        "url": record["files"]["full"]["url"],
    }


@dataclass(frozen=True)
class Cap:
    """Keep at most ``max_images``, taking whole lesions in a seeded random order."""

    max_images: int
    seed: int
    fingerprint: str | None = None
    n_images: int | None = None


@dataclass(frozen=True)
class Site:
    name: str
    collections: tuple[int, ...]
    cap: Cap | None = None


def load_sites(config: dict[str, Any]) -> list[Site]:
    sites = []
    for name, spec in config["sites"].items():
        cap = Cap(**spec["cap"]) if spec.get("cap") else None
        sites.append(Site(name, tuple(spec["collections"]), cap))
    return sites


def fingerprint(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def select_whole_lesions(rows: list[Row], max_images: int, seed: int) -> list[Row]:
    """Shuffle lesions with ``seed`` and add whole lesions until the next would pass the cap.

    Whole lesions keep a later lesion-level split possible. Images without a lesion ID count
    as their own lesion.
    """
    by_lesion: dict[str, list[Row]] = {}
    for row in rows:
        by_lesion.setdefault(row["isic_lesion_id"] or row["isic_id"], []).append(row)
    order = np.random.default_rng(seed).permutation(sorted(by_lesion))
    chosen: list[Row] = []
    for lesion in order:
        if len(chosen) + len(by_lesion[lesion]) > max_images:
            break
        chosen.extend(by_lesion[lesion])
    return chosen


def site_rows(site: Site, fetch: Fetch = http_json) -> list[Row]:
    """All images for a site, de-duplicated across its collections, capped if configured.

    With a cap, the selection must match the fingerprint recorded in the config; a changed
    collection or code raises instead of silently choosing different images.
    """
    rows: dict[str, Row] = {}
    for collection in site.collections:
        for record in collection_records(collection, fetch):
            rows.setdefault(record["isic_id"], flatten(record, site.name, collection))
    selected = sorted(rows.values(), key=lambda r: r["isic_id"])
    if site.cap is not None:
        selected = select_whole_lesions(selected, site.cap.max_images, site.cap.seed)
        got = fingerprint(r["isic_id"] for r in selected)
        if site.cap.fingerprint is not None and got != site.cap.fingerprint:
            raise DataError(
                f"{site.name}: selection fingerprint {got} ({len(selected)} images) does not "
                f"match config {site.cap.fingerprint} ({site.cap.n_images} images)"
            )
    return selected


def ham_rows(ham_meta: pd.DataFrame, collection: int, fetch: Fetch = http_json) -> list[Row]:
    """HAM10000's images from ISIC by image ID, labelled with HAM10000's own metadata.

    ``lesion_id`` and ``label`` come from HAM10000_metadata.csv so the Phase 0 split is
    reproduced exactly; ISIC's lesion ID and mapped label are kept for comparison.
    """
    by_id = ham_meta.set_index("image_id")
    rows = []
    for record in collection_records(collection, fetch):
        if record["isic_id"] in by_id.index:
            row = flatten(record, "ham10000", collection)
            row["lesion_id"] = by_id.loc[record["isic_id"], "lesion_id"]
            row["label"] = by_id.loc[record["isic_id"], "dx"]
            rows.append(row)
    missing = set(by_id.index) - {r["isic_id"] for r in rows}
    if missing:
        raise DataError(f"{len(missing)} HAM10000 images missing from ISIC collection {collection}")
    return sorted(rows, key=lambda r: r["isic_id"])


def land(
    row: Row,
    images_dir: Path,
    tmp_dir: Path,
    opener: Opener = http_bytes,
    known: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Download one image to ``tmp_dir``, then copy it into ``images_dir``.

    Direct writes into a Unity Catalog volume fail on large files, so the file is written
    locally first and copied with ``shutil.copyfile``. The SHA-256 is taken from the
    downloaded bytes, so the volume copy is not read back. A file already in ``images_dir``
    is never downloaded again: its ``known`` manifest row from an earlier run is reused if
    given, otherwise the file is hashed. Reading files back from a volume is slow (about six
    per second), and the pipeline re-hashes every file in bronze and fails the
    ``files_match_manifest`` gate if one changed, so trusting the earlier row is safe.
    Returns the manifest row.
    """
    dest = images_dir / row["source"] / f"{row['isic_id']}.jpg"
    if dest.exists():
        if known is not None and known.get("path") == str(dest):
            return {k: known[k] for k in ("isic_id", "source", "path", "bytes", "sha256")} | {
                "downloaded": False
            }
        data, downloaded = dest.read_bytes(), False
    else:
        data, downloaded = opener(row["url"]), True
        local = tmp_dir / f"{row['isic_id']}.jpg"
        local.write_bytes(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
        local.unlink()
    return {
        "isic_id": row["isic_id"],
        "source": row["source"],
        "path": str(dest),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "downloaded": downloaded,
    }
