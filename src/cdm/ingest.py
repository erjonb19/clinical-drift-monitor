"""Phase 1 ingest task: land HAM10000 and site images and metadata in the raw volume.

Runs as a Databricks job task (entry point ``cdm-ingest``) on serverless compute, and works
against any local folder, which is how the tests run it. Under ``--raw`` it writes:

    metadata/<source>/<run_id>.jsonl   one row per image
    images/<source>/<isic_id>.jpg      files as ISIC delivers them
    manifest/<source>/<run_id>.jsonl   path, bytes, SHA-256, whether downloaded this run

Every run writes new metadata and manifest files, so Auto Loader picks them up; silver keeps
the latest row per image. Images already in the volume are never downloaded again.

``--make-broken`` writes a faulted copy of the latest metadata and manifest under
``broken/`` for the "a broken input fails a gate loudly" demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cdm.sources import (
    Fetch,
    Opener,
    Row,
    ham_rows,
    http_bytes,
    http_json,
    land,
    load_sites,
    site_rows,
)
from cdm.splits import DataError

HAM_SOURCE = "ham10000"


def log(message: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%d %H:%M:%S}Z] {message}", flush=True)


def verify_ham_metadata(path: Path, md5: str) -> pd.DataFrame:
    if not path.exists():
        raise DataError(
            f"{path} is missing: upload HAM10000_metadata.csv (see docs/phase1-runbook.md)"
        )
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    if actual != md5:
        raise DataError(f"{path}: MD5 {actual}, expected Dataverse's {md5}")
    return pd.read_csv(path)


def write_jsonl(rows: Iterable[dict[str, Any]], dest: Path, tmp_dir: Path) -> None:
    """Write locally, then copy into place: direct writes into a volume fail on large files."""
    local = tmp_dir / dest.name
    with local.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, dest)
    local.unlink()


def read_latest_jsonl(folder: Path) -> list[dict[str, Any]]:
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise DataError(f"no metadata or manifest files in {folder}")
    return [json.loads(line) for line in files[-1].read_text(encoding="utf-8").splitlines()]


def land_all(
    rows: list[Row], images_dir: Path, tmp_dir: Path, workers: int, opener: Opener = http_bytes
) -> list[dict[str, Any]]:
    """Land every image; raise after the batch if any failed, listing them, so a partial run
    is never reported as complete."""

    def one(row: Row) -> dict[str, Any] | str:
        try:
            return land(row, images_dir, tmp_dir, opener)
        except OSError as exc:
            return f"{row['isic_id']}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, rows))
    failures = [r for r in results if isinstance(r, str)]
    if failures:
        raise DataError(f"{len(failures)} images failed to land, e.g. {failures[:3]}")
    return [r for r in results if not isinstance(r, str)]


def source_rows(
    config: dict[str, Any], raw: Path, fetch: Fetch = http_json
) -> dict[str, list[Row]]:
    ham_cfg = config["ham10000"]
    meta = verify_ham_metadata(raw / "uploads" / ham_cfg["metadata_file"], ham_cfg["metadata_md5"])
    rows = {HAM_SOURCE: ham_rows(meta, ham_cfg["collection"], fetch)}
    for site in load_sites(config):
        site_list = site_rows(site, fetch)
        for row in site_list:
            row["lesion_id"] = row["isic_lesion_id"] or row["isic_id"]
            row["label"] = row["isic_label"]
        rows[site.name] = site_list
    return rows


def ingest(
    config: dict[str, Any],
    raw: Path,
    tmp_dir: Path,
    workers: int,
    code_version: str,
    fetch: Fetch = http_json,
    opener: Opener = http_bytes,
) -> dict[str, int]:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for source, rows in source_rows(config, raw, fetch).items():
        for row in rows:
            row["run_id"], row["code_version"] = run_id, code_version
        log(f"{source}: {len(rows)} images, landing")
        manifest = land_all(rows, raw / "images", tmp_dir, workers, opener)
        write_jsonl(rows, raw / "metadata" / source / f"{run_id}.jsonl", tmp_dir)
        write_jsonl(
            ({**m, "run_id": run_id} for m in manifest),
            raw / "manifest" / source / f"{run_id}.jsonl",
            tmp_dir,
        )
        downloaded = sum(m["downloaded"] for m in manifest)
        log(f"{source}: landed {len(manifest)} ({downloaded} downloaded this run)")
        counts[source] = len(manifest)
    return counts


CORRUPT_ID = "ISIC_BROKEN_0000001"


def make_broken(raw: Path, tmp_dir: Path, sources: Iterable[str]) -> list[str]:
    """Copy the latest metadata and manifest into ``raw/broken`` with four faults.

    Returns the gates the broken run must fail. The corrupt JPEG must be quarantined without
    failing any gate.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    broken = raw / "broken"
    meta = {s: read_latest_jsonl(raw / "metadata" / s) for s in sources}
    manifest = {s: read_latest_jsonl(raw / "manifest" / s) for s in sources}
    ham, site = meta[HAM_SOURCE], meta["barcelona"]

    # 1. Move one HAM10000 image to another lesion: the split no longer matches Phase 0.
    ham[0]["lesion_id"] = next(r["lesion_id"] for r in ham if r["lesion_id"] != ham[0]["lesion_id"])
    # 2. Drop one HAM10000 label.
    ham[1]["label"] = None
    # 3. Slip one HAM10000 image into Barcelona, file and all.
    slipped = dict(ham[2], source="barcelona")
    site.append(slipped)
    src = raw / "images" / HAM_SOURCE / f"{slipped['isic_id']}.jpg"
    dst = broken / "images" / "barcelona" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    data = dst.read_bytes()
    manifest["barcelona"].append(
        {"isic_id": slipped["isic_id"], "source": "barcelona", "path": str(dst),
         "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "downloaded": False}
    )  # fmt: skip
    # 4. A corrupt JPEG in Barcelona: quarantined, not a gate failure.
    corrupt = broken / "images" / "barcelona" / f"{CORRUPT_ID}.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff\xe0 this is not a complete JPEG")
    site.append(dict(site[0], isic_id=CORRUPT_ID, lesion_id=CORRUPT_ID))
    manifest["barcelona"].append(
        {"isic_id": CORRUPT_ID, "source": "barcelona", "path": str(corrupt),
         "bytes": corrupt.stat().st_size,
         "sha256": hashlib.sha256(corrupt.read_bytes()).hexdigest(), "downloaded": False}
    )  # fmt: skip

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for s in sources:
        write_jsonl(meta[s], broken / "metadata" / s / f"{run_id}.jsonl", tmp_dir)
        write_jsonl(manifest[s], broken / "manifest" / s / f"{run_id}.jsonl", tmp_dir)
    return [
        "split_fingerprint_matches_phase0",
        "no_missing_labels",
        "class_counts_as_expected",
        "diagnosis_mapping_matches_ham10000",
        "no_ham10000_image_in_a_site",
        "no_duplicate_files",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="raw volume root")
    parser.add_argument("--tmp", type=Path, default=Path("/tmp/cdm-ingest"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--code-version", default="unknown")
    parser.add_argument("--make-broken", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.make_broken:
        sources = [HAM_SOURCE, *config["sites"]]
        expected = make_broken(args.raw, args.tmp, sources)
        log(f"wrote {args.raw / 'broken'}; the broken run must fail: {', '.join(expected)}")
        return 0
    counts = ingest(config, args.raw, args.tmp, args.workers, args.code_version)
    log(f"done: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
