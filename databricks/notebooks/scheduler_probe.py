# Databricks notebook source
# MAGIC %md
# MAGIC # Scheduler probe (backlog item 1; temporary, delete after 10 November 2026)
# MAGIC The monthly drift job's schedule was active and correct on 1 November 2026, but no run
# MAGIC started. This job has the same kind of schedule (quartz, UTC, day-of-month list) and fires
# MAGIC at 06:00 UTC on 8, 9 and 10 November, to show whether the scheduler starts runs at all.
# MAGIC It writes one small JSON file per run to its own folder and touches no table.
# MAGIC
# MAGIC It also tries to download one public ISIC image (owner's OK, 2026-11-06), to learn whether
# MAGIC Free Edition's outbound internet reaches the host that serves ISIC image files. The image
# MAGIC is hashed and discarded; a failed download is recorded and does not fail the run, so it
# MAGIC cannot be mistaken for a scheduler failure. Python standard library only.

# COMMAND ----------

for name, default in {
    "out": "/Volumes/workspace/cdm/raw/results/scheduler_probe",
    "image_id": "ISIC_6589782",
    "job_run_id": "",
    "trigger_type": "",
    "trigger_time": "",
}.items():
    dbutils.widgets.text(name, default)  # type: ignore[name-defined]  # noqa: F821

# COMMAND ----------

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

W = dbutils.widgets  # type: ignore[name-defined]  # noqa: F821
OUT = Path(W.get("out"))
API = "https://api.isic-archive.com/api/v2/images/{}/"
MAX_BYTES = 20 * 1024 * 1024


def fetch(url: str, limit: int) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"more than {limit} bytes")
    return data


def download_test(image_id: str) -> dict[str, Any]:
    """Metadata from the ISIC API, then the image file itself; every step recorded."""
    out: dict[str, Any] = {"image_id": image_id, "api_host": "api.isic-archive.com"}
    try:
        meta = json.loads(fetch(API.format(image_id), 1024 * 1024))
        out["metadata"] = "ok"
        out["license"] = meta.get("copyright_license")
        out["attribution"] = meta.get("attribution")
        url = meta["files"]["full"]["url"]
        out["file_host"] = urllib.parse.urlparse(url).hostname
    except Exception as err:  # noqa: BLE001 - recorded, never raised
        out["metadata"] = f"failed: {type(err).__name__}: {str(err)[:200]}"
        return out
    try:
        blob = fetch(url, MAX_BYTES)
        out["file"] = "ok"
        out["bytes"] = len(blob)
        out["sha256"] = hashlib.sha256(blob).hexdigest()
    except Exception as err:  # noqa: BLE001 - recorded, never raised
        out["file"] = f"failed: {type(err).__name__}: {str(err)[:200]}"
    return out


# COMMAND ----------

record = {
    "what": "scheduler probe (backlog item 1), temporary",
    "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "job_run_id": W.get("job_run_id"),
    "trigger_type": W.get("trigger_type"),
    "trigger_time": W.get("trigger_time"),
    "download_test": download_test(W.get("image_id")),
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"probe_{record['job_run_id'] or 'manual'}.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(json.dumps(record))
