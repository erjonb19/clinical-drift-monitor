# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: the monthly drift job (serverless CPU)
# MAGIC Under `results/phase3/monthly/spec.json`. Each run handles the next month (or `count`
# MAGIC months, for the replay backfill): it scores the month's images with the evaluated serving
# MAGIC bundle on CPU, tests the ensemble OOD scores against the validation baseline (KS and PSI,
# MAGIC overall and per Fitzpatrick type), reports the previous month's accuracy now that its labels
# MAGIC have arrived, logs the ISIC Archive's public image count, and appends to
# MAGIC `gold_drift_monthly` and `gold_prediction_audit`. A month already in the table is refused.
# MAGIC Months 1 to 12 replay real shifts in a fixed order; later months log the live check only.
# MAGIC Every table write is a Spark query over the tables its rows come from, so Unity Catalog lineage
# MAGIC reaches the three tables (`results/phase3/lineage/spark_writes_spec.json`).

# COMMAND ----------

for name, default in {
    "code_version": "",
    "wheel": "",
    "wheel_sha256": "",
    "repo": "",
    "files_sha256": "{}",
    "bundle": "",
    "out": "/Volumes/workspace/cdm/raw/results/phase3/monthly",
    "count": "1",
    "job_run_id": "",
}.items():
    dbutils.widgets.text(name, default)

# COMMAND ----------

import hashlib
import io
import json
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyspark.sql import functions as F

from cdm.data import eval_transform
from cdm.monitor import by_skin_type, delayed_accuracy, draw_schedule, drift_test
from cdm.serving import Service, mahalanobis
from cdm.splits import CLASSES
from cdm.train import features_and_logits

CODE = dbutils.widgets.get("code_version")
REPO = Path(dbutils.widgets.get("repo"))
OUT = Path(dbutils.widgets.get("out"))
BUNDLE = Path(dbutils.widgets.get("bundle"))
COUNT = int(dbutils.widgets.get("count"))
JOB_RUN = dbutils.widgets.get("job_run_id") or "manual"
MONTHLY, AUDIT, BASELINE = (
    f"workspace.cdm.{t}"
    for t in ("gold_drift_monthly", "gold_prediction_audit", "gold_drift_baseline")
)
INDEX = {c: i for i, c in enumerate(CLASSES)}

WHEEL_SHA = hashlib.sha256(Path(dbutils.widgets.get("wheel")).read_bytes()).hexdigest()
assert dbutils.widgets.get("wheel_sha256") == WHEEL_SHA, f"wheel sha256 {WHEEL_SHA}"
expected = json.loads(dbutils.widgets.get("files_sha256"))
for rel, digest in expected.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert got == digest, f"{rel}: sha256 {got}, expected {digest}"
spec = json.loads((REPO / "results/phase3/monthly/spec.json").read_text(encoding="utf-8"))
evaluated = json.loads((REPO / "results/phase3/serving/results.json").read_text(encoding="utf-8"))
manifest_sha = hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest()
assert manifest_sha == evaluated["bundle"]["manifest_sha256"], "not the evaluated bundle"
VERSION = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))["version"]
LAST_REPLAY = max(int(m) for m in spec["replay"]["schedule"])
transform = eval_transform(224)


@torch.no_grad()
def score(images: list[bytes]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ensemble probabilities and OOD scores, as the API computes them, 32 images at a time."""
    probs, oods, shas = [], [], []
    for start in range(0, len(images), 32):
        chunk = images[start : start + 32]
        x = torch.stack([transform(Image.open(io.BytesIO(b)).convert("RGB")) for b in chunk])
        p, s = [], []
        for model, arrays in zip(service.models, service.detectors, strict=True):
            f, z = features_and_logits(model, x)
            p.append(torch.softmax(z, dim=1).double().numpy())
            s.append(mahalanobis(arrays, f.double().numpy()))
        probs.append(np.mean(p, axis=0))
        oods.append(np.mean(s, axis=0))
        shas += [hashlib.sha256(b).hexdigest() for b in chunk]
    return np.concatenate(probs), np.concatenate(oods), shas


def table_exists(name: str) -> bool:
    return bool(spark.catalog.tableExists(name))


def append_through_silver(rows: pd.DataFrame, table: str) -> None:
    """Append rows keyed by isic_id, joined to silver_images so lineage records the source."""
    joined = spark.createDataFrame(rows).join(
        spark.read.table("workspace.cdm.silver_images").select("isic_id"), "isic_id", "inner"
    )
    if joined.count() != len(rows):
        raise RuntimeError(f"{table}: an ISIC ID is missing from or repeated in silver_images")
    joined.select(*rows.columns).write.mode("append").saveAsTable(table)


# COMMAND ----------

START = (
    (max(spark.table(MONTHLY).select("month").toPandas()["month"]) + 1)
    if table_exists(MONTHLY)
    else 1
)
REPLAYING = START <= LAST_REPLAY
# After the replay a run logs the live check and the last delayed accuracy only: no model.
service = Service.load(BUNDLE) if REPLAYING else None

# The validation baseline, once per bundle, checked for size.
if REPLAYING and (
    not table_exists(BASELINE)
    or spark.table(BASELINE).where(F.col("bundle_version") == service.version).count() == 0
):
    val = (
        spark.read.table("workspace.cdm.silver_images")
        .where("role = 'client' AND split = 'val' AND label IS NOT NULL")
        .select("isic_id", "image")
        .orderBy("source", "isic_id")
        .toPandas()
    )
    _, val_ood, _ = score(list(val["image"]))
    append_through_silver(
        pd.DataFrame(
            {"bundle_version": service.version, "isic_id": val["isic_id"], "ood_score": val_ood}
        ),
        BASELINE,
    )
baseline = (
    spark.table(BASELINE)
    .where(F.col("bundle_version") == VERSION)
    .toPandas()["ood_score"]
    .to_numpy()
    if REPLAYING
    else np.array([])
)

# The replay schedule, fixed by the spec and the seed.
meta = (
    spark.read.table("workspace.cdm.silver_images")
    .alias("s")
    .join(
        spark.read.table("workspace.cdm.silver_metadata").alias("m"), ["source", "isic_id"], "left"
    )
    .select(
        "s.source",
        "s.isic_id",
        "s.role",
        "s.split",
        "s.label",
        F.col("m.fitzpatrick_skin_type").alias("skin_type"),
    )
    .orderBy("source", "isic_id")
    .toPandas()
)
pools = {
    "client_test": list(
        meta[(meta["role"] == "client") & (meta["split"] == "test") & meta["label"].notna()][
            "isic_id"
        ]
    ),
    "buenos_aires": list(meta[meta["source"] == "buenos_aires"]["isic_id"]),
    "pad_ufes": list(meta[meta["source"] == "pad_ufes"]["isic_id"]),
}
if meta["isic_id"].duplicated().any():
    raise RuntimeError("an ISIC ID appears twice in silver: labels and skin types would be wrong")
SIZES = {"baseline_images": int(len(baseline)), **{f"pool_{k}": len(v) for k, v in pools.items()}}
months = draw_schedule(pools, spec["replay"]["schedule"])
meta = meta.set_index("isic_id")

# COMMAND ----------


def isic_public_count() -> dict:
    try:
        with urllib.request.urlopen(
            "https://api.isic-archive.com/api/v2/images/?limit=1", timeout=30
        ) as r:
            return {"count": int(json.load(r)["count"])}
    except Exception as err:  # noqa: BLE001 - the live check must not stop the run
        return {"count": None, "error": type(err).__name__}


def run_month(month: int) -> dict:
    done = (
        set(spark.table(MONTHLY).select("month").toPandas()["month"])
        if table_exists(MONTHLY)
        else set()
    )
    if month in done:
        raise RuntimeError(f"month {month} is already in {MONTHLY}")
    live = isic_public_count()
    previous = spark.table(MONTHLY).orderBy(F.desc("month")).first() if done else None
    if previous is not None and live["count"] is not None and previous["live_count"] is not None:
        live["new_since_last_run"] = live["count"] - int(previous["live_count"])
    row = {"month": month, "run_utc": datetime.now(UTC).isoformat(timespec="seconds"), "job_run": JOB_RUN,
           "code_version": CODE, "bundle_version": VERSION, "live": live, "sizes": SIZES}  # fmt: skip
    # The previous month's accuracy: its labels have now "arrived".
    prev = (
        spark.table(AUDIT).where(F.col("month") == month - 1).toPandas()
        if table_exists(AUDIT) and (month - 1) in done
        else pd.DataFrame()
    )
    if len(prev):  # only a month that scored images has accuracy to report
        labels = (
            prev["isic_id"]
            .map(
                lambda i: (
                    INDEX.get(meta.loc[i, "label"], -1)
                    if isinstance(meta.loc[i, "label"], str)
                    else -1
                )
            )
            .to_numpy()
        )
        probs = prev[[f"p_{c}" for c in CLASSES]].to_numpy(np.float64)
        row["delayed_accuracy_of_month"] = {
            "month": month - 1,
            **delayed_accuracy(labels.astype(np.int64), probs),
        }
    batch = months.get(month) if REPLAYING else None
    if batch is None:
        row["replay"] = "none: the replay ended at month 12; live check only"
    else:
        ids = [i for _, i in batch]
        blobs = (
            spark.read.table("workspace.cdm.silver_images")
            .where(F.col("isic_id").isin(ids))
            .select("isic_id", "image")
            .toPandas()
            .set_index("isic_id")
            .loc[ids, "image"]
        )
        probs, ood, shas = score(list(blobs))
        row["by_source"] = {k: sum(1 for p, _ in batch if p == k) for k in pools}
        row["drift"] = drift_test(baseline, ood)
        row["by_skin_type"] = by_skin_type(baseline, ood, [meta.loc[i, "skin_type"] for i in ids])
        row["abstain_share"] = float(np.mean(ood > service.abstain_threshold))
        now = datetime.now(UTC).isoformat(timespec="seconds")
        audit = pd.DataFrame({
            "request_id": [str(uuid.uuid4()) for _ in ids], "month": month, "isic_id": ids,
            "model_version": service.version, "bundle_manifest_sha256": manifest_sha, "input_sha256": shas,
            "label": [CLASSES[k] for k in probs.argmax(1)],
            **{f"p_{c}": probs[:, k] for k, c in enumerate(CLASSES)},
            "ood_score": ood, "abstain": ood > service.abstain_threshold,
            "refer_melanoma": probs[:, CLASSES.index("mel")] >= service.refer_threshold, "scored_utc": now,
        })  # fmt: skip
        append_through_silver(audit, AUDIT)
    # The table's alert column: the overall alert OR any skin-type alert (the README says so).
    alert = bool(row.get("drift", {}).get("alert")) or any(
        v.get("alert") for v in row.get("by_skin_type", {}).values()
    )
    record = {
        "month": month,
        "run_utc": row["run_utc"],
        "alert": alert,
        "live_count": live["count"],
        "summary": json.dumps(row),
    }
    # The row's two counts come from the tables it summarises, so lineage reaches it from both.
    audit_rows = (
        spark.table(AUDIT).where(F.col("month") == month).agg(F.count(F.lit(1)).alias("audit_rows"))
        if table_exists(AUDIT)
        else spark.range(1).select(F.lit(0).cast("bigint").alias("audit_rows"))
    )
    baseline_rows = (
        spark.table(BASELINE)
        .where(F.col("bundle_version") == VERSION)
        .agg(F.count(F.lit(1)).alias("baseline_rows"))
    )
    out = (
        spark.createDataFrame(pd.DataFrame([record])).crossJoin(audit_rows).crossJoin(baseline_rows)
    )
    got = out.select("audit_rows", "baseline_rows").first()
    # After the replay no baseline is loaded (SIZES says 0), so only the audit count is checked.
    if got["audit_rows"] != sum(row.get("by_source", {}).values()) or (
        REPLAYING and got["baseline_rows"] != SIZES["baseline_images"]
    ):
        raise RuntimeError(f"month {month}: table counts {got} do not match the run's counts")
    out.write.mode("append").saveAsTable(MONTHLY)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"month_{month:02d}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


# COMMAND ----------

for m in range(START, START + COUNT):
    r = run_month(int(m))
    print(m, json.dumps({k: r.get(k) for k in ("by_source", "drift", "abstain_share")}), flush=True)
