"""Phase 1 Lakeflow Declarative Pipeline: bronze to silver, with gates that fail the update.

Deployed by scripts/deploy_databricks.py, which sets these pipeline configuration keys:

    cdm.config          sites.json in the volume
    cdm.images_glob     Auto Loader path of image files (the broken run adds broken/images)
    cdm.metadata_path   folder of ingest metadata JSONL files
    cdm.manifest_path   folder of ingest manifest JSONL files

Gold is built by the ``cdm-gold`` job task after this pipeline, because it needs torch.

Every dataset function stays lazy: Databricks calls these functions while planning an
update, so pandas logic (splits, gates, decoding) runs inside ``applyInPandas`` or a pandas
UDF, never through ``toPandas`` at definition time. The pandas functions are the ones in
``cdm`` that the test suite exercises.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
from pyspark import pipelines as dp
from pyspark.sql import DataFrame, GroupedData, SparkSession, Window
from pyspark.sql import functions as F

from cdm.gates import run_all, with_splits
from cdm.images import DecodeError, decode_resize

_spark = SparkSession.getActiveSession()
assert _spark is not None, "run inside a Databricks pipeline"
spark: SparkSession = _spark

with open(spark.conf.get("cdm.config"), encoding="utf-8") as f:
    CONFIG = json.load(f)
IMAGES_GLOB = spark.conf.get("cdm.images_glob")
METADATA_PATH = spark.conf.get("cdm.metadata_path")
MANIFEST_PATH = spark.conf.get("cdm.manifest_path")

METADATA_COLUMNS = """
    isic_id STRING, source STRING, collection INT, license STRING, attribution STRING,
    image_type STRING, pixels_x INT, pixels_y INT, isic_lesion_id STRING, patient_id STRING,
    fitzpatrick_skin_type STRING, sex STRING, age_approx DOUBLE, anatom_site_1 STRING,
    diagnosis_1 STRING, diagnosis_2 STRING, diagnosis_3 STRING, isic_label STRING, url STRING,
    lesion_id STRING, label STRING, run_id STRING, code_version STRING, ham_dataset STRING
"""
MANIFEST_COLUMNS = """
    isic_id STRING, source STRING, path STRING, bytes BIGINT, sha256 STRING,
    downloaded BOOLEAN, run_id STRING
"""
SILVER_METADATA_COLUMNS = (
    METADATA_COLUMNS
    + ", manifest_sha256 STRING, sha256 STRING, split STRING, client STRING, role STRING"
)
KEY = ["source", "isic_id"]


def _one_group(df: DataFrame) -> GroupedData:
    """Group every row together for a whole-dataset pandas step.

    The group key must be a named column: Spark reads an integer literal in ``groupBy`` as a
    column position, so ``groupBy(F.lit(1))`` fails analysis (the first Databricks run).
    """
    return df.withColumn("_group", F.lit("all")).groupBy("_group")


def _latest(df: DataFrame, order: str) -> DataFrame:
    w = Window.partitionBy(*KEY).orderBy(F.col(order).desc())
    return df.withColumn("_rank", F.row_number().over(w)).filter("_rank = 1").drop("_rank")


def _jsonl(path: str, columns: str) -> DataFrame:
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(columns)
        .load(path)
        .withColumn("_file", F.col("_metadata.file_path"))
    )


# ---- bronze: files and records exactly as the ingest task delivered them -------------------


@dp.table(comment="Image files as delivered, with the SHA-256 computed here.")
def bronze_images() -> DataFrame:
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .load(IMAGES_GLOB)
        .withColumn("isic_id", F.regexp_extract("path", r"([^/]+)\.jpg$", 1))
        .withColumn("source", F.regexp_extract("path", r"/([^/]+)/[^/]+\.jpg$", 1))
        .withColumn("sha256", F.sha2(F.col("content"), 256))
        .withColumn("ingested_at", F.current_timestamp())
    )


@dp.table(comment="Per-image metadata rows from every ingest run.")
def bronze_metadata() -> DataFrame:
    return _jsonl(METADATA_PATH, METADATA_COLUMNS)


@dp.table(comment="Per-file manifest rows (path, bytes, SHA-256) from every ingest run.")
def bronze_manifest() -> DataFrame:
    return _jsonl(MANIFEST_PATH, MANIFEST_COLUMNS)


# ---- silver ---------------------------------------------------------------------------------


@dp.materialized_view(comment="Latest metadata per image with both SHA-256s and the split.")
@dp.expect("labelled", "label IS NOT NULL")
def silver_metadata() -> DataFrame:
    meta = _latest(spark.read.table("bronze_metadata"), "run_id").drop("_file")
    recorded = _latest(spark.read.table("bronze_manifest"), "run_id").select(
        *KEY, F.col("sha256").alias("manifest_sha256")
    )
    landed = _latest(spark.read.table("bronze_images"), "modificationTime").select(*KEY, "sha256")
    joined = meta.join(recorded, KEY, "left").join(landed, KEY, "left")
    # One group: the split is a whole-dataset computation (Phase 0's function).
    return _one_group(joined).applyInPandas(
        lambda pdf: with_splits(pdf.drop(columns="_group"), CONFIG), SILVER_METADATA_COLUMNS
    )


# Stored image settings (Phase 2 rung 1a); without a "silver" section: 224 px PNG, cropped.
SILVER = CONFIG.get("silver", {})


@F.pandas_udf("image BINARY, width INT, height INT, jpeg_quant_mean DOUBLE, error STRING")
def _decode(content: pd.Series) -> pd.DataFrame:
    rows: list[tuple[bytes | None, int | None, int | None, float | None, str | None]] = []
    for data in content:
        try:
            d = decode_resize(
                bytes(data),
                size=int(SILVER.get("shorter_side", 224)),
                center_crop=bool(SILVER.get("center_crop", True)),
                fmt=str(SILVER.get("format", "PNG")),
                quality=int(SILVER.get("quality", 95)),
            )
            rows.append((d.image, d.width, d.height, d.jpeg_quant_mean, None))
        except DecodeError as exc:
            rows.append((None, None, None, None, str(exc)))
    return pd.DataFrame(rows, columns=["image", "width", "height", "jpeg_quant_mean", "error"])


@dp.materialized_view(comment="Every landed file decoded once; errors kept for quarantine.")
def silver_decoded() -> DataFrame:
    files = _latest(spark.read.table("bronze_images"), "modificationTime")
    return files.select(*KEY, "path", "sha256", _decode(F.col("content")).alias("d")).select(
        *KEY, "path", "sha256", "d.*"
    )


@dp.materialized_view(comment="Decoded 224 x 224 images joined to their metadata and split.")
def silver_images() -> DataFrame:
    decoded = spark.read.table("silver_decoded").filter("error IS NULL").drop("error", "sha256")
    return decoded.join(spark.read.table("silver_metadata"), KEY, "inner")


@dp.materialized_view(comment="Files that failed to decode: kept and reported, never dropped.")
def silver_quarantine() -> DataFrame:
    return spark.read.table("silver_decoded").filter("error IS NOT NULL").drop("image")


# ---- gates ----------------------------------------------------------------------------------


def _gates(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([asdict(g) for g in run_all(frame, CONFIG)])


@dp.materialized_view(comment="One row per Phase 1 gate; any violation fails the update.")
@dp.expect_all_or_fail({"gate_passes": "violations = 0"})
def gate_results() -> DataFrame:
    return _one_group(spark.read.table("silver_metadata")).applyInPandas(
        lambda pdf: _gates(pdf.drop(columns="_group")),
        "name STRING, violations BIGINT, detail STRING",
    )
