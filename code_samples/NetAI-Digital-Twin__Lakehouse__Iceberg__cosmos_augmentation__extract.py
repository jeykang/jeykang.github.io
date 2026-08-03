"""
Clip extraction from Gold tables for Cosmos augmentation.

Reads clip metadata from the existing nvidia_gold namespace.
This module is read-only — it never writes to nvidia_* tables.
"""

from dataclasses import dataclass
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, concat_ws, col, lit

from .config import CosmosPipelineConfig


@dataclass
class ClipRecord:
    """Metadata for a single clip eligible for Cosmos generation."""

    clip_id: str
    split: str                       # train / val / test
    country: str = ""
    month: int = 0
    hour_of_day: int = 0
    platform_class: str = ""
    source_table: str = ""


def extract_clips(
    spark: SparkSession,
    config: CosmosPipelineConfig,
) -> List[ClipRecord]:
    """Query the Gold sensor_fusion_clip table for eligible clips.

    The sensor_fusion_clip table has one row per clip with metadata
    columns (split, country, month, hour_of_day, etc.) but no UUID
    clip_id.  We synthesise a deterministic clip_id from
    chunk + row position for traceability.

    If max_clips is set in the config, only that many clips are returned.
    """
    cat = config.spark_catalog_name
    gold_ns = config.nvidia.namespace_gold
    table = f"{cat}.{gold_ns}.sensor_fusion_clip"

    print(f"  Reading clip index from {table} ...")
    df = spark.table(table).filter("clip_is_valid = true")

    # Build a deterministic clip_id from chunk + monotonic row id
    df = df.withColumn("_row_id", monotonically_increasing_id())
    df = df.withColumn(
        "clip_id",
        concat_ws("-", lit("clip"), col("chunk").cast("string"), col("_row_id").cast("string")),
    )

    # Detect available columns
    columns = set(df.columns)
    select_cols = ["clip_id", "split"]
    for c in ["country", "month", "hour_of_day", "platform_class"]:
        if c in columns:
            select_cols.append(c)

    limit = config.cosmos.max_clips
    if limit > 0:
        df = df.limit(limit)

    rows = df.select(*select_cols).collect()

    records: List[ClipRecord] = []
    for row in rows:
        records.append(
            ClipRecord(
                clip_id=row["clip_id"],
                split=row["split"] or "unknown",
                country=row["country"] if "country" in row.asDict() else "",
                month=row["month"] if "month" in row.asDict() else 0,
                hour_of_day=row["hour_of_day"] if "hour_of_day" in row.asDict() else 0,
                platform_class=row["platform_class"] if "platform_class" in row.asDict() else "",
                source_table=table,
            )
        )

    print(f"  Found {len(records)} eligible clips (limit={limit or 'ALL'})")
    return records
