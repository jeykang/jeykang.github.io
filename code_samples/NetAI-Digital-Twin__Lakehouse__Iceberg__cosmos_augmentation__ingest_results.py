"""
Ingest Cosmos-generated videos into the lakehouse.

Uploads MP4s to MinIO and writes metadata + lineage into a dedicated
nvidia_cosmos Iceberg namespace.  Never touches nvidia_bronze/silver/gold.
"""

import io
from datetime import datetime, timezone
from typing import List

import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from kaist_ingestion.config import KAISTConfig, apply_ad_table_optimizations

from .config import CosmosPipelineConfig, create_namespace
from .generate import GeneratedVideo


# ── Iceberg schemas ──────────────────────────────────────────────────────────

GENERATED_SCENES_SCHEMA = StructType([
    StructField("clip_id", StringType(), False),
    StructField("variation", StringType(), False),
    StructField("prompt", StringType(), False),
    StructField("model", StringType(), False),
    StructField("seed", IntegerType(), True),
    StructField("video_s3_uri", StringType(), False),
    StructField("generation_time_s", DoubleType(), False),
    StructField("source_split", StringType(), True),
    StructField("created_at", TimestampType(), False),
])

GENERATION_LINEAGE_SCHEMA = StructType([
    StructField("source_clip_id", StringType(), False),
    StructField("source_table", StringType(), False),
    StructField("variation", StringType(), False),
    StructField("generated_video_uri", StringType(), False),
    StructField("model", StringType(), False),
    StructField("created_at", TimestampType(), False),
])


class CosmosResultIngester:
    """Uploads generated videos to MinIO and records metadata in Iceberg."""

    SCENES_TABLE = "generated_scenes"
    LINEAGE_TABLE = "generation_lineage"

    def __init__(self, spark: SparkSession, config: CosmosPipelineConfig):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.ns = config.cosmos.namespace
        self.s3_prefix = config.cosmos.output_s3_prefix
        self.bucket = config.storage.bucket

        create_namespace(spark, config)

        self._s3 = boto3.client(
            "s3",
            endpoint_url=config.storage.endpoint,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key,
            region_name=config.storage.region,
        )

    def _fqn(self, table: str) -> str:
        return f"{self.cat}.{self.ns}.{table}"

    # ── S3 upload ────────────────────────────────────────────────────────

    def upload_video(self, video: GeneratedVideo) -> str:
        """Upload an MP4 to MinIO and return its S3 URI."""
        key = f"{self.s3_prefix}/{video.clip_id}/{video.variation}.mp4"
        self._s3.upload_fileobj(
            io.BytesIO(video.video_bytes),
            self.bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        uri = f"s3://{self.bucket}/{key}"
        return uri

    # ── Metadata table ───────────────────────────────────────────────────

    def write_metadata(self, videos: List[GeneratedVideo]) -> int:
        """Write generation metadata to the generated_scenes Iceberg table.

        Creates the table on first call; appends on subsequent calls.
        Returns the number of rows written.
        """
        if not videos:
            return 0

        now = datetime.now(timezone.utc)
        rows = []
        for v in videos:
            s3_uri = self.upload_video(v)
            rows.append((
                v.clip_id,
                v.variation,
                v.prompt,
                v.model,
                v.seed,
                s3_uri,
                v.generation_time_s,
                v.source_split,
                now,
            ))

        df = self.spark.createDataFrame(rows, schema=GENERATED_SCENES_SCHEMA)
        fqn = self._fqn(self.SCENES_TABLE)

        df.writeTo(fqn).tableProperty(
            "format-version", "2"
        ).partitionedBy(
            "variation"
        ).createOrReplace()

        # Apply AD-style optimizations (sort, metrics, snapshot retention)
        opt_cfg = KAISTConfig(
            write_distribution_mode=self.config.nvidia.write_distribution_mode,
            snapshot_min_to_keep=self.config.nvidia.snapshot_min_to_keep,
            snapshot_max_age_hours=self.config.nvidia.snapshot_max_age_hours,
        )
        apply_ad_table_optimizations(
            self.spark,
            fqn,
            sort_columns=["clip_id"],
            partition_columns=["variation"],
            metrics_columns=["clip_id", "variation", "source_split"],
            config=opt_cfg,
        )

        print(f"  Wrote {len(rows)} rows to {fqn}")
        return len(rows)

    # ── Lineage table ────────────────────────────────────────────────────

    def write_lineage(self, videos: List[GeneratedVideo]) -> int:
        """Write source→synthetic lineage records for traceability.

        Each row links a source clip to a generated video, enabling
        downstream consumers to trace augmented data back to its origin.
        """
        if not videos:
            return 0

        now = datetime.now(timezone.utc)
        gold_ns = self.config.nvidia.namespace_gold
        source_table = f"{self.cat}.{gold_ns}.sensor_fusion_clip"

        rows = []
        for v in videos:
            s3_uri = f"s3://{self.bucket}/{self.s3_prefix}/{v.clip_id}/{v.variation}.mp4"
            rows.append((
                v.clip_id,
                source_table,
                v.variation,
                s3_uri,
                v.model,
                now,
            ))

        df = self.spark.createDataFrame(rows, schema=GENERATION_LINEAGE_SCHEMA)
        fqn = self._fqn(self.LINEAGE_TABLE)

        df.writeTo(fqn).tableProperty(
            "format-version", "2"
        ).createOrReplace()

        print(f"  Wrote {len(rows)} lineage rows to {fqn}")
        return len(rows)
