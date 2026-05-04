"""
Configuration for Nvidia PhysicalAI Autonomous Vehicles dataset ingestion.

Extends the lakehouse config pattern with dataset-specific paths and settings
for ingesting from NFS-mounted HuggingFace Hub zip archives.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from pyspark.sql import SparkSession

from kaist_ingestion.config import (
    CatalogConfig,
    PipelineConfig,
    StorageConfig,
    _env,
)


SNAP_DEFAULT = (
    "/mnt/netai-e2e/nvidia-physicalai-av-subset"
)

# Original full dataset (kept as reference for pipelines that need it)
SNAP_FULL = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)


@dataclass
class NvidiaConfig:
    """Nvidia PhysicalAI dataset-specific configuration."""

    source_path: str = field(
        default_factory=lambda: _env("NVIDIA_SOURCE_PATH", SNAP_DEFAULT)
    )

    # Iceberg namespace names
    namespace_bronze: str = "nvidia_bronze"
    namespace_silver: str = "nvidia_silver"
    namespace_gold: str = "nvidia_gold"

    # Performance tuning
    target_file_size_bytes: int = 134_217_728  # 128 MB
    shuffle_partitions: int = 200

    # Ingestion limits (set to 0 for unlimited)
    max_zip_chunks: int = 0  # 0 = all chunks
    max_clips_per_chunk: int = 0  # 0 = all clips inside each chunk

    # Lidar decode mode: "blob" keeps Draco binary, "decoded" expands to x/y/z
    lidar_mode: str = field(
        default_factory=lambda: _env("NVIDIA_LIDAR_MODE", "blob")
    )

    # Spark driver memory (larger than default for big parquet concat)
    driver_memory: str = field(
        default_factory=lambda: _env("SPARK_DRIVER_MEMORY", "4g")
    )

    # Silver layer mode: "inplace" rewrites parquet files on disk with
    # enrichment columns (clip_id, sensor_name); "view" creates SQL views.
    silver_mode: str = field(
        default_factory=lambda: _env("NVIDIA_SILVER_MODE", "inplace")
    )

    # Gold layer mode: "materialized" (default) or "view" (zero storage)
    gold_mode: str = field(
        default_factory=lambda: _env("NVIDIA_GOLD_MODE", "materialized")
    )

    # AD-Specific Iceberg optimisation defaults
    snapshot_min_to_keep: int = 10
    snapshot_max_age_hours: int = 168
    write_distribution_mode: str = "hash"
    overwrite_existing: bool = True


@dataclass
class NvidiaPipelineConfig:
    """Complete pipeline configuration for the Nvidia dataset."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)
    spark_catalog_name: str = "iceberg"


def build_spark_session(config: NvidiaPipelineConfig, app_name: str = "nvidia-ingestion"):
    """Build a SparkSession with Iceberg, S3/Polaris, and memory tuning."""
    catalog = config.spark_catalog_name
    storage = config.storage
    cat_cfg = config.catalog
    nv = config.nvidia

    # PYSPARK_SUBMIT_ARGS must be set BEFORE SparkSession creation so the
    # JVM launcher picks up --driver-memory (SparkConf is too late for local mode).
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--driver-memory {nv.driver_memory} pyspark-shell"
    )

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", nv.driver_memory)
        .config("spark.sql.defaultCatalog", catalog)
        # Iceberg catalog
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{catalog}.uri", cat_cfg.uri)
        .config(f"spark.sql.catalog.{catalog}.warehouse", cat_cfg.warehouse)
        # ResolvingFileIO: handles both file:// (NFS/FUSE) and s3:// (MinIO) paths
        .config(f"spark.sql.catalog.{catalog}.io-impl", "org.apache.iceberg.io.ResolvingFileIO")
        .config(f"spark.sql.catalog.{catalog}.s3.endpoint", storage.endpoint)
        .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", str(storage.path_style_access).lower())
        .config(f"spark.sql.catalog.{catalog}.s3.access-key-id", storage.access_key)
        .config(f"spark.sql.catalog.{catalog}.s3.secret-access-key", storage.secret_key)
        # OAuth2
        .config(f"spark.sql.catalog.{catalog}.oauth2-server-uri", cat_cfg.oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.credential", cat_cfg.credential)
        .config(f"spark.sql.catalog.{catalog}.scope", cat_cfg.scope)
        .config(f"spark.sql.catalog.{catalog}.oauth2.server-uri", cat_cfg.oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.oauth2.credential", cat_cfg.credential)
        .config(f"spark.sql.catalog.{catalog}.oauth2.scope", cat_cfg.scope)
        # Hadoop S3A
        .config("spark.hadoop.fs.s3a.endpoint", storage.endpoint)
        .config("spark.hadoop.fs.s3a.access.key", storage.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", storage.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", str(storage.path_style_access).lower())
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.region", storage.region)
        # Map s3:// scheme to S3A (Polaris catalog returns s3:// paths)
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Tuning
        .config("spark.sql.shuffle.partitions", str(nv.shuffle_partitions))
        .config("spark.sql.iceberg.write.target-file-size-bytes", str(nv.target_file_size_bytes))
        # Use zstd compression for ~30-40% smaller Parquet files vs default snappy
        .config("spark.sql.parquet.compression.codec", "zstd")
        # Extensions
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    )
    return builder.getOrCreate()


def create_namespaces(spark, config: NvidiaPipelineConfig):
    """Create bronze/silver/gold namespaces for the Nvidia dataset."""
    catalog = config.spark_catalog_name
    for ns in [
        config.nvidia.namespace_bronze,
        config.nvidia.namespace_silver,
        config.nvidia.namespace_gold,
    ]:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{ns}")
        print(f"  Ensured namespace: {catalog}.{ns}")
