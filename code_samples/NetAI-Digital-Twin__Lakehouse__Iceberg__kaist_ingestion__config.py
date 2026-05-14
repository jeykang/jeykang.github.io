"""
Configuration module for KAIST ingestion pipeline.

Handles environment variables, Spark session configuration, and storage backend settings.
Supports both MinIO (development) and Ceph (production) S3-compatible backends.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from pyspark.sql import SparkSession


def _env(name: str, default: Optional[str] = None) -> str:
    """Get environment variable or raise if missing and no default."""
    value = os.environ.get(name)
    if value is None or value == "":
        if default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    return value


@dataclass
class StorageConfig:
    """S3-compatible storage configuration."""
    
    endpoint: str = field(default_factory=lambda: _env("AWS_S3_ENDPOINT", "http://minio:9000"))
    access_key: str = field(default_factory=lambda: _env("AWS_ACCESS_KEY_ID", "minioadmin"))
    secret_key: str = field(default_factory=lambda: _env("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "spark1"))
    path_style_access: bool = True  # Required for MinIO and Ceph RGW


@dataclass
class CatalogConfig:
    """Iceberg REST Catalog (Polaris) configuration."""
    
    uri: str = field(default_factory=lambda: _env("POLARIS_URI", "http://polaris:8181/api/catalog"))
    warehouse: str = field(default_factory=lambda: _env("POLARIS_CATALOG_NAME", "lakehouse_catalog"))
    credential: str = field(default_factory=lambda: _env("POLARIS_CREDENTIAL", "root:s3cr3t"))
    scope: str = field(default_factory=lambda: _env("POLARIS_SCOPE", "PRINCIPAL_ROLE:ALL"))
    
    @property
    def oauth2_server_uri(self) -> str:
        return os.environ.get("POLARIS_OAUTH2_SERVER_URI", f"{self.uri}/v1/oauth/tokens")


@dataclass
class KAISTConfig:
    """KAIST dataset-specific configuration."""
    
    # Source data location (mounted volume or S3 path)
    source_path: str = field(default_factory=lambda: _env("KAIST_SOURCE_PATH", "/user_data/kaist-simulated"))
    
    # Iceberg namespace names
    namespace_bronze: str = "kaist_bronze"
    namespace_silver: str = "kaist_silver"
    namespace_gold: str = "kaist_gold"
    
    # Performance tuning
    target_file_size_bytes: int = 134_217_728  # 128 MB
    shuffle_partitions: int = 200
    
    # Ingestion behavior
    overwrite_existing: bool = True
    validate_on_ingest: bool = True
    
    # AD-Specific Iceberg Optimization Defaults
    # Applied to all tables to optimize for autonomous driving access patterns
    snapshot_min_to_keep: int = 10  # Training reproducibility via time travel
    snapshot_max_age_hours: int = 168  # 7 days retention
    write_distribution_mode: str = "hash"  # Distribute by partition key


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    storage: StorageConfig = field(default_factory=StorageConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    kaist: KAISTConfig = field(default_factory=KAISTConfig)
    
    # Spark catalog alias (used in SQL queries)
    spark_catalog_name: str = "iceberg"


def build_spark_session(config: PipelineConfig, app_name: str = "kaist-ingestion") -> SparkSession:
    """
    Build a SparkSession configured for Iceberg with Polaris catalog and S3 storage.
    
    This follows the pattern established in ingest_nuscenes_mini.py but is
    parameterized for flexibility.
    
    Args:
        config: Pipeline configuration
        app_name: Spark application name
        
    Returns:
        Configured SparkSession
    """
    catalog = config.spark_catalog_name
    storage = config.storage
    cat_cfg = config.catalog
    
    builder = (
        SparkSession.builder.appName(app_name)
        # Set default catalog
        .config("spark.sql.defaultCatalog", catalog)
        
        # Iceberg Spark catalog configuration
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{catalog}.uri", cat_cfg.uri)
        .config(f"spark.sql.catalog.{catalog}.warehouse", cat_cfg.warehouse)
        
        # S3 FileIO configuration
        .config(f"spark.sql.catalog.{catalog}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"spark.sql.catalog.{catalog}.s3.endpoint", storage.endpoint)
        .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", str(storage.path_style_access).lower())
        .config(f"spark.sql.catalog.{catalog}.s3.access-key-id", storage.access_key)
        .config(f"spark.sql.catalog.{catalog}.s3.secret-access-key", storage.secret_key)
        
        # OAuth2 for Polaris authentication
        .config(f"spark.sql.catalog.{catalog}.oauth2-server-uri", cat_cfg.oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.credential", cat_cfg.credential)
        .config(f"spark.sql.catalog.{catalog}.scope", cat_cfg.scope)
        
        # Alternate OAuth2 key spellings for version compatibility
        .config(f"spark.sql.catalog.{catalog}.oauth2.server-uri", cat_cfg.oauth2_server_uri)
        .config(f"spark.sql.catalog.{catalog}.oauth2.credential", cat_cfg.credential)
        .config(f"spark.sql.catalog.{catalog}.oauth2.scope", cat_cfg.scope)
        
        # Hadoop S3A configuration (for reading source files from S3 if needed)
        .config("spark.hadoop.fs.s3a.endpoint", storage.endpoint)
        .config("spark.hadoop.fs.s3a.access.key", storage.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", storage.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", str(storage.path_style_access).lower())
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.region", storage.region)
        
        # Performance tuning
        .config("spark.sql.shuffle.partitions", str(config.kaist.shuffle_partitions))
        .config("spark.sql.iceberg.write.target-file-size-bytes", 
                str(config.kaist.target_file_size_bytes))
        
        # Iceberg extensions
        .config("spark.sql.extensions", 
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    )
    
    return builder.getOrCreate()


def create_namespaces(spark: SparkSession, config: PipelineConfig) -> None:
    """Create the bronze, silver, and gold namespaces if they don't exist."""
    catalog = config.spark_catalog_name
    
    for namespace in [
        config.kaist.namespace_bronze,
        config.kaist.namespace_silver,
        config.kaist.namespace_gold,
    ]:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")
        print(f"Ensured namespace exists: {catalog}.{namespace}")


def apply_ad_table_optimizations(
    spark: SparkSession,
    full_table: str,
    sort_columns: Optional[List[str]] = None,
    partition_columns: Optional[List[str]] = None,
    metrics_columns: Optional[List[str]] = None,
    config: Optional[KAISTConfig] = None,
) -> None:
    """
    Apply AD-specific Iceberg table optimizations.

    These optimizations are fundamental to the AD lakehouse design and ensure
    that all ingested autonomous driving data is automatically optimized for
    the dominant access patterns in AD/ML workloads:

    1. Persisted sort orders — Iceberg-native write ordering ensures temporal/
       sequential locality within partitions (critical for frame replay,
       sensor stream reconstruction, and sequential training data loading).

    2. Column-level metrics — Full min/max statistics on sensor timestamps,
       frame indices, and sensor names enable Iceberg's predicate pushdown
       to skip irrelevant data files during planning (before any I/O).

    3. Snapshot retention — Retains table history for training dataset
       reproducibility via Iceberg time travel (VERSION AS OF).

    4. Write distribution mode — Hash-distributes writes by partition key
       to minimize small files and maintain partition alignment.

    These properties are persisted in Iceberg table metadata and apply to
    all future writes (appends, overwrites), not just the initial load.

    Args:
        spark: Active SparkSession
        full_table: Fully qualified table name (catalog.namespace.table)
        sort_columns: Columns for Iceberg-native write sort order
        partition_columns: Partition columns (determines distribution strategy)
        metrics_columns: Columns needing full min/max metrics for predicate pushdown
        config: KAIST config (uses defaults if None)
    """
    if config is None:
        config = KAISTConfig()

    # --- Table properties for AD optimization ---
    props = {
        "write.distribution-mode": config.write_distribution_mode,
        "history.expire.min-snapshots-to-keep": str(config.snapshot_min_to_keep),
        "history.expire.max-snapshot-age-ms": str(config.snapshot_max_age_hours * 3600 * 1000),
        "write.metadata.metrics.default": "truncate(16)",
    }

    # Full min/max metrics for AD-critical columns (enables predicate pushdown
    # on sensor timestamps, frame indices, clip boundaries, and sensor names)
    if metrics_columns:
        for col_name in metrics_columns:
            props[f"write.metadata.metrics.column.{col_name}"] = "full"

    props_sql = ", ".join(f"'{k}' = '{v}'" for k, v in props.items())
    spark.sql(f"ALTER TABLE {full_table} SET TBLPROPERTIES ({props_sql})")

    # --- Iceberg-native write sort order ---
    # Persisted in table metadata; all future writes automatically maintain
    # temporal/sequential ordering within partitions
    if sort_columns:
        order_clause = ", ".join(sort_columns)
        if partition_columns:
            # Distribute by partition key, sort locally within each partition
            spark.sql(
                f"ALTER TABLE {full_table} WRITE "
                f"DISTRIBUTED BY PARTITION LOCALLY ORDERED BY {order_clause}"
            )
        else:
            spark.sql(f"ALTER TABLE {full_table} WRITE ORDERED BY {order_clause}")

    print(f"  [AD-OPT] Applied AD optimizations to {full_table}")
