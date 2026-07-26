"""
Configuration for Cosmos augmentation pipeline.

Extends the lakehouse config pattern with Cosmos NIM API settings.
This module is fully self-contained — the base lakehouse has no
awareness of Cosmos and operates identically without it.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from pyspark.sql import SparkSession

from kaist_ingestion.config import (
    CatalogConfig,
    StorageConfig,
    _env,
)
from nvidia_ingestion.config import NvidiaConfig


# ── Variation prompt templates ──────────────────────────────────────────────
# Each key is a variation name; the value is a prompt suffix appended to the
# base driving-scene prompt.  Add new entries here to extend the pipeline
# without touching any logic.

VARIATION_PROMPTS = {
    "foggy": "dense fog, low visibility, diffused headlights, moisture in the air",
    "rainy": "heavy rain, wet reflective road surface, rain drops on windshield",
    "night": "nighttime, dark sky, street lights, headlight illumination, glare",
    "snowy": "heavy snowfall, snow-covered road, white landscape, reduced visibility",
    "golden_hour": "golden hour sunset lighting, long warm shadows, sun low on horizon",
    "overcast": "overcast sky, flat diffused lighting, grey clouds",
}

BASE_DRIVING_PROMPT = (
    "First person view from a car driving on an urban road, "
    "photorealistic, high detail"
)


# ── API Catalog endpoint patterns ────────────────────────────────────────────
# build.nvidia.com hosted models use integrate.api.nvidia.com with nvapi- keys.
# Model slug is appended to the base URL.  These may change as NVIDIA updates
# their catalog — override via COSMOS_API_CATALOG_URL if needed.

API_CATALOG_BASE = "https://integrate.api.nvidia.com/v1"
API_CATALOG_MODELS = {
    "transfer":    f"{API_CATALOG_BASE}/cosmos/nvidia/cosmos-transfer1-7b",
    "transfer2.5": f"{API_CATALOG_BASE}/cosmos/nvidia/cosmos-transfer2-5-2b",
    "text2world":  f"{API_CATALOG_BASE}/cosmos/nvidia/cosmos-predict1-7b-text2world",
    "video2world": f"{API_CATALOG_BASE}/cosmos/nvidia/cosmos-predict1-7b-video2world",
}


@dataclass
class CosmosConfig:
    """Nvidia Cosmos API configuration.

    Supports two backends:
      - "nim": Self-hosted NIM container (default endpoint http://cosmos:8000).
               Requires GPU host running the NIM Docker image.
      - "api-catalog": NVIDIA build.nvidia.com hosted API.
               Requires an nvapi- API key (set COSMOS_API_KEY or --api-key).
    """

    # Backend: "nim" (self-hosted container) or "api-catalog" (build.nvidia.com)
    backend: str = field(
        default_factory=lambda: _env("COSMOS_BACKEND", "api-catalog")
    )

    # NIM endpoint (used when backend="nim")
    endpoint: str = field(
        default_factory=lambda: _env("COSMOS_ENDPOINT", "http://cosmos:8000")
    )

    # API Catalog key (used when backend="api-catalog")
    # Generate at https://build.nvidia.com — key starts with "nvapi-"
    api_key: str = field(
        default_factory=lambda: _env("COSMOS_API_KEY", "")
    )

    # API Catalog URL override (used when backend="api-catalog")
    # Defaults to the model-specific URL from API_CATALOG_MODELS.
    api_catalog_url: str = field(
        default_factory=lambda: _env("COSMOS_API_CATALOG_URL", "")
    )

    # Model selection: "transfer" | "transfer2.5" | "text2world" | "video2world"
    model: str = field(
        default_factory=lambda: _env("COSMOS_MODEL", "text2world")
    )

    # Iceberg output namespace (isolated from nvidia_* namespaces)
    namespace: str = "nvidia_cosmos"

    # Weather/lighting variations to generate per clip
    variations: List[str] = field(
        default_factory=lambda: _env(
            "COSMOS_VARIATIONS",
            "foggy,rainy,night,snowy,golden_hour",
        ).split(",")
    )

    # Clip processing limits (0 = all eligible clips)
    max_clips: int = 0

    # Generation parameters
    resolution: str = "704"
    guidance_scale: float = 7.0
    seed: Optional[int] = None

    # API reliability
    timeout_seconds: int = 600
    max_retries: int = 3

    # S3 output location for generated MP4s
    output_s3_prefix: str = "cosmos_augmented"

    # Spark tuning
    shuffle_partitions: int = 200
    target_file_size_bytes: int = 134_217_728  # 128 MB

    @property
    def infer_url(self) -> str:
        """Resolve the inference URL based on backend and model."""
        if self.backend == "api-catalog":
            if self.api_catalog_url:
                return self.api_catalog_url
            return API_CATALOG_MODELS.get(self.model, f"{API_CATALOG_BASE}/cosmos/nvidia/{self.model}")
        # NIM: always POST /v1/infer on the container
        return f"{self.endpoint.rstrip('/')}/v1/infer"

    @property
    def health_url(self) -> str:
        """Resolve the health-check URL (NIM only; API Catalog has none)."""
        return f"{self.endpoint.rstrip('/')}/v1/health/ready"


@dataclass
class CosmosPipelineConfig:
    """Complete pipeline configuration for Cosmos augmentation."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    nvidia: NvidiaConfig = field(default_factory=NvidiaConfig)   # for reading Gold
    cosmos: CosmosConfig = field(default_factory=CosmosConfig)
    spark_catalog_name: str = "iceberg"


def build_spark_session(
    config: CosmosPipelineConfig,
    app_name: str = "cosmos-augmentation",
) -> SparkSession:
    """Build a SparkSession configured for Iceberg + S3 (same as nvidia pipeline)."""
    catalog = config.spark_catalog_name
    storage = config.storage
    cat_cfg = config.catalog
    cosmos = config.cosmos

    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--driver-memory {config.nvidia.driver_memory} pyspark-shell"
    )

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", config.nvidia.driver_memory)
        .config("spark.sql.defaultCatalog", catalog)
        # Iceberg catalog
        .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
        .config(f"spark.sql.catalog.{catalog}.uri", cat_cfg.uri)
        .config(f"spark.sql.catalog.{catalog}.warehouse", cat_cfg.warehouse)
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
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Tuning
        .config("spark.sql.shuffle.partitions", str(cosmos.shuffle_partitions))
        .config("spark.sql.iceberg.write.target-file-size-bytes", str(cosmos.target_file_size_bytes))
        .config("spark.sql.parquet.compression.codec", "zstd")
        # Extensions
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    )
    return builder.getOrCreate()


def create_namespace(spark: SparkSession, config: CosmosPipelineConfig) -> None:
    """Create the nvidia_cosmos namespace if it doesn't exist."""
    catalog = config.spark_catalog_name
    ns = config.cosmos.namespace
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{ns}")
    print(f"  Ensured namespace: {catalog}.{ns}")
