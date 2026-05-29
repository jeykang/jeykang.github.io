"""Re-register the Bronze lidar table from the 9 salvaged NFS chunks.

The earlier lidar Bronze table was broken (manifests point at 0-byte parquets
after the extract_remaining.sh data-loss incident on 2026-04-10; catalog now
returns 404 for the table). This script drops any lingering table and
re-runs add_files() against the chunks that actually contain data.

Run: spark-submit nvidia_ingestion/register_lidar_only.py
"""
from nvidia_ingestion.config import (
    NvidiaPipelineConfig,
    build_spark_session,
    create_namespaces,
)
from nvidia_ingestion.register_bronze import BronzeRegistrar

cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg, app_name="nvidia-register-lidar-salvage")
try:
    create_namespaces(spark, cfg)
    fq = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_bronze}.lidar"
    spark.sql(f"DROP TABLE IF EXISTS {fq}")  # NO PURGE — add_files-registered, source files would be deleted
    print(f"  [DROP] {fq}")
    reg = BronzeRegistrar(spark, cfg)
    rows = reg.register_nfs_sensor("lidar", "lidar/lidar_top_360fov")
    print(f"\n  LIDAR Bronze re-registered: {rows:,} rows")
    # Summary: chunks + clips
    df = spark.table(fq)
    print(f"  Schema: {df.schema.simpleString()}")
    print(f"  Distinct spin_index values: {df.select('spin_index').distinct().count()}")
finally:
    spark.stop()
