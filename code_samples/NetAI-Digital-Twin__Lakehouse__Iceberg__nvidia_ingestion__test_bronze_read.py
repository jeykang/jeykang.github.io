"""Smoke test: can Spark read Bronze tables when run as uid 1000:1007?"""
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session
from pyspark.sql import functions as F

cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg, app_name="bronze-read-smoke")
try:
    fq = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_bronze}.clip_index"
    df = spark.table(fq)
    print(f"clip_index count: {df.count():,}")
    print(f"sample:")
    df.limit(3).show(truncate=False)
    dc = spark.table(f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_bronze}.data_collection")
    print(f"data_collection count: {dc.count():,}")
    # Full join like scoring would do
    joined = df.join(dc, "clip_id").filter(F.col("clip_is_valid") == True).limit(5)
    joined.select("clip_id", "chunk", "country", "hour_of_day").show(truncate=False)
    print("OK: Bronze reads succeeded as non-root")
finally:
    spark.stop()
