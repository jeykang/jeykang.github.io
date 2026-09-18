"""Warmed count(*) latency on the largest registered production Bronze table.
Production Bronze is consolidated (canonical schema): one `Radar` table holds all
radar rows. count(*) with no filter is answered from Iceberg manifest stats
(metadata-only) -> does NOT scan NFS data. Reports median-of-5 on `Radar`."""
import time, statistics
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("prod-count-bench")
    .config("spark.sql.defaultCatalog", "iceberg")
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.iceberg.catalog-impl", "org.apache.iceberg.rest.RESTCatalog")
    .config("spark.sql.catalog.iceberg.uri", "http://polaris:8181/api/catalog")
    .config("spark.sql.catalog.iceberg.warehouse", "lakehouse_catalog")
    .config("spark.sql.catalog.iceberg.io-impl", "org.apache.iceberg.io.ResolvingFileIO")
    .config("spark.sql.catalog.iceberg.s3.endpoint", "http://minio:9000")
    .config("spark.sql.catalog.iceberg.s3.path-style-access", "true")
    .config("spark.sql.catalog.iceberg.s3.access-key-id", "minioadmin")
    .config("spark.sql.catalog.iceberg.s3.secret-access-key", "minioadmin")
    .config("spark.sql.catalog.iceberg.oauth2-server-uri", "http://polaris:8181/api/catalog/v1/oauth/tokens")
    .config("spark.sql.catalog.iceberg.credential", "root:s3cr3t")
    .config("spark.sql.catalog.iceberg.scope", "PRINCIPAL_ROLE:ALL")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# consolidated registered Bronze tables (case-sensitive Iceberg names -> backticks)
cand = ["Radar", "Camera", "Lidar", "EgoMotion", "aux_egomotion", "clip_index"]
counts = {}
for t in cand:
    try:
        counts[t] = spark.sql(f"SELECT count(*) FROM iceberg.nvidia_bronze.`{t}`").collect()[0][0]
    except Exception as e:
        print(f"  skip {t}: {str(e)[:80]}")
for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {t:16s} {n:>16,} rows")

largest = max(counts, key=counts.get)
TBL = f"iceberg.nvidia_bronze.`{largest}`"
print(f"\nlargest registered Bronze table: {largest} = {counts[largest]:,} rows")

spark.sql("CLEAR CACHE")
spark.sql(f"SELECT count(*) FROM {TBL}").collect()          # warm-up
times = []
for _ in range(5):
    t = time.time()
    n = spark.sql(f"SELECT count(*) FROM {TBL}").collect()[0][0]
    times.append((time.time() - t) * 1000.0)
print(f"\n=== warmed SELECT count(*) on {largest} ({n:,} rows), median of 5 ===")
print("per-run ms:", [round(x, 1) for x in times])
print(f"MEDIAN: {round(statistics.median(times), 1)} ms   (min {round(min(times),1)} / max {round(max(times),1)})")
spark.stop()
