#!/usr/bin/env python3
"""Quick timing debug: which step is slow for clip_index registration?"""
import os, time, tempfile, shutil
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"
from pyspark.sql import SparkSession

NFS_ROOT = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)

t0 = time.time()
spark = (SparkSession.builder.appName("debug-timing")
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
    .config("spark.sql.catalog.iceberg.oauth2.server-uri",
            "http://polaris:8181/api/catalog/v1/oauth/tokens")
    .config("spark.sql.catalog.iceberg.oauth2.credential", "root:s3cr3t")
    .config("spark.sql.catalog.iceberg.oauth2.scope", "PRINCIPAL_ROLE:ALL")
    .config("spark.sql.catalog.iceberg.credential", "root:s3cr3t")
    .config("spark.sql.catalog.iceberg.scope", "PRINCIPAL_ROLE:ALL")
    .config("spark.sql.catalog.iceberg.oauth2-server-uri",
            "http://polaris:8181/api/catalog/v1/oauth/tokens")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.driver.memory", "4g")
    .getOrCreate())
print(f"[1] Spark session: {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.debug_test")
print(f"[2] Create namespace: {time.time()-t1:.1f}s", flush=True)

t2 = time.time()
sample_dir = tempfile.mkdtemp(prefix="pq_debug_")
clip_pq = f"{NFS_ROOT}/clip_index.parquet"
os.symlink(clip_pq, os.path.join(sample_dir, "sample.parquet"))
df = spark.read.parquet(f"file://{sample_dir}")
print(f"[3] Read schema: {time.time()-t2:.1f}s, cols={len(df.columns)}", flush=True)

t3 = time.time()
df.limit(0).writeTo("iceberg.debug_test.clip_index").using("iceberg") \
    .tableProperty("format-version", "2").createOrReplace()
print(f"[4] Create table: {time.time()-t3:.1f}s", flush=True)

t4 = time.time()
spark.sql(
    f"CALL iceberg.system.add_files("
    f"  table => 'iceberg.debug_test.clip_index',"
    f"  source_table => '`parquet`.`file://{sample_dir}`'"
    f")"
)
print(f"[5] add_files: {time.time()-t4:.1f}s", flush=True)

t5 = time.time()
count = spark.table("iceberg.debug_test.clip_index").count()
print(f"[6] Count: {count} rows in {time.time()-t5:.1f}s", flush=True)

spark.sql("DROP TABLE IF EXISTS iceberg.debug_test.clip_index PURGE")
spark.sql("DROP NAMESPACE IF EXISTS iceberg.debug_test")
shutil.rmtree(sample_dir)
spark.stop()
print("Done")
