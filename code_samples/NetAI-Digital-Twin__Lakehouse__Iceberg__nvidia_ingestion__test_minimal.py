#!/usr/bin/env python3
"""Minimal test: check if Spark can read/write Iceberg tables from file:// paths."""
import time, sys, os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.appName("minimal-test")
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
    .config("spark.sql.catalog.iceberg.oauth2-server-uri",
            "http://polaris:8181/api/catalog/v1/oauth/tokens")
    .config("spark.sql.catalog.iceberg.credential", "root:s3cr3t")
    .config("spark.sql.catalog.iceberg.scope", "PRINCIPAL_ROLE:ALL")
    # Hadoop S3A
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .getOrCreate()
)

SRC = "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"

# Step 1: Read a parquet from NFS
print("\n[STEP 1] Read clip_index.parquet from NFS")
t0 = time.time()
df = spark.read.parquet(f"file://{SRC}/clip_index.parquet")
print(f"  Columns: {df.columns[:5]}")
print(f"  Read schema in {time.time()-t0:.2f}s")

# Step 2: Write to Iceberg (normal path — data goes to MinIO via S3)
print("\n[STEP 2] Write to Iceberg (copies data to MinIO)")
t0 = time.time()
spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.nvidia_test")
df.writeTo("iceberg.nvidia_test.clip_index_copy").using("iceberg").tableProperty(
    "format-version", "2"
).createOrReplace()
count = spark.table("iceberg.nvidia_test.clip_index_copy").count()
print(f"  [OK] {count} rows written in {time.time()-t0:.2f}s")

# Step 3: Test input_file_name()
print("\n[STEP 3] Test input_file_name()")
t0 = time.time()
result = spark.sql("""
    SELECT input_file_name() AS fp
    FROM iceberg.nvidia_test.clip_index_copy
    LIMIT 1
""").collect()
print(f"  file_path: {result[0].fp[:100]}...")
print(f"  [OK] {time.time()-t0:.2f}s")

# Step 4: Try add_files() with file:// path for calibration
print("\n[STEP 4] add_files() from file:// path")
t0 = time.time()
cal_uri = f"file://{SRC}/calibration/camera_intrinsics"
cal_df = spark.read.parquet(cal_uri)
print(f"  Schema: {cal_df.columns[:5]}")
cal_df.limit(0).writeTo("iceberg.nvidia_test.cam_intrinsics").using("iceberg").tableProperty(
    "format-version", "2"
).createOrReplace()
spark.sql(
    "CALL iceberg.system.add_files("
    f"  table => 'iceberg.nvidia_test.cam_intrinsics',"
    f"  source_table => '`parquet`.`{cal_uri}`'"
    ")"
)
count2 = spark.table("iceberg.nvidia_test.cam_intrinsics").count()
print(f"  [OK] {count2} rows registered via add_files() in {time.time()-t0:.2f}s")

# Step 5: Verify data readable
print("\n[STEP 5] Read back add_files() table")
t0 = time.time()
result2 = spark.sql("""
    SELECT *, input_file_name() AS fp
    FROM iceberg.nvidia_test.cam_intrinsics
    LIMIT 3
""").collect()
for r in result2:
    print(f"  fp: {r.fp[:100]}...")
print(f"  [OK] {time.time()-t0:.2f}s")

# Cleanup
print("\n[CLEANUP]")
spark.sql("DROP TABLE IF EXISTS iceberg.nvidia_test.clip_index_copy PURGE")
spark.sql("DROP TABLE IF EXISTS iceberg.nvidia_test.cam_intrinsics PURGE")
spark.sql("DROP NAMESPACE IF EXISTS iceberg.nvidia_test")
print("  Done")

spark.stop()
print("\nALL TESTS PASSED")
