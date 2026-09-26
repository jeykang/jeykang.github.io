from pyspark.sql import SparkSession
import os
import sys

# -----------------------------------------------------------------------------
# 1. Load Environment Variables (Passed from Docker Compose)
# -----------------------------------------------------------------------------
# Fetch env vars with defaults, or None if missing
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")
s3_endpoint = os.getenv("AWS_S3_ENDPOINT", "http://minio:9000")
nessie_uri = os.getenv("NESSIE_URI", "http://nessie:19120/api/v1")

# Validate essential credentials
if not aws_access_key or not aws_secret_key:
    print("Error: AWS Access Key or Secret Key is missing in environment variables.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. Initialize Spark Session (Inject Env Vars)
# -----------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("NessieMinioSpark") \
    .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,org.projectnessie.spark.extensions.NessieSparkSessionExtensions') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkCatalog') \
    .config('spark.sql.catalog.spark_catalog.catalog-impl', 'org.apache.iceberg.nessie.NessieCatalog') \
    .config('spark.sql.catalog.spark_catalog.uri', nessie_uri) \
    .config('spark.sql.catalog.spark_catalog.warehouse', 's3://spark1') \
    .config('spark.sql.catalog.spark_catalog.io-impl', 'org.apache.iceberg.aws.s3.S3FileIO') \
    .config('spark.sql.catalog.spark_catalog.s3.endpoint', s3_endpoint) \
    .config('spark.sql.catalog.spark_catalog.s3.path-style-access', 'true') \
    .config('spark.sql.defaultCatalog', 'spark_catalog') \
    .config('spark.sql.catalog.nessie', 'org.apache.iceberg.spark.SparkCatalog') \
    .config('spark.sql.catalog.nessie.warehouse', 's3://spark1') \
    .config('spark.sql.catalog.nessie.catalog-impl', 'org.apache.iceberg.nessie.NessieCatalog') \
    .config('spark.sql.catalog.nessie.io-impl', 'org.apache.iceberg.aws.s3.S3FileIO') \
    .config('spark.sql.catalog.nessie.uri', nessie_uri) \
    .config('spark.sql.catalog.nessie.ref', 'main') \
    .config('spark.sql.catalog.nessie.cache-enabled', 'false') \
    .config('spark.sql.catalog.nessie.s3.endpoint', s3_endpoint) \
    .config('spark.sql.catalog.nessie.s3.region', aws_region) \
    .config('spark.sql.catalog.nessie.s3.path-style-access', 'true') \
    .config('spark.sql.catalog.nessie.s3.access-key-id', aws_access_key) \
    .config('spark.sql.catalog.nessie.s3.secret-access-key', aws_secret_key) \
    .config('spark.hadoop.fs.s3a.access.key', aws_access_key) \
    .config('spark.hadoop.fs.s3a.secret.key', aws_secret_key) \
    .config('spark.hadoop.fs.s3a.endpoint', s3_endpoint) \
    .config('spark.hadoop.fs.s3a.path.style.access', 'true') \
    .config('spark.hadoop.fs.s3a.connection.ssl.enabled', 'false') \
    .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem') \
    .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider') \
    .getOrCreate()

# -----------------------------------------------------------------------------
# 3. Business Logic (Nessie Operations)
# -----------------------------------------------------------------------------

# Create Namespace
spark.sql("CREATE NAMESPACE IF NOT EXISTS nessie.db").show()

# Create Table & Insert Initial Data
spark.sql("CREATE TABLE IF NOT EXISTS nessie.db.demo (id bigint, data string) USING iceberg").show()
spark.sql("INSERT INTO nessie.db.demo (id, data) VALUES (1, 'a'), (2, 'b')").show()
spark.sql("SELECT * FROM nessie.db.demo").show()

# 1. List References
print("Current References:")
spark.sql("LIST REFERENCES IN nessie").show()

# Nessie Test
# 2. Create New Branch (etl_branch)
try:
    spark.sql("DROP BRANCH IF EXISTS etl_branch IN nessie")
except:
    pass
spark.sql("CREATE BRANCH etl_branch IN nessie").show()

# 3. Switch to Branch & Insert Data
spark.sql("USE REFERENCE etl_branch IN nessie")

print("Inserting data into etl_branch...")
spark.sql("INSERT INTO nessie.db.demo VALUES (3, 'c'), (4, 'd')")

# 4. Compare Branches
print("Main Branch Data (Original):")
spark.sql("USE REFERENCE main IN nessie")
spark.sql("SELECT * FROM nessie.db.demo").show()

print("etl_branch Data (New):")
spark.sql("USE REFERENCE etl_branch IN nessie")
spark.sql("SELECT * FROM nessie.db.demo").show()

# 5. Merge Branch
print("Merging etl_branch into main...")
spark.sql("MERGE BRANCH etl_branch INTO main IN nessie").show()

# 6. Verify Merge
print("Main Branch Data after Merge:")
spark.sql("USE REFERENCE main IN nessie") 
spark.sql("SELECT * FROM nessie.db.demo").show()

# 7. Cleanup Branch
spark.sql("DROP BRANCH etl_branch IN nessie").show()

import time
print("Spark Web UI is alive for 30 seconds. Go check it!")
# Keep container alive for UI inspection
time.sleep(30) 

spark.stop()