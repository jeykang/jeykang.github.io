from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import sys
import time

# =============================================================================
# [PART 1] Environment & Spark Init (Polaris REST Catalog)
# =============================================================================
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
aws_region = os.getenv("AWS_REGION", "us-east-1")
s3_endpoint = os.getenv("AWS_S3_ENDPOINT", "http://minio:9000")
polaris_uri = os.getenv("POLARIS_URI", "http://polaris:8181/api/catalog")
polaris_warehouse = os.getenv("POLARIS_CATALOG_NAME", "lakehouse_catalog")
polaris_credential = os.getenv("POLARIS_CREDENTIAL", "root:s3cr3t")
polaris_scope = os.getenv("POLARIS_SCOPE", "PRINCIPAL_ROLE:ALL")
RAW_DATA_PATH = "/user_data/nuscenes-mini/v1.0-mini/v1.0-mini"

catalog = "iceberg"

spark = SparkSession.builder \
    .appName("Polaris-Iceberg-Gold") \
    .config("spark.sql.defaultCatalog", catalog) \
    .config(f"spark.sql.catalog.{catalog}", "org.apache.iceberg.spark.SparkCatalog") \
    .config(f"spark.sql.catalog.{catalog}.catalog-impl", "org.apache.iceberg.rest.RESTCatalog") \
    .config(f"spark.sql.catalog.{catalog}.uri", polaris_uri) \
    .config(f"spark.sql.catalog.{catalog}.warehouse", polaris_warehouse) \
    .config(f"spark.sql.catalog.{catalog}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .config(f"spark.sql.catalog.{catalog}.s3.endpoint", s3_endpoint) \
    .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", "true") \
    .config(f"spark.sql.catalog.{catalog}.s3.access-key-id", aws_access_key) \
    .config(f"spark.sql.catalog.{catalog}.s3.secret-access-key", aws_secret_key) \
    .config(f"spark.sql.catalog.{catalog}.credential", polaris_credential) \
    .config(f"spark.sql.catalog.{catalog}.scope", polaris_scope) \
    .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint) \
    .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
    .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .getOrCreate()

# =============================================================================
# [PART 2] Phase 1: Ingestion & Gold Table Creation (ETL)
# =============================================================================
print(">>> [Lakehouse] Phase 1: Gold Table 생성 (Join + Denormalization)")
setup_start = time.time()

# 1. JSON 읽기
df_sample = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample.json")
df_sample_data = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_data.json")
df_annotation = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sample_annotation.json")
df_category = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/category.json")
df_instance = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/instance.json")
df_sensor = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/sensor.json")
df_calibrated = spark.read.option("multiLine", True).json(f"{RAW_DATA_PATH}/calibrated_sensor.json")

# 2. 복잡한 조인을 미리 수행 (Denormalization)
# CAM_FRONT 채널 정보 결합
df_channel_map = df_calibrated.join(df_sensor, df_calibrated["sensor_token"] == df_sensor["token"]) \
    .select(df_calibrated["token"].alias("calib_token"), df_sensor["channel"])

# 전체 조인 수행 (Gold Table용 데이터 구성)
df_gold_raw = df_sample_data.join(df_channel_map, df_sample_data["calibrated_sensor_token"] == df_channel_map["calib_token"]) \
    .join(df_annotation, df_sample_data["sample_token"] == df_annotation["sample_token"]) \
    .join(df_instance, df_annotation["instance_token"] == df_instance["token"]) \
    .join(df_category, df_instance["category_token"] == df_category["token"]) \
    .select(
        df_sample_data["filename"].alias("img_path"),
        df_annotation["translation"],
        df_annotation["size"],
        df_annotation["rotation"],
        df_sensor["channel"],
        df_category["name"].alias("category_name")
    )

# 3. 데이터 스케일링 (10배 증강 -> 조인 폭발 시뮬레이션 결과와 맞추기 위해 100배 효과 적용 가능)
SCALE_FACTOR = 7
def scale_df(df, factor):
    if factor <= 1: return df
    # Baseline과 동일하게 10x10=100배 효과를 내기 위해 factor*factor로 증강
    return df.crossJoin(spark.range(factor * factor)).drop("id")

print(f">>> [Experiment] Scaling Gold Table by {SCALE_FACTOR}x{SCALE_FACTOR}=100x ...")
df_gold_final = scale_df(df_gold_raw, SCALE_FACTOR)

# 4. Iceberg Gold Table 저장
# CAM_FRONT나 Category에 상관없이 일단 저장한 뒤 쿼리에서 필터링하는 방식이 실용적입니다.
df_gold_final.write.format("iceberg") \
    .partitionBy("channel") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog}.nusc_exp.gold_train_set")

print(f"Gold Table Ingestion Finished: {time.time() - setup_start:.2f}s")

# -----------------------------------------------------------------------------
# [PART 3] Phase 2: Experiment (Query from Single Gold Table)
# -----------------------------------------------------------------------------
print(">>> [Lakehouse] Phase 2: 실험 시작 - 단일 Gold 테이블 쿼리")
query_start = time.time()

# 조인이 전혀 없는 단순 필터링 쿼리
query = f"""
SELECT img_path, translation, size, rotation
FROM {catalog}.nusc_exp.gold_train_set
WHERE channel = 'CAM_FRONT' 
  AND category_name = 'human.pedestrian.adult'
"""

result_df = spark.sql(query)
count = result_df.count()

query_end = time.time()
print(f">>> [Lakehouse] 결과: 보행자 데이터 {count}건 추출 완료")
print(f">>> [Lakehouse] 데이터셋 구성 소요 시간: {query_end - query_start:.4f}초")

spark.stop()