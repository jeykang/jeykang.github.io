from pyspark.sql import SparkSession
from pyspark.sql.types import *
from datetime import datetime

# Spark 세션 생성 - 수정된 버전
spark = SparkSession.builder \
    .appName("DeltaLakeExample") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.warehouse.dir", "/tmp/delta-table") \
    .enableHiveSupport() \
    .getOrCreate()

# Delta Lake를 기본 데이터 소스로 설정
spark.sql("SET spark.sql.legacy.createHiveTableByDefault = false")

# 스키마 정의
patient_schema = StructType([
    StructField("patient_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("diagnosis", StringType(), True),
    StructField("registration_date", TimestampType(), True)
])

# 샘플 데이터
patient_data = [
    ("P001", "John Doe", 45, "Pneumonia", datetime.now()),
    ("P002", "Jane Smith", 35, "Fracture", datetime.now())
]

# DataFrame 생성
patient_df = spark.createDataFrame(patient_data, patient_schema)

# 방법 1: 경로를 지정하여 저장
patient_df.write.format("delta").mode("overwrite").save("delta-table/patients")

# 저장된 데이터 읽기 테스트
read_df = spark.read.format("delta").load("delta-table/patients")
read_df.show()