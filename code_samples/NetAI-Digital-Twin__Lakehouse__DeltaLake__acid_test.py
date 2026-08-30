from pyspark.sql import SparkSession
from delta.tables import *
import os

# SparkSession 생성
spark = SparkSession.builder \
    .appName("ACID Test: Data Lake vs. Delta Lake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 경로 설정
delta_path = "s3a://your-bucket/delta-acid-test"
datalake_path = "s3a://your-bucket/datalake-acid-test"

# 테스트 데이터 준비
data1 = spark.createDataFrame(
    [(1, "A"), (2, "B"), (3, "C")], ["id", "value"]
)

data2 = spark.createDataFrame(
    [(2, "B_updated"), (4, "D")], ["id", "value"]
)

# 1. Atomicity 테스트
print("\n--- Atomicity Test ---")

# Delta Lake
try:
    print("Delta Lake Atomicity Test:")
    data1.write.format("delta").mode("overwrite").save(delta_path)
    # 오류 유발 트랜잭션
    faulty_data = spark.createDataFrame([(5, "E"), ("X", "Faulty")], ["id", "value"])
    faulty_data.write.format("delta").mode("append").save(delta_path)
except Exception as e:
    print(f"Delta Lake Atomicity Failed: {e}")

# Data Lake
try:
    print("Data Lake Atomicity Test:")
    data1.write.format("parquet").mode("overwrite").save(datalake_path)
    # 오류 유발 트랜잭션
    faulty_data.write.format("parquet").mode("append").save(datalake_path)
except Exception as e:
    print(f"Data Lake Atomicity Failed: {e}")

print("\nData after Atomicity Test (Delta Lake):")
DeltaTable.forPath(spark, delta_path).toDF().show()

print("\nData after Atomicity Test (Data Lake):")
spark.read.format("parquet").load(datalake_path).show()

# 2. Consistency 테스트
print("\n--- Consistency Test ---")

# Delta Lake
try:
    print("Delta Lake Consistency Test:")
    data2.write.format("delta").mode("append").save(delta_path)
except Exception as e:
    print(f"Delta Lake Consistency Failed: {e}")

# Data Lake
try:
    print("Data Lake Consistency Test:")
    data2.write.format("parquet").mode("append").save(datalake_path)
except Exception as e:
    print(f"Data Lake Consistency Failed: {e}")

print("\nData after Consistency Test (Delta Lake):")
DeltaTable.forPath(spark, delta_path).toDF().show()

print("\nData after Consistency Test (Data Lake):")
spark.read.format("parquet").load(datalake_path).show()

# 3. Isolation 테스트
print("\n--- Isolation Test ---")
from threading import Thread
import time

def transaction1_delta():
    spark.createDataFrame([(10, "T1")], ["id", "value"]).write.format("delta").mode("append").save(delta_path)
    time.sleep(5)
    print("Delta Lake Transaction 1 Committed.")

def transaction2_delta():
    time.sleep(1)
    spark.createDataFrame([(10, "T2")], ["id", "value"]).write.format("delta").mode("append").save(delta_path)
    print("Delta Lake Transaction 2 Committed.")

def transaction1_datalake():
    spark.createDataFrame([(10, "T1")], ["id", "value"]).write.format("parquet").mode("append").save(datalake_path)
    time.sleep(5)
    print("Data Lake Transaction 1 Committed.")

def transaction2_datalake():
    time.sleep(1)
    spark.createDataFrame([(10, "T2")], ["id", "value"]).write.format("parquet").mode("append").save(datalake_path)
    print("Data Lake Transaction 2 Committed.")

# Delta Lake Isolation Test
t1_delta = Thread(target=transaction1_delta)
t2_delta = Thread(target=transaction2_delta)
t1_delta.start()
t2_delta.start()
t1_delta.join()
t2_delta.join()

print("\nData after Isolation Test (Delta Lake):")
DeltaTable.forPath(spark, delta_path).toDF().show()

# Data Lake Isolation Test
t1_datalake = Thread(target=transaction1_datalake)
t2_datalake = Thread(target=transaction2_datalake)
t1_datalake.start()
t2_datalake.join()
t1_datalake.join()

print("\nData after Isolation Test (Data Lake):")
spark.read.format("parquet").load(datalake_path).show()

# 4. Durability 테스트
print("\n--- Durability Test ---")

# Delta Lake 내구성 테스트
spark.stop()
spark = SparkSession.builder \
    .appName("Durability Test") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

print("\nDelta Lake Data after Restart:")
DeltaTable.forPath(spark, delta_path).toDF().show()

print("\nData Lake Data after Restart:")
spark.read.format("parquet").load(datalake_path).show()

spark.stop()
