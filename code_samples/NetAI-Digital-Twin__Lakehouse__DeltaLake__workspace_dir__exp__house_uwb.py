from minio import Minio
from minio.error import S3Error
from minio.versioningconfig import VersioningConfig
import pandas as pd
import json, os
from datetime import datetime, timedelta
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DoubleType, ArrayType, IntegerType
from pyspark.sql.functions import col, lit, struct
from delta import *

from exp_spark_builder import SparkBuilder


builder = SparkBuilder(save_path='../lakehouse/jars')
spark = builder.spark

bucket_name = "lakehouse-uwb"
bucket_name = "lake-uwb"

# schema = StructType([
#     StructField("batch_id", IntegerType(), True),
#     StructField("error_logs", StringType(), True)
# ])

schema = StructType([
    StructField("body", StructType([
        StructField("id", StringType(), True),
        StructField("datastreams", ArrayType(StructType([
            StructField("id", StringType(), True),
            StructField("current_value", StringType(), True),
            StructField("at", StringType(), True)
        ])), True),
        StructField("uuid", StringType(), True),
        StructField("address", StringType(), True),
        StructField("extended_tag_position", StructType([
            StructField("master", StringType(), True),
            StructField("corrected", BooleanType(), True),
            StructField("slaves", ArrayType(StructType([
                StructField("address", StringType(), True),
                StructField("time", DoubleType(), True),
                StructField("fp", DoubleType(), True),
                StructField("rssi", DoubleType(), True)
            ])), True)
        ]), True)
    ]), True),
    StructField("resource", StringType(), True)
])

# JSON 데이터 읽기
# minio_data = spark.read \
#     .option("multiline", True) \
#     .json(f"s3a://{bucket_name}/uwb-rtls-feed16_20241229T003757.json")

# # posX, posY, resource 추출
# filtered_data = minio_data.select(
#     col("resource"),
#     col("body.datastreams")[0].getField("current_value").alias("posX"),
#     col("body.datastreams")[1].getField("current_value").alias("posY"),
#     col("body.datastreams")[2].getField("current_value").alias("clr")
# )
# 데이터 확인
# filtered_data.show(truncate=False)
# print(filtered_data['resource'])

data = {
    "resource": [f"/feeds/{i}" for i in range(100)],
    "posX": [22.62] * 100,
    "posY": [-21.40] * 100,
    "clr": [0.36] * 100
}

df = spark.createDataFrame(pd.DataFrame(data))

single_file_df = df.coalesce(1)

single_file_df.write \
    .format("delta") \
    .mode("append") \
    .save(f"s3a://lakehouse-uwb")

# minio_data = spark.read \
#     .schema(schema) \
#     .option("multiline", True) \
#     .json(f"s3a://{bucket_name}/uwb-rtls-feed16_20241229T003757.json")
# # minio_data.printSchema()
# # minio_data.show(truncate=False)

# minio_data.write \
#     .format("delta") \
#     .mode("append") \
#     .save(f"s3a://lakehouse-uwb")
    
# mode("overwrite"): -> 기존 데이터를 덮어쓰고 새 데이터를 저장
# minio_data = spark.read \
#     .format("json") \
#     .option("mode", "DROPMALFORMED") \
#     .load(f"s3a://{bucket_name}/uwb-rtls-feed16_20241229T003757.json")
#     # .schema(schema) \
        
# minio_data.printSchema()
# print("\n\nminio_data:")
# minio_data.show(truncate=False)