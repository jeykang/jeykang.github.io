import os
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from delta import *
import logging

from spark_builder import SparkBuilder

builder = SparkBuilder(save_path='jars')
logger = builder.logger
spark = builder.spark
bucket_name = "deltalake-bucket2"

# Delta Lake에 데이터를 저장하고 분석하는 예시
try:
    # PostgreSQL에서 데이터 가져오기
    l40_metrics_data = spark.read \
        .format("jdbc") \
        .option("url", f"jdbc:postgresql://{os.getenv('LOCAL_IP_ADDRESS')}:5432/postgres") \
        .option("dbtable", "l40_metrics") \
        .option("user", "postgres") \
        .option("password", "postgres") \
        .load()
    
    print("\n\npostgres_data_l40_metrics:")
    l40_metrics_data.show(truncate=False)

        
    # MinIo 에서 데이터 가져오기
    # JSON 파일 스키마 정의
    schema = StructType([
        StructField("batch_id", IntegerType(), True),
        StructField("error_logs", StringType(), True)
    ])
    
    minio_data = spark.read \
        .schema(schema) \
        .format("json") \
        .option("mode", "DROPMALFORMED") \
        .load("s3a://l40-logs-bucket/logs.json")
        
    print("\n\nminio_data:")
    minio_data.show(truncate=False)
    
    
    # Delta Lake로 변환하여 저장
    l40_metrics_data.write \
        .format("delta") \
        .mode("append") \
        .save(f"s3a://{bucket_name}/delta/l40_metrics_data")

    minio_data.write \
        .format("delta") \
        .mode("append") \
        .save(f"s3a://{bucket_name}/delta/minio_data")

    # 조인 연산 (id와 customer_id를 기준으로 조인)
    merged_data = l40_metrics_data.join(minio_data, "batch_id")
    merged_data.show(truncate=False)
    
    merged_data.write \
        .format("delta") \
        .mode("append") \
        .save(f"s3a://{bucket_name}/delta/merged_data")

except Exception as e:
    logger.error(f"에러 발생: {str(e)}", exc_info=True)

finally:
    spark.stop()