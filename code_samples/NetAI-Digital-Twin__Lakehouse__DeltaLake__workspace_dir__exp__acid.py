import time
import random
import uuid
import json
import io
import threading

from exp_spark_builder import SparkBuilder
# -------------------------------
# 1. MinIO Data Lake 설정
# -------------------------------
from minio import Minio

MINIO_ENDPOINT = "10.32.174.125:9040"          # 실제 환경에 맞게 수정
MINIO_ACCESS_KEY = "pRWLQmzIoCE5nUKyac1O"      # 실제 환경에 맞게 수정
MINIO_SECRET_KEY = "8FpYdGdHL14opVBipvvGzjScTMNaQSHOjH9WaUZp"      # 실제 환경에 맞게 수정
USE_HTTPS = False

DATA_LAKE_BUCKET = "datalake-bucket"

minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=USE_HTTPS
)

# 버킷이 없으면 생성
if not minio_client.bucket_exists(DATA_LAKE_BUCKET):
    minio_client.make_bucket(DATA_LAKE_BUCKET)

# -------------------------------
# 2. Spark + Delta Lake 설정
# -------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType

builder = SparkBuilder(save_path='../lakehouse/jars')
spark = builder.spark

# 실제론 s3a:// 경로를 사용할 수 있지만, 여기서는 /tmp/delta_lake/ 예시
DELTA_TABLE_UWB = "/tmp/delta_lake/uwb_rtls_table"
DELTA_TABLE_GPU = "/tmp/delta_lake/gpu_metric_table"

# -------------------------------
# 3. 스키마 정의
# -------------------------------
uwb_schema = StructType([
    StructField("device_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True),
    StructField("z", DoubleType(), True),
])

gpu_schema = StructType([
    StructField("host_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("gpu_usage", DoubleType(), True),
    StructField("gpu_temp", DoubleType(), True),
])

# -------------------------------
# 4. 데이터 생성 함수 (UWB, GPU)
# -------------------------------
def generate_uwb_data(device_id):
    return {
        "device_id": device_id,
        "timestamp": int(time.time()),  # epoch
        "x": round(random.uniform(0, 100), 2),
        "y": round(random.uniform(0, 100), 2),
        "z": round(random.uniform(0, 100), 2),
    }

def generate_gpu_data(host_id):
    return {
        "host_id": host_id,
        "timestamp": int(time.time()),
        "gpu_usage": round(random.uniform(0, 100), 2),
        "gpu_temp": round(random.uniform(30, 90), 2),
    }

# ===================================================================
# (A) Data Lake 코드 (MinIO JSON)
# ===================================================================

def write_data_lake_uwb(data):
    """
    Data Lake: UWB를 JSON 파일 형태로 MinIO에 업로드
    (동시성 제어 X -> 충돌 시 덮어쓰기, 중복 가능)
    """
    filename = f"uwb_{data['device_id']}_{data['timestamp']}_{uuid.uuid4()}.json"
    json_str = json.dumps(data, ensure_ascii=False)
    json_bytes = json_str.encode('utf-8')

    minio_client.put_object(
        bucket_name=DATA_LAKE_BUCKET,
        object_name=filename,
        data=io.BytesIO(json_bytes),
        length=len(json_bytes),
        content_type='application/json'
    )

def write_data_lake_gpu(data):
    """
    Data Lake: GPU 데이터를 JSON 파일로 MinIO에 업로드
    """
    filename = f"gpu_{data['host_id']}_{data['timestamp']}_{uuid.uuid4()}.json"
    json_str = json.dumps(data, ensure_ascii=False)
    json_bytes = json_str.encode('utf-8')

    minio_client.put_object(
        bucket_name=DATA_LAKE_BUCKET,
        object_name=filename,
        data=io.BytesIO(json_bytes),
        length=len(json_bytes),
        content_type='application/json'
    )

def concurrent_update_data_lake(thread_id, device_id, host_id, iterations=5):
    """
    Data Lake 동시성 테스트:
    여러 스레드가 JSON 파일로 UWB, GPU 데이터를 저장
    """
    for _ in range(iterations):
        uwb_rec = generate_uwb_data(device_id)
        gpu_rec = generate_gpu_data(host_id)
        write_data_lake_uwb(uwb_rec)
        write_data_lake_gpu(gpu_rec)
        time.sleep(random.uniform(0.01, 0.1))


# ===================================================================
# (B) Data Lakehouse 코드 (Spark + Delta)
# ===================================================================

def init_delta_tables():
    """
    Delta 테이블 초기화(빈 DF를 overwrite)
    """
    # UWB 테이블 초기화
    empty_uwb_df = spark.createDataFrame([], uwb_schema)
    empty_uwb_df.write.format("delta").mode("overwrite").save(DELTA_TABLE_UWB)

    # GPU 테이블 초기화
    empty_gpu_df = spark.createDataFrame([], gpu_schema)
    empty_gpu_df.write.format("delta").mode("overwrite").save(DELTA_TABLE_GPU)

def write_uwb_delta(device_id, use_merge=False):
    """
    Delta Lake에 UWB 데이터 쓰기
    - use_merge=True면 MERGE 사용 (동일 Key면 UPDATE, 없으면 INSERT)
    - 아니면 단순 Append
    """
    record = [generate_uwb_data(device_id)]
    df = spark.createDataFrame(record, uwb_schema)

    if use_merge:
        df.createOrReplaceTempView("new_uwb")
        spark.sql(f"""
        MERGE INTO delta.`{DELTA_TABLE_UWB}` as T
        USING new_uwb as S
        ON T.device_id = S.device_id AND T.timestamp = S.timestamp
        WHEN MATCHED THEN UPDATE SET
          x = S.x,
          y = S.y,
          z = S.z
        WHEN NOT MATCHED THEN INSERT *
        """)
    else:
        df.write.format("delta").mode("append").save(DELTA_TABLE_UWB)

def write_gpu_delta(host_id, use_merge=False):
    record = [generate_gpu_data(host_id)]
    df = spark.createDataFrame(record, gpu_schema)

    if use_merge:
        df.createOrReplaceTempView("new_gpu")
        spark.sql(f"""
        MERGE INTO delta.`{DELTA_TABLE_GPU}` as T
        USING new_gpu as S
        ON T.host_id = S.host_id AND T.timestamp = S.timestamp
        WHEN MATCHED THEN UPDATE SET
          gpu_usage = S.gpu_usage,
          gpu_temp = S.gpu_temp
        WHEN NOT MATCHED THEN INSERT *
        """)
    else:
        df.write.format("delta").mode("append").save(DELTA_TABLE_GPU)

def concurrent_update_data_lakehouse(thread_id, device_id, host_id, iterations=5, use_merge=False):
    """
    Data Lakehouse(Delta) 동시성 테스트:
    여러 스레드가 MERGE/INSERT/UPDATE 등으로 UWB, GPU 데이터 삽입
    """
    for _ in range(iterations):
        write_uwb_delta(device_id, use_merge=use_merge)
        write_gpu_delta(host_id, use_merge=use_merge)
        time.sleep(random.uniform(0.01, 0.1))


# ===================================================================
# (C) 최종 실행 함수
# ===================================================================

def run_data_lake_test(num_threads=5, iterations=5):
    """
    여러 스레드로 Data Lake(MinIO JSON) 동시 테스트
    """
    threads = []
    for i in range(num_threads):
        device_id = f"device_{i}"
        host_id = f"host_{i}"
        t = threading.Thread(
            target=concurrent_update_data_lake,
            args=(i, device_id, host_id, iterations)
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

def run_data_lakehouse_test(num_threads=5, iterations=5, use_merge=False):
    """
    여러 스레드로 Data Lakehouse(Delta) 동시 테스트
    """
    threads = []
    for i in range(num_threads):
        device_id = f"device_{i}"
        host_id = f"host_{i}"
        t = threading.Thread(
            target=concurrent_update_data_lakehouse,
            args=(i, device_id, host_id, iterations, use_merge)
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

def main():
    # 1) Data Lake Test
    print("=== [1] Data Lake Test (MinIO JSON) ===")
    run_data_lake_test(num_threads=5, iterations=5)
    print("[Data Lake] Completed. Check MinIO for JSON files.\n")

    # 2) Data Lakehouse Init
    print("=== [2] Initialize Delta Lake Tables ===")
    init_delta_tables()

    # 3) Data Lakehouse Test (Append Mode)
    print("=== [3] Data Lakehouse Test (Append Mode) ===")
    run_data_lakehouse_test(num_threads=5, iterations=5, use_merge=False)

    # 4) Data Lakehouse Test (MERGE Mode)
    print("=== [4] Data Lakehouse Test (MERGE Mode) ===")
    run_data_lakehouse_test(num_threads=5, iterations=5, use_merge=True)

    # 5) 결과 확인: Row Count
    uwb_count = spark.read.format("delta").load(DELTA_TABLE_UWB).count()
    gpu_count = spark.read.format("delta").load(DELTA_TABLE_GPU).count()
    print(f"[Delta Lake] UWB Table Row Count = {uwb_count}")
    print(f"[Delta Lake] GPU Table Row Count = {gpu_count}")

    # Time Travel 예시 (원한다면)
    # old_df = spark.read.format("delta").option("versionAsOf", 0).load(DELTA_TABLE_UWB)
    # old_df.show()

    print("=== All tests completed. ===")
    print("Data Lake (MinIO JSON) => No ACID")
    print("Data Lakehouse (Delta) => ACID Transaction & Time Travel possible.")

if __name__ == "__main__":
    main()
