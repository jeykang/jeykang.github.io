import time
import random
import threading
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
import datetime
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

from exp_spark_builder import SparkBuilder

# 빌더 초기화
builder = SparkBuilder(save_path='../lakehouse/jars')
logger = builder.logger
spark = builder.spark


############################
# 2) 공통 스키마 정의
############################
schema = StructType([
    StructField("sensor_type", StringType(), True),     # "UWB" or "GPU"
    StructField("value", FloatType(), True),           # 임의의 센싱 값
    StructField("ts", TimestampType(), True),          # 기록 시점
])

############################
# 3) 경로 설정
############################
# 예) prim_1_20240101T103000.json - 단일 파일 (Lake)
LAKE_JSON_FILE = "s3a://lake-uwb/uwb-acid-test.json"

# Delta Lake 저장 경로
LAKEHOUSE_DELTA_PATH = "s3a://lakehouse-uwb/delta_prim_1"

############################
# 4) 디렉토리 초기화 함수
############################
# def cleanup_dirs():
#     # JSON 파일 위치
#     lake_path_local = LAKE_JSON_FILE.replace("file://", "")
#     try:
#         os.remove(lake_path_local)  # 파일 삭제
#     except:
#         pass

#     # Delta Lake 디렉토리
#     lakehouse_local = LAKEHOUSE_DELTA_PATH.replace("file://","")
#     try:
#         shutil.rmtree(lakehouse_local)
#     except:
#         pass

############################
# 5) 스레드 함수: UWB 센서 / GPU 센서
############################
def lake_write(sensor_type):
    """
    Lake 시나리오: 단일 JSON 파일에 동시에 쓰기
    충돌 유발을 위해 'append'와 'overwrite'를 랜덤하게 선택.
    """
    for i in range(5):  # 총 5회 쓰기
        val = random.random() * 100
        now = datetime.datetime.now()
        df = spark.createDataFrame([(sensor_type, val, now)], schema=schema)

        # 70% 확률로 append, 30% 확률로 overwrite
        mode_choice = "append" if random.random() < 0.7 else "overwrite"

        # JSON 파일에 쓰기
        df.write.json(LAKE_JSON_FILE, mode=mode_choice)

        # 잠시 대기하여 동시성 타이밍 충돌 유발
        time.sleep(random.uniform(0.01, 0.05))

def lakehouse_write(sensor_type):
    """
    Lakehouse 시나리오: Delta Lake에 쓰기(append)
    Spark가 내부적으로 ACID 트랜잭션을 처리하므로
    동시성 충돌을 자동 조정함.
    """
    for i in range(5):
        val = random.random() * 100
        now = datetime.datetime.now()
        df = spark.createDataFrame([(sensor_type, val, now)], schema=schema)

        # Delta Append
        df.write.format("delta").mode("append").save(LAKEHOUSE_DELTA_PATH)

        time.sleep(random.uniform(0.01, 0.05))

############################
# 6) 실험: Lake 시나리오
############################
def run_lake_experiment():
    """
    UWB 센서 스레드 + GPU 센서 스레드가
    동일 JSON 파일에 동시에 5회씩 쓰기.
    """
    t_uwb = threading.Thread(target=lake_write, args=("UWB",))
    t_gpu = threading.Thread(target=lake_write, args=("GPU",))

    t_uwb.start()
    t_gpu.start()
    t_uwb.join()
    t_gpu.join()

    # 최종 결과 읽기
    lake_path_local = LAKE_JSON_FILE.replace("file://","")
    if not os.path.exists(lake_path_local):
        print("[Lake] JSON file not found. Possibly overwritten or never created.")
        return

    df_lake = spark.read.json(LAKE_JSON_FILE, schema=schema)

    # UWB/GPU 각각 몇 개의 레코드가 있는지 출력
    cnt_uwb = df_lake.filter(df_lake.sensor_type == "UWB").count()
    cnt_gpu = df_lake.filter(df_lake.sensor_type == "GPU").count()
    total = df_lake.count()

    print("[Lake] Final JSON content")
    df_lake.show()
    print(f"[Lake] UWB count={cnt_uwb}, GPU count={cnt_gpu}, total={total}")

############################
# 7) 실험: Lakehouse 시나리오
############################
def run_lakehouse_experiment():
    """
    UWB 센서 스레드 + GPU 센서 스레드가
    동시에 Delta Lake에 append.
    """
    t_uwb = threading.Thread(target=lakehouse_write, args=("UWB",))
    t_gpu = threading.Thread(target=lakehouse_write, args=("GPU",))

    t_uwb.start()
    t_gpu.start()
    t_uwb.join()
    t_gpu.join()

    # 최종 결과 읽기
    df_lh = spark.read.format("delta").load(LAKEHOUSE_DELTA_PATH)
    cnt_uwb = df_lh.filter(df_lh.sensor_type == "UWB").count()
    cnt_gpu = df_lh.filter(df_lh.sensor_type == "GPU").count()
    total = df_lh.count()

    print("[Lakehouse] Final Delta content")
    df_lh.show()
    print(f"[Lakehouse] UWB count={cnt_uwb}, GPU count={cnt_gpu}, total={total}")

############################
# 8) 메인 실행
############################
if __name__ == "__main__":
    # 매번 실행 전 정리
    # cleanup_dirs()

    print("=== Running Lake Experiment (JSON) ===")
    run_lake_experiment()

    # cleanup_dirs()

    print("\n=== Running Lakehouse Experiment (Delta) ===")
    run_lakehouse_experiment()

    # 마무리
    spark.stop()
