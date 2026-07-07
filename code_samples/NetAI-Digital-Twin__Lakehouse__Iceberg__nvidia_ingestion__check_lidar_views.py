from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session
cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg, app_name='lidar-silver-check')
for ns in ['nvidia_bronze', 'nvidia_silver', 'nvidia_gold']:
    try:
        tbls = [r.tableName for r in spark.sql(f'SHOW TABLES IN iceberg.{ns}').collect()]
        lidar_tbls = [t for t in tbls if 'lidar' in t.lower()]
        print(f'{ns}: lidar-related = {lidar_tbls}')
    except Exception as e:
        print(f'{ns}: ERR {e}')
# Test querying the new Bronze lidar
cnt = spark.sql('SELECT COUNT(*) n FROM iceberg.nvidia_bronze.lidar').collect()[0].n
print(f'Bronze lidar rows: {cnt}')
clips = spark.sql(
    "SELECT COUNT(DISTINCT regexp_extract(input_file_name(), "
    "'([0-9a-f-]{36})\\.lidar', 1)) AS n_clips FROM iceberg.nvidia_bronze.lidar"
).collect()[0].n_clips
print(f'Bronze lidar clips: {clips}')
spark.stop()
