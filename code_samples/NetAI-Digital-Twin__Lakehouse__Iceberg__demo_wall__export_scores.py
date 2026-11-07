#!/usr/bin/env python3
"""export_scores.py — snapshot the Gold difficulty scores for the wall display.

Runs INSIDE the spark-iceberg container (demo_wall/ is not bind-mounted there, so
copy it in first). Reads iceberg.nvidia_gold.clip_scores, applies the same Gold cut
as edge_case_scorer.build_gold_subset (top 10% of the sensor-covered clips on
`difficulty_camera`, percentile_approx), and writes:

  /user_data/wall_scores.parquet   clip_id, difficulty_camera, sensor_covered, gold,
                                   conflict (behavioral axis), camera (perceptual
                                   camera axis) — difficulty_camera is their noisy-OR
  /user_data/wall_scores.json      threshold + counts (printed too)

/user_data is the repo's gitignored user_data/ directory on the host, where
build_media.py picks both files up.

    docker cp demo_wall/export_scores.py spark-iceberg:/tmp/export_scores.py
    docker exec -w /opt/spark spark-iceberg /opt/spark/bin/spark-submit /tmp/export_scores.py
"""
import json
import sys

sys.path.insert(0, "/opt/spark")  # nvidia_ingestion/ is bind-mounted here

from pyspark.sql import functions as F  # noqa: E402

from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session  # noqa: E402

TOP_PCT = 10.0
OUT = "/user_data/wall_scores"


def main():
    cfg = NvidiaPipelineConfig()
    spark = build_spark_session(cfg, app_name="wall-export-scores")
    spark.sparkContext.setLogLevel("ERROR")
    table = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.clip_scores"
    df = spark.table(table).select(
        "clip_id", "difficulty_camera", "sensor_covered",
        F.get_json_object("detail", "$.sub_scores.behavioral_axis").cast("double").alias("conflict"),
        F.get_json_object("detail", "$.sub_scores.perceptual_camera_axis").cast("double").alias("camera"),
    )

    covered = df.filter(F.col("sensor_covered")).filter(F.col("difficulty_camera") >= 0)
    q = 1.0 - TOP_PCT / 100.0
    threshold = covered.selectExpr(f"percentile_approx(difficulty_camera, {q}) AS t").collect()[0].t
    out = df.withColumn(
        "gold",
        F.col("sensor_covered") & (F.col("difficulty_camera") >= F.lit(threshold)),
    )
    pdf = out.toPandas()
    pdf.to_parquet(OUT + ".parquet", index=False)

    snap = spark.sql(f"SELECT snapshot_id, committed_at FROM {table}.snapshots "
                     "ORDER BY committed_at DESC LIMIT 1").collect()[0]
    summary = {
        "table": table,
        "snapshot_id": str(snap.snapshot_id),
        "committed_at": str(snap.committed_at),
        "scored": int(len(pdf)),
        "sensor_covered": int(pdf["sensor_covered"].sum()),
        "gold": int(pdf["gold"].sum()),
        "gold_axis": "camera",
        "top_pct": TOP_PCT,
        "threshold": float(threshold),
    }
    with open(OUT + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[wall] " + json.dumps(summary))


if __name__ == "__main__":
    main()
