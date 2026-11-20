"""Pick 3 distinct Gold 'hard' clip examples for presentation use.

Runs inside spark-iceberg (root) which can read Gold (s3/MinIO) but not
NFS-backed Bronze tables. So we only pull clip_scores here and leave
metadata enrichment + frame extraction to a host-side step.
"""
from __future__ import annotations

import json

from pyspark.sql import functions as F

from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session


def main() -> None:
    cfg = NvidiaPipelineConfig()
    spark = build_spark_session(cfg, app_name="gold-hard-examples")
    try:
        scores_fq = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.clip_scores"
        scores = spark.table(scores_fq).filter(F.col("difficulty_score") >= 0)
        total = scores.count()
        print(f"clip_scores rows: {total:,}")

        top = (
            scores.orderBy(F.col("difficulty_score").desc())
            .limit(5000)
            .select("clip_id", "difficulty_score", "detail")
            .collect()
        )
        out = [
            {"clip_id": r.clip_id,
             "difficulty_score": float(r.difficulty_score),
             "detail": r.detail}
            for r in top
        ]
        with open("/user_data/gold_top5000.json", "w") as f:
            json.dump(out, f)
        print(f"Wrote /user_data/gold_top5000.json ({len(out)} rows)")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
