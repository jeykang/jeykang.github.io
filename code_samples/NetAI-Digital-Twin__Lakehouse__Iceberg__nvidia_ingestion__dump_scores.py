"""Dump clip_scores (clip_id, difficulty_score) to a CSV for A/B Gold diffing."""
import sys
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/netai-e2e/_scores_dump.csv"
cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg)
df = spark.table(f"{cfg.spark_catalog_name}.nvidia_gold.clip_scores") \
    .select("clip_id", "difficulty_score", "perception_score", "sensor_covered")
rows = [(r["clip_id"], r["difficulty_score"], r["perception_score"], r["sensor_covered"])
        for r in df.collect()]
with open(out, "w") as f:
    for cid, d, p, sc in rows:
        f.write(f"{cid},{d},{p},{sc}\n")
print(f"dumped {len(rows)} clip scores -> {out}")
