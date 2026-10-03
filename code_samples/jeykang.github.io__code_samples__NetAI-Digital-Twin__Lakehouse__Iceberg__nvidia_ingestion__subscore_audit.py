"""Audit sub-score distribution in Gold clip_scores table."""
from pyspark.sql import functions as F

from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg, app_name="subscore-audit")

tbl = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.clip_scores"
df = spark.table(tbl)

parsed = df.withColumn("sub", F.from_json(
    F.col("detail"),
    "struct<sub_scores:struct<"
    "time_of_day:double,"
    "season_geography:double,"
    "sensor_coverage:double,"
    "ego_dynamics:double,"
    "obstacle_density:double>>"
))

sub_fields = ["time_of_day", "season_geography", "sensor_coverage",
              "ego_dynamics", "obstacle_density"]

print("\n=== sub-score statistics ===")
for f in sub_fields:
    col = F.col(f"sub.sub_scores.{f}")
    stats = (
        parsed
        .agg(
            F.min(col).alias("min"),
            F.max(col).alias("max"),
            F.avg(col).alias("mean"),
            F.stddev(col).alias("std"),
            F.countDistinct(col).alias("n_distinct"),
        )
        .collect()[0]
    )
    print(f"  {f:20s} "
          f"min={stats['min']:.4f}  "
          f"max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}  "
          f"std={stats['std']:.4f}  "
          f"n_distinct={stats['n_distinct']}")

print("\n=== overall score percentiles ===")
pcts = df.selectExpr(
    "percentile_approx(difficulty_score, 0.00) as p0",
    "percentile_approx(difficulty_score, 0.10) as p10",
    "percentile_approx(difficulty_score, 0.25) as p25",
    "percentile_approx(difficulty_score, 0.50) as p50",
    "percentile_approx(difficulty_score, 0.75) as p75",
    "percentile_approx(difficulty_score, 0.90) as p90",
    "percentile_approx(difficulty_score, 0.95) as p95",
    "percentile_approx(difficulty_score, 0.99) as p99",
    "percentile_approx(difficulty_score, 1.00) as p100",
).collect()[0]
for k in ["p0", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p100"]:
    print(f"  {k}: {pcts[k]:.4f}")

spark.stop()
