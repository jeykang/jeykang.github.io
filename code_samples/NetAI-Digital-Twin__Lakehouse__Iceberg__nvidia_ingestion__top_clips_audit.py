"""Inspect top and bottom clips by difficulty in new Gold scoring."""
from pyspark.sql import functions as F

from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

cfg = NvidiaPipelineConfig()
spark = build_spark_session(cfg, app_name="top-clips-audit")

tbl = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.clip_scores"
df = spark.table(tbl).withColumn(
    "sub", F.from_json(F.col("detail"),
    "struct<hour:int,season:string,country:string,"
    "sub_scores:struct<time_of_day:double,season_geography:double,"
    "sensor_coverage:double,ego_dynamics:double,obstacle_density:double>>")
)

print("\n=== 10 hardest clips ===")
df.select(
    "clip_id", "difficulty_score",
    F.col("sub.hour").alias("hr"),
    F.col("sub.season").alias("season"),
    F.col("sub.country").alias("country"),
    F.round(F.col("sub.sub_scores.time_of_day"), 3).alias("tod"),
    F.round(F.col("sub.sub_scores.season_geography"), 3).alias("sg"),
    F.round(F.col("sub.sub_scores.sensor_coverage"), 3).alias("sc"),
    F.round(F.col("sub.sub_scores.ego_dynamics"), 3).alias("ego"),
).orderBy(F.desc("difficulty_score")).show(10, truncate=False)

print("\n=== 10 easiest clips ===")
df.select(
    "clip_id", "difficulty_score",
    F.col("sub.hour").alias("hr"),
    F.col("sub.season").alias("season"),
    F.col("sub.country").alias("country"),
    F.round(F.col("sub.sub_scores.time_of_day"), 3).alias("tod"),
    F.round(F.col("sub.sub_scores.season_geography"), 3).alias("sg"),
    F.round(F.col("sub.sub_scores.sensor_coverage"), 3).alias("sc"),
    F.round(F.col("sub.sub_scores.ego_dynamics"), 3).alias("ego"),
).orderBy("difficulty_score").show(10, truncate=False)

print("\n=== top-10% threshold & sub-score means for that cohort ===")
thresh = df.approxQuantile("difficulty_score", [0.90], 0.001)[0]
print(f"  threshold (p90): {thresh:.4f}")
top = df.filter(F.col("difficulty_score") >= thresh)
stats = top.agg(
    F.count("*").alias("n"),
    F.avg(F.col("sub.sub_scores.time_of_day")).alias("tod"),
    F.avg(F.col("sub.sub_scores.season_geography")).alias("sg"),
    F.avg(F.col("sub.sub_scores.sensor_coverage")).alias("sc"),
    F.avg(F.col("sub.sub_scores.ego_dynamics")).alias("ego"),
).collect()[0]
print(f"  n={stats['n']:,}  tod={stats['tod']:.3f}  sg={stats['sg']:.3f}  "
      f"sc={stats['sc']:.3f}  ego={stats['ego']:.3f}")

print("\n=== bottom-10% sub-score means (for contrast) ===")
low = df.orderBy("difficulty_score").limit(int(df.count() * 0.1))
stats = low.agg(
    F.count("*").alias("n"),
    F.avg(F.col("sub.sub_scores.time_of_day")).alias("tod"),
    F.avg(F.col("sub.sub_scores.season_geography")).alias("sg"),
    F.avg(F.col("sub.sub_scores.sensor_coverage")).alias("sc"),
    F.avg(F.col("sub.sub_scores.ego_dynamics")).alias("ego"),
).collect()[0]
print(f"  n={stats['n']:,}  tod={stats['tod']:.3f}  sg={stats['sg']:.3f}  "
      f"sc={stats['sc']:.3f}  ego={stats['ego']:.3f}")

spark.stop()
