"""
Nvidia PhysicalAI Dataset — Query Latency Benchmark Suite

Measures read-path performance across Bronze, Silver, and Gold tiers:
  1. Gold vs Silver+Join latency comparison (pre-materialized vs ad-hoc)
  2. Partition pruning effectiveness (sensor_name filter on radar_ego_fusion)
  3. Temporal range scan (timestamp predicate pushdown via Iceberg metrics)
  4. Point lookup vs full scan (clip_id partition pruning)
  5. Aggregation queries (GROUP BY on partitioned vs non-partitioned columns)
  6. Time travel (snapshot-based read)

Usage (inside spark-iceberg container):
    python3 -m nvidia_ingestion.query_benchmark
"""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Must set before SparkSession import
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell"
)

from pyspark.sql import DataFrame, SparkSession


@dataclass
class QueryResult:
    experiment: str
    variant: str
    description: str
    median_s: float
    row_count: int
    runs: int
    all_times_s: List[float] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


CAT = "iceberg"
NS_BRONZE = "nvidia_bronze"
NS_SILVER = "nvidia_silver"
NS_GOLD = "nvidia_gold"

OUTPUT_PATH = "/tmp/nvidia_query_benchmark.json"


def _build_spark() -> SparkSession:
    from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session
    cfg = NvidiaPipelineConfig()
    return build_spark_session(cfg, "nvidia-query-benchmark")


def _time_query(
    spark: SparkSession,
    query_fn: Callable[[], DataFrame],
    warmup: int = 2,
    runs: int = 5,
    use_count: bool = False,
) -> Tuple[float, int, List[float]]:
    """Run query_fn with warmup, return (median_s, row_count, all_times)."""
    for _ in range(warmup):
        df = query_fn()
        if use_count:
            df.count()
        else:
            df.collect()

    times = []
    row_count = 0
    for _ in range(runs):
        t0 = time.time()
        df = query_fn()
        if use_count:
            row_count = df.count()
        else:
            rows = df.collect()
            row_count = len(rows)
        times.append(time.time() - t0)

    times.sort()
    median = times[len(times) // 2]
    return median, row_count, times


def _warmup_jvm(spark: SparkSession):
    """Trivial queries to warm up JVM/catalog caches."""
    print("  Warming up JVM...")
    for _ in range(3):
        spark.sql(
            f"SELECT count(*) FROM {CAT}.{NS_GOLD}.radar_ego_fusion"
        ).collect()
    print("  JVM warm.\n")


# ──────────────────────────────────────────────────────────────
# Experiment 1: Gold vs Silver+Join
# ──────────────────────────────────────────────────────────────
def exp1_gold_vs_silver_join(spark: SparkSession) -> List[QueryResult]:
    """Compare pre-materialized Gold read vs ad-hoc Silver multi-table join."""
    print("=" * 70)
    print("EXPERIMENT 1: Gold vs Silver+Join Latency")
    print("=" * 70)
    results = []

    # --- 1a: lidar_with_ego ---
    print("\n  [1a] lidar_with_ego — Gold single read vs Silver join")

    def gold_lidar():
        return spark.sql(f"""
            SELECT clip_id, reference_timestamp, ego_sample_count
            FROM {CAT}.{NS_GOLD}.lidar_with_ego
        """)

    def silver_lidar_join():
        return spark.sql(f"""
            SELECT l.clip_id, l.reference_timestamp, count(e.timestamp) as ego_sample_count
            FROM {CAT}.{NS_SILVER}.lidar l
            LEFT JOIN {CAT}.{NS_SILVER}.egomotion e ON l.clip_id = e.clip_id
            GROUP BY l.clip_id, l.reference_timestamp
        """)

    med_g, rows_g, times_g = _time_query(spark, gold_lidar)
    med_s, rows_s, times_s = _time_query(spark, silver_lidar_join)
    speedup = med_s / med_g if med_g > 0 else float("inf")

    print(f"    Gold:  {med_g:.3f}s ({rows_g:,} rows)")
    print(f"    Silver+Join: {med_s:.3f}s ({rows_s:,} rows)")
    print(f"    Speedup: {speedup:.1f}×")

    results.append(QueryResult("Gold vs Silver+Join", "Gold lidar_with_ego",
        "Pre-materialized Gold table read", med_g, rows_g, 5, times_g))
    results.append(QueryResult("Gold vs Silver+Join", "Silver lidar+ego join",
        "Ad-hoc join of Silver lidar + egomotion", med_s, rows_s, 5, times_s,
        {"speedup_vs_gold": f"{speedup:.1f}x"}))

    # --- 1b: radar_ego_fusion ---
    print("\n  [1b] radar_ego_fusion — Gold single read vs Silver union+join")

    def gold_radar():
        return spark.sql(f"""
            SELECT sensor_name, clip_id, timestamp, ego_sample_count
            FROM {CAT}.{NS_GOLD}.radar_ego_fusion
        """)

    # Build the same UNION ALL + JOIN that build_gold does
    radar_tables = [r.tableName for r in spark.sql(
        f"SHOW TABLES IN {CAT}.{NS_SILVER}"
    ).collect() if r.tableName.startswith("radar_")]

    union_parts = " UNION ALL ".join([
        f"SELECT clip_id, timestamp, '{t.replace('radar_', '', 1)}' as sn "
        f"FROM {CAT}.{NS_SILVER}.{t}"
        for t in radar_tables
    ])

    def silver_radar_join():
        return spark.sql(f"""
            SELECT r.sn, r.clip_id, r.timestamp, e.ego_sample_count
            FROM ({union_parts}) r
            LEFT JOIN (
                SELECT clip_id, count(*) as ego_sample_count
                FROM {CAT}.{NS_SILVER}.egomotion
                GROUP BY clip_id
            ) e ON r.clip_id = e.clip_id
        """)

    med_g2, rows_g2, times_g2 = _time_query(spark, gold_radar, use_count=True)
    med_s2, rows_s2, times_s2 = _time_query(spark, silver_radar_join, use_count=True)
    speedup2 = med_s2 / med_g2 if med_g2 > 0 else float("inf")

    print(f"    Gold:  {med_g2:.3f}s ({rows_g2:,} rows)")
    print(f"    Silver union+join: {med_s2:.3f}s ({rows_s2:,} rows)")
    print(f"    Speedup: {speedup2:.1f}×")

    results.append(QueryResult("Gold vs Silver+Join", "Gold radar_ego_fusion",
        "Pre-materialized Gold table count", med_g2, rows_g2, 5, times_g2))
    results.append(QueryResult("Gold vs Silver+Join", "Silver 19-table union+ego join",
        "Ad-hoc UNION ALL of 19 radar Silver tables + ego join", med_s2, rows_s2, 5, times_s2,
        {"speedup_vs_gold": f"{speedup2:.1f}x"}))

    # --- 1c: sensor_fusion_clip ---
    print("\n  [1c] sensor_fusion_clip — Gold single read vs Silver 4-table join")

    def gold_fusion():
        return spark.sql(f"""
            SELECT * FROM {CAT}.{NS_GOLD}.sensor_fusion_clip
        """)

    def silver_fusion_join():
        # Read 4 source tables individually and count (what an analyst
        # must do without the pre-joined Gold table)
        c1 = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_SILVER}.clip_index").collect()
        c2 = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_SILVER}.data_collection").collect()
        c3 = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_SILVER}.sensor_presence").collect()
        c4 = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_SILVER}.vehicle_dimensions").collect()
        return spark.sql(f"SELECT {c1[0][0] + c2[0][0] + c3[0][0] + c4[0][0]} as total")

    med_g3, rows_g3, times_g3 = _time_query(spark, gold_fusion, use_count=True)
    med_s3, rows_s3, times_s3 = _time_query(spark, silver_fusion_join, use_count=True)
    speedup3 = med_s3 / med_g3 if med_g3 > 0 else float("inf")

    print(f"    Gold:  {med_g3:.3f}s ({rows_g3:,} rows)")
    print(f"    Silver join: {med_s3:.3f}s ({rows_s3:,} rows)")
    print(f"    Speedup: {speedup3:.1f}×")

    results.append(QueryResult("Gold vs Silver+Join", "Gold sensor_fusion_clip",
        "Pre-materialized Gold table count", med_g3, rows_g3, 5, times_g3))
    results.append(QueryResult("Gold vs Silver+Join", "Silver clip+data+sensor join",
        "Ad-hoc join of 3 Silver metadata tables", med_s3, rows_s3, 5, times_s3,
        {"speedup_vs_gold": f"{speedup3:.1f}x"}))

    return results


# ──────────────────────────────────────────────────────────────
# Experiment 2: Partition Pruning
# ──────────────────────────────────────────────────────────────
def exp2_partition_pruning(spark: SparkSession) -> List[QueryResult]:
    """Measure partition pruning on radar_ego_fusion (partitioned by sensor_name)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Partition Pruning (radar_ego_fusion)")
    print("=" * 70)
    results = []
    tbl = f"{CAT}.{NS_GOLD}.radar_ego_fusion"

    # Get distinct sensors
    sensors = [r.sensor_name for r in spark.sql(
        f"SELECT DISTINCT sensor_name FROM {tbl}"
    ).collect()]
    n_sensors = len(sensors)
    target = sensors[0]
    print(f"  {n_sensors} sensor partitions, target: {target}")

    # 2a: Full scan (all partitions)
    print(f"\n  [2a] Full scan — all {n_sensors} partitions")
    def full_scan():
        return spark.sql(f"SELECT count(*) as cnt FROM {tbl}")

    med_full, rows_full, times_full = _time_query(spark, full_scan)
    total_rows = spark.sql(f"SELECT count(*) as cnt FROM {tbl}").collect()[0]["cnt"]
    print(f"    {med_full:.3f}s, {total_rows:,} rows")
    results.append(QueryResult("Partition Pruning", "Full scan (all partitions)",
        f"Read all {n_sensors} sensor partitions", med_full, total_rows, 5, times_full))

    # 2b: Single partition (sensor_name = target)
    print(f"\n  [2b] Single-partition — sensor_name = '{target}'")
    def single_partition():
        return spark.sql(
            f"SELECT count(*) as cnt FROM {tbl} WHERE sensor_name = '{target}'"
        )

    med_single, rows_single, times_single = _time_query(spark, single_partition)
    single_cnt = spark.sql(
        f"SELECT count(*) as cnt FROM {tbl} WHERE sensor_name = '{target}'"
    ).collect()[0]["cnt"]
    pruning_pct = (1 - 1/n_sensors) * 100
    speedup = med_full / med_single if med_single > 0 else float("inf")
    print(f"    {med_single:.3f}s, {single_cnt:,} rows (pruning: {pruning_pct:.0f}%)")
    print(f"    Speedup vs full scan: {speedup:.1f}×")

    results.append(QueryResult("Partition Pruning", f"Single partition (sensor_name='{target}')",
        f"Reads 1/{n_sensors} partitions, {pruning_pct:.0f}% pruned", med_single, single_cnt, 5, times_single,
        {"speedup_vs_full": f"{speedup:.1f}x", "pct_pruned": f"{pruning_pct:.0f}%"}))

    # 2c: Two partitions
    if n_sensors >= 2:
        t1, t2 = sensors[0], sensors[1]
        print(f"\n  [2c] Two-partition — sensor_name IN ('{t1}', '{t2}')")
        def two_partition():
            return spark.sql(
                f"SELECT count(*) as cnt FROM {tbl} "
                f"WHERE sensor_name IN ('{t1}', '{t2}')"
            )

        med_two, _, times_two = _time_query(spark, two_partition)
        two_cnt = spark.sql(
            f"SELECT count(*) as cnt FROM {tbl} WHERE sensor_name IN ('{t1}', '{t2}')"
        ).collect()[0]["cnt"]
        speedup_two = med_full / med_two if med_two > 0 else float("inf")
        print(f"    {med_two:.3f}s, {two_cnt:,} rows")
        print(f"    Speedup vs full scan: {speedup_two:.1f}×")

        results.append(QueryResult("Partition Pruning", f"Two partitions",
            f"Reads 2/{n_sensors} partitions", med_two, two_cnt, 5, times_two,
            {"speedup_vs_full": f"{speedup_two:.1f}x"}))

    # 2d: Clip-level partition pruning on egomotion (partitioned by clip_id)
    print(f"\n  [2d] clip_id partition pruning — Silver egomotion")
    ego_tbl = f"{CAT}.{NS_SILVER}.egomotion"
    clip_id = spark.sql(f"SELECT DISTINCT clip_id FROM {ego_tbl} LIMIT 1").collect()[0]["clip_id"]

    def ego_full():
        return spark.sql(f"SELECT count(*) as cnt FROM {ego_tbl}")

    def ego_clip():
        return spark.sql(f"SELECT count(*) as cnt FROM {ego_tbl} WHERE clip_id = '{clip_id}'")

    med_ego_full, _, times_ego_full = _time_query(spark, ego_full)
    med_ego_clip, _, times_ego_clip = _time_query(spark, ego_clip)
    clip_cnt = spark.sql(f"SELECT count(*) as cnt FROM {ego_tbl} WHERE clip_id = '{clip_id}'").collect()[0]["cnt"]
    n_clips = spark.sql(f"SELECT count(DISTINCT clip_id) as cnt FROM {ego_tbl}").collect()[0]["cnt"]
    ego_speedup = med_ego_full / med_ego_clip if med_ego_clip > 0 else float("inf")

    print(f"    Full scan: {med_ego_full:.3f}s")
    print(f"    Single clip ({clip_id[:8]}...): {med_ego_clip:.3f}s, {clip_cnt:,} rows")
    print(f"    Speedup: {ego_speedup:.1f}× ({n_clips} clip partitions)")

    results.append(QueryResult("Partition Pruning", "Egomotion full scan",
        f"All {n_clips} clip partitions", med_ego_full, 0, 5, times_ego_full))
    results.append(QueryResult("Partition Pruning", "Egomotion single clip_id",
        f"1/{n_clips} partition filtered", med_ego_clip, clip_cnt, 5, times_ego_clip,
        {"speedup_vs_full": f"{ego_speedup:.1f}x", "n_clips": n_clips}))

    return results


# ──────────────────────────────────────────────────────────────
# Experiment 3: Timestamp Range Predicate (Metrics Pushdown)
# ──────────────────────────────────────────────────────────────
def exp3_timestamp_range(spark: SparkSession) -> List[QueryResult]:
    """Narrow timestamp range vs full scan — tests Iceberg column-level min/max metrics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Timestamp Range Predicate Pushdown")
    print("=" * 70)
    results = []
    tbl = f"{CAT}.{NS_GOLD}.radar_ego_fusion"

    # Get timestamp range
    stats = spark.sql(f"""
        SELECT min(timestamp) as ts_min, max(timestamp) as ts_max,
               count(*) as cnt
        FROM {tbl}
    """).collect()[0]
    ts_min, ts_max = stats["ts_min"], stats["ts_max"]
    ts_range = ts_max - ts_min
    # Narrow: first 10% of range
    ts_narrow = ts_min + ts_range // 10

    print(f"  Timestamp range: {ts_min} → {ts_max} (span: {ts_range:,})")
    print(f"  Narrow cutoff (10%): {ts_narrow}")

    # 3a: Full scan
    def ts_full():
        return spark.sql(f"SELECT count(*) as cnt FROM {tbl}")

    med_full, _, times_full = _time_query(spark, ts_full)
    print(f"\n  [3a] Full scan: {med_full:.3f}s, {stats['cnt']:,} rows")
    results.append(QueryResult("Timestamp Range", "Full scan (no filter)",
        "Scan all timestamps", med_full, stats["cnt"], 5, times_full))

    # 3b: Narrow range
    def ts_narrow_q():
        return spark.sql(
            f"SELECT count(*) as cnt FROM {tbl} WHERE timestamp < {ts_narrow}"
        )

    med_narrow, _, times_narrow = _time_query(spark, ts_narrow_q)
    narrow_cnt = spark.sql(
        f"SELECT count(*) as cnt FROM {tbl} WHERE timestamp < {ts_narrow}"
    ).collect()[0]["cnt"]
    speedup = med_full / med_narrow if med_narrow > 0 else float("inf")
    pct_rows = narrow_cnt / stats["cnt"] * 100

    print(f"  [3b] Narrow range (<10% span): {med_narrow:.3f}s, {narrow_cnt:,} rows ({pct_rows:.1f}%)")
    print(f"    Speedup: {speedup:.1f}×")

    results.append(QueryResult("Timestamp Range", f"Narrow range (first 10%)",
        f"timestamp < {ts_narrow} — {pct_rows:.1f}% of rows", med_narrow, narrow_cnt, 5, times_narrow,
        {"speedup_vs_full": f"{speedup:.1f}x", "pct_rows": f"{pct_rows:.1f}%"}))

    # 3c: Combined: timestamp + sensor_name partition
    sensor = spark.sql(f"SELECT DISTINCT sensor_name FROM {tbl} LIMIT 1").collect()[0]["sensor_name"]
    def ts_plus_partition():
        return spark.sql(f"""
            SELECT count(*) as cnt FROM {tbl}
            WHERE sensor_name = '{sensor}' AND timestamp < {ts_narrow}
        """)

    med_combo, _, times_combo = _time_query(spark, ts_plus_partition)
    combo_cnt = spark.sql(f"""
        SELECT count(*) as cnt FROM {tbl}
        WHERE sensor_name = '{sensor}' AND timestamp < {ts_narrow}
    """).collect()[0]["cnt"]
    speedup_combo = med_full / med_combo if med_combo > 0 else float("inf")

    print(f"  [3c] Combined (sensor + timestamp): {med_combo:.3f}s, {combo_cnt:,} rows")
    print(f"    Speedup: {speedup_combo:.1f}×")

    results.append(QueryResult("Timestamp Range", "Combined (partition + timestamp)",
        f"sensor_name='{sensor}' AND timestamp < narrow", med_combo, combo_cnt, 5, times_combo,
        {"speedup_vs_full": f"{speedup_combo:.1f}x"}))

    return results


# ──────────────────────────────────────────────────────────────
# Experiment 4: Aggregation Query Patterns
# ──────────────────────────────────────────────────────────────
def exp4_aggregation(spark: SparkSession) -> List[QueryResult]:
    """Common AD analytics aggregation patterns."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Aggregation Queries")
    print("=" * 70)
    results = []

    # 4a: Radar stats per sensor (GROUP BY on partition column — fast)
    print("\n  [4a] Radar stats grouped by sensor_name (partitioned)")
    tbl = f"{CAT}.{NS_GOLD}.radar_ego_fusion"

    def radar_per_sensor():
        return spark.sql(f"""
            SELECT sensor_name,
                   count(*) as n_returns,
                   avg(distance) as avg_dist,
                   avg(radial_velocity) as avg_vel,
                   min(timestamp) as first_ts,
                   max(timestamp) as last_ts
            FROM {tbl}
            GROUP BY sensor_name
        """)

    med_a, rows_a, times_a = _time_query(spark, radar_per_sensor)
    print(f"    {med_a:.3f}s, {rows_a} groups")
    results.append(QueryResult("Aggregation", "Radar stats per sensor (partitioned col)",
        "GROUP BY sensor_name on radar_ego_fusion", med_a, rows_a, 5, times_a))

    # 4b: Radar stats per clip (GROUP BY on non-partition column)
    print("\n  [4b] Radar stats grouped by clip_id (non-partition)")
    def radar_per_clip():
        return spark.sql(f"""
            SELECT clip_id,
                   count(*) as n_returns,
                   avg(distance) as avg_dist
            FROM {tbl}
            GROUP BY clip_id
        """)

    med_b, rows_b, times_b = _time_query(spark, radar_per_clip)
    print(f"    {med_b:.3f}s, {rows_b} groups")
    results.append(QueryResult("Aggregation", "Radar stats per clip (non-partition col)",
        "GROUP BY clip_id on radar_ego_fusion", med_b, rows_b, 5, times_b))

    # 4c: Ego velocity statistics per clip
    print("\n  [4c] Ego velocity stats per clip")
    ego_tbl = f"{CAT}.{NS_SILVER}.egomotion"
    def ego_stats():
        return spark.sql(f"""
            SELECT clip_id,
                   avg(vx) as avg_vx, avg(vy) as avg_vy, avg(vz) as avg_vz,
                   max(sqrt(vx*vx + vy*vy + vz*vz)) as max_speed,
                   avg(curvature) as avg_curvature
            FROM {ego_tbl}
            GROUP BY clip_id
        """)

    med_c, rows_c, times_c = _time_query(spark, ego_stats)
    print(f"    {med_c:.3f}s, {rows_c} groups")
    results.append(QueryResult("Aggregation", "Ego velocity stats per clip",
        "GROUP BY clip_id on Silver egomotion", med_c, rows_c, 5, times_c))

    # 4d: Sensor coverage summary from Gold fusion table
    print("\n  [4d] sensor_fusion_clip wide read (all columns)")
    fusion_tbl = f"{CAT}.{NS_GOLD}.sensor_fusion_clip"
    def fusion_read():
        return spark.sql(f"SELECT count(*) as cnt FROM {fusion_tbl}")

    med_d, _, times_d = _time_query(spark, fusion_read)
    fusion_cnt = spark.sql(f"SELECT count(*) as cnt FROM {fusion_tbl}").collect()[0]["cnt"]
    print(f"    {med_d:.3f}s, {fusion_cnt:,} rows")
    results.append(QueryResult("Aggregation", "sensor_fusion_clip count",
        "Count on wide Gold table (35+ columns)", med_d, fusion_cnt, 5, times_d))

    return results


# ──────────────────────────────────────────────────────────────
# Experiment 5: Time Travel
# ──────────────────────────────────────────────────────────────
def exp5_time_travel(spark: SparkSession) -> List[QueryResult]:
    """Measure time-travel read latency via Iceberg snapshot."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Time Travel (Snapshot Read)")
    print("=" * 70)
    results = []
    tbl = f"{CAT}.{NS_GOLD}.radar_ego_fusion"

    # Get snapshots
    snaps = spark.sql(f"SELECT * FROM {tbl}.snapshots ORDER BY committed_at").collect()
    if len(snaps) < 2:
        print("  Only 1 snapshot — skipping time travel comparison.")
        return results

    oldest = snaps[0]
    latest = snaps[-1]
    print(f"  Oldest snapshot: {oldest.snapshot_id} ({oldest.committed_at})")
    print(f"  Latest snapshot: {latest.snapshot_id} ({latest.committed_at})")

    # 5a: Current read
    def current_read():
        return spark.sql(f"SELECT count(*) as cnt FROM {tbl}")

    med_cur, _, times_cur = _time_query(spark, current_read)
    cur_cnt = spark.sql(f"SELECT count(*) as cnt FROM {tbl}").collect()[0]["cnt"]
    print(f"\n  [5a] Current read: {med_cur:.3f}s, {cur_cnt:,} rows")
    results.append(QueryResult("Time Travel", "Current snapshot read",
        f"Latest snapshot {latest.snapshot_id}", med_cur, cur_cnt, 5, times_cur))

    # 5b: Historical read
    snap_id = oldest.snapshot_id
    def historical_read():
        return spark.sql(
            f"SELECT count(*) as cnt FROM {tbl} VERSION AS OF {snap_id}"
        )

    med_hist, _, times_hist = _time_query(spark, historical_read)
    hist_cnt = spark.sql(
        f"SELECT count(*) as cnt FROM {tbl} VERSION AS OF {snap_id}"
    ).collect()[0]["cnt"]
    overhead = med_hist - med_cur
    print(f"  [5b] Historical read (snapshot {snap_id}): {med_hist:.3f}s, {hist_cnt:,} rows")
    print(f"    Overhead vs current: {overhead:+.3f}s")

    results.append(QueryResult("Time Travel", "Historical snapshot read",
        f"VERSION AS OF {snap_id}", med_hist, hist_cnt, 5, times_hist,
        {"overhead_s": f"{overhead:+.3f}"}))

    return results


# ──────────────────────────────────────────────────────────────
# Experiment 6: Bronze vs Silver vs Gold Read Latency
# ──────────────────────────────────────────────────────────────
def exp6_tier_read_latency(spark: SparkSession) -> List[QueryResult]:
    """Direct comparison: how fast is a simple count across tiers?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Bronze vs Silver vs Gold Count Latency")
    print("=" * 70)
    results = []

    # Pick a radar table that exists in all three tiers
    bronze_t = f"{CAT}.{NS_BRONZE}.radar_radar_front_center_imaging_lrr_1"
    silver_t = f"{CAT}.{NS_SILVER}.radar_radar_front_center_imaging_lrr_1"
    gold_t   = f"{CAT}.{NS_GOLD}.radar_ego_fusion"

    for label, table, extra_filter in [
        ("Bronze (single radar table)", bronze_t, ""),
        ("Silver (single radar table, partitioned)", silver_t, ""),
        ("Gold (radar_ego_fusion, all sensors)", gold_t, ""),
        ("Gold (radar_ego_fusion, single sensor)", gold_t,
         " WHERE sensor_name = 'radar_front_center_imaging_lrr_1'"),
    ]:
        def mk_q(t=table, f=extra_filter):
            return lambda: spark.sql(f"SELECT count(*) as cnt FROM {t}{f}")

        med, _, times = _time_query(spark, mk_q())
        cnt = spark.sql(f"SELECT count(*) as cnt FROM {table}{extra_filter}").collect()[0]["cnt"]
        print(f"  {label}: {med:.3f}s, {cnt:,} rows")
        results.append(QueryResult("Tier Read Latency", label,
            f"SELECT count(*) FROM {table}{extra_filter}", med, cnt, 5, times))

    return results


def run_all() -> List[QueryResult]:
    spark = _build_spark()
    _warmup_jvm(spark)

    all_results: List[QueryResult] = []
    all_results.extend(exp1_gold_vs_silver_join(spark))
    all_results.extend(exp2_partition_pruning(spark))
    all_results.extend(exp3_timestamp_range(spark))
    all_results.extend(exp4_aggregation(spark))
    all_results.extend(exp5_time_travel(spark))
    all_results.extend(exp6_tier_read_latency(spark))

    spark.stop()

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("QUERY BENCHMARK SUMMARY")
    print("=" * 70)

    current_exp = ""
    for r in all_results:
        if r.experiment != current_exp:
            current_exp = r.experiment
            print(f"\n  [{current_exp}]")
        print(f"    {r.variant:50s} {r.median_s:>8.3f}s  {r.row_count:>12,} rows"
              + (f"  {list(r.extra.values())[0]}" if r.extra else ""))

    # --- Save JSON ---
    with open(OUTPUT_PATH, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return all_results


if __name__ == "__main__":
    run_all()
