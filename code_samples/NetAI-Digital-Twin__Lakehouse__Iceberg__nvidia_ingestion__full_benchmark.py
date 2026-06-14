#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Nvidia PhysicalAI Lakehouse Ingestion.

Pipeline (FUSE-based zero-copy):
  1. Bronze — register FUSE-exposed parquets via add_files() (zero-copy, metadata only)
  2. Silver — create SQL views over Bronze (clip_id derived at query time)
  3. Gold — pre-joined tables (materialized if overhead <20%, else views)
  4. Query benchmarks across tiers (raw NFS vs Bronze/Silver vs Gold)

Measures:
  - Per-phase wall clock, throughput, memory (RSS sampled via background thread)
  - Query latency across tiers with memory tracking
  - Gold storage overhead vs raw data
  - Partition pruning and predicate pushdown effectiveness

Prerequisites:
  - ratarmount FUSE mounts active at /tmp/nvidia-fuse
  - NFS dataset at /mnt/netai-e2e/nvidia-physicalai-av-subset

Usage (inside spark-iceberg container):
    python -u -m nvidia_ingestion.full_benchmark
"""

import gc
import glob as globmod
import json
import os
import resource
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS", "--driver-memory 32g pyspark-shell"
)

FUSE_ROOT = "/tmp/nvidia-fuse"


# ── Memory Sampler ─────────────────────────────────────────────────────
class MemorySampler:
    """Background thread that samples RSS every `interval` seconds."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self):
        self._t0 = time.time()
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Tuple[float, float]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return list(self.samples)

    def _run(self):
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            self.samples.append((round(elapsed, 1), round(rss, 1)))
            self._stop.wait(self.interval)

    def peak_mb(self) -> float:
        return max((s[1] for s in self.samples), default=0.0)


@dataclass
class PhaseResult:
    phase: str
    table: str
    wall_s: float
    rows: int
    bytes_in: int = 0
    throughput_rows_per_s: float = 0.0
    throughput_mb_per_s: float = 0.0
    peak_rss_mb: float = 0.0
    memory_samples: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class QueryBenchResult:
    name: str
    tier: str
    description: str
    median_s: float
    row_count: int
    runs: int
    all_times_s: List[float] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Camera symlink helper ─────────────────────────────────────────────
def _prepare_camera_symlinks(fuse_root: str) -> Dict[str, str]:
    """Create symlink directories that separate camera parquet types.

    add_files() reads all parquets in a directory, but camera FUSE dirs
    contain both .timestamps.parquet and .blurred_boxes.parquet with
    different schemas. This creates filtered symlink dirs.

    Returns a dict mapping table_name -> symlink_dir_path.
    """
    cam_fuse = os.path.join(fuse_root, "camera")
    if not os.path.isdir(cam_fuse):
        return {}

    filter_root = "/tmp/nvidia-fuse-filtered/camera"
    result = {}

    for sensor in sorted(os.listdir(cam_fuse)):
        sensor_dir = os.path.join(cam_fuse, sensor)
        if not os.path.isdir(sensor_dir):
            continue

        safe = sensor.replace("-", "_")

        for suffix, label in [
            (".timestamps.parquet", f"cam_{safe}_ts"),
            (".blurred_boxes.parquet", f"cam_{safe}_blur"),
        ]:
            out_dir = os.path.join(filter_root, label)
            os.makedirs(out_dir, exist_ok=True)

            # Clear old symlinks
            for old in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, old))

            # Create symlinks from all chunk subdirs
            for chunk_dir in sorted(globmod.glob(os.path.join(sensor_dir, "*.zip"))):
                if not os.path.isdir(chunk_dir):
                    continue
                for f in globmod.glob(os.path.join(chunk_dir, f"*{suffix}")):
                    link = os.path.join(out_dir, os.path.basename(f))
                    if not os.path.exists(link):
                        os.symlink(f, link)

            count = len(os.listdir(out_dir))
            if count > 0:
                result[label] = out_dir
                print(f"  [SYMLINK] {label}: {count} files -> {out_dir}")

    return result


# ── Ingestion ─────────────────────────────────────────────────────────
def run_ingestion_with_benchmarks(
    fuse_root: str = FUSE_ROOT,
) -> Tuple[List[PhaseResult], Dict]:
    """Run Bronze-register → Silver-views → Gold with benchmarking."""
    from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces
    from .benchmark import BenchmarkTracker

    cfg = NvidiaPipelineConfig()
    cfg.nvidia.silver_mode = "view"
    cfg.nvidia.gold_mode = "materialized"

    print(f"\n{'='*70}")
    print("FULL BENCHMARK: FUSE → Bronze (add_files) → Silver (views) → Gold")
    print(f"  source     = {cfg.nvidia.source_path}")
    print(f"  fuse_root  = {fuse_root}")
    print(f"  driver_mem = {cfg.nvidia.driver_memory}")
    print(f"{'='*70}\n")

    results: List[PhaseResult] = []
    overall_sampler = MemorySampler(interval=10.0)
    overall_sampler.start()
    pipeline_t0 = time.time()

    # ── PHASE 1: BRONZE REGISTRATION (add_files, zero-copy) ──────
    print("[PHASE 1/3] BRONZE REGISTRATION — add_files() zero-copy from FUSE")
    print("-" * 50)
    tracker_b = BenchmarkTracker("full-benchmark-bronze-register")

    spark = build_spark_session(cfg, app_name="nvidia-full-benchmark")
    create_namespaces(spark, cfg)

    bronze_sampler = MemorySampler(interval=5.0)
    bronze_sampler.start()
    bronze_t0 = time.time()

    from .register_bronze import BronzeRegistrar

    registrar = BronzeRegistrar(spark, cfg, tracker=tracker_b)
    bronze_results: Dict[str, int] = {}

    # Bare parquets (metadata, clip_index) — directly on NFS
    bronze_results.update(registrar.register_bare_parquets(mode="fuse"))

    # Calibration directories — directly on NFS
    bronze_results.update(registrar.register_calibration(mode="fuse"))

    # Egomotion — from FUSE mount
    bronze_results["egomotion"] = registrar.register_fuse_sensor(
        "egomotion", "labels/egomotion", fuse_root=fuse_root,
    )

    # Lidar — from FUSE mount
    bronze_results["lidar"] = registrar.register_fuse_sensor(
        "lidar", "lidar/lidar_top_360fov", fuse_root=fuse_root,
    )

    # Radar — discover all sensors from FUSE mount
    radar_fuse = os.path.join(fuse_root, "radar")
    if os.path.isdir(radar_fuse):
        for sensor in sorted(os.listdir(radar_fuse)):
            sensor_path = os.path.join(radar_fuse, sensor)
            if os.path.isdir(sensor_path):
                safe = sensor.replace("-", "_")
                table_name = f"radar_{safe}"
                bronze_results[table_name] = registrar.register_fuse_sensor(
                    table_name, f"radar/{sensor}", fuse_root=fuse_root,
                )

    # Camera metadata — needs symlink filtering
    print("\n[CAMERA] Preparing symlink directories for schema separation...")
    cam_tables = _prepare_camera_symlinks(fuse_root)
    for table_name, symlink_dir in cam_tables.items():
        print(f"[REGISTER] {table_name}")
        if tracker_b:
            tracker_b.begin("bronze", table_name)
        full = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_bronze}.{table_name}"
        uri = f"file://{symlink_dir}"
        try:
            df = spark.read.parquet(uri)
            bytes_in = 0  # skip per-file stat on FUSE symlinks (slow over NFS)
            df.limit(0).writeTo(full).using("iceberg").tableProperty(
                "format-version", "2"
            ).createOrReplace()
            spark.sql(
                f"CALL {cfg.spark_catalog_name}.system.add_files("
                f"  table => '{full}',"
                f"  source_table => '`parquet`.`{uri}`'"
                f")"
            )
            rows = spark.table(full).count()
            bronze_results[table_name] = rows
            if tracker_b:
                tracker_b.end(rows_out=rows, bytes_in=bytes_in)
            print(f"  [DONE] {full}: {rows:,} rows")
        except Exception as e:
            print(f"  [ERROR] {table_name}: {e}")
            bronze_results[table_name] = 0
            if tracker_b:
                tracker_b.end()

    bronze_wall = time.time() - bronze_t0
    bronze_mem = bronze_sampler.stop()

    for step in tracker_b.steps:
        results.append(PhaseResult(
            phase="bronze_register", table=step.table,
            wall_s=step.wall_s, rows=step.rows_out,
            bytes_in=step.bytes_in,
            peak_rss_mb=step.peak_rss_mb,
        ))

    results.append(PhaseResult(
        phase="bronze_register_total", table="ALL",
        wall_s=round(bronze_wall, 2),
        rows=sum(v for v in bronze_results.values() if v > 0),
        peak_rss_mb=bronze_sampler.peak_mb(),
        memory_samples=bronze_mem,
    ))

    print(f"\nBronze registration total: {bronze_wall:.1f}s")
    for tbl, cnt in bronze_results.items():
        print(f"  {tbl}: {cnt:,} rows")

    gc.collect()

    # ── PHASE 2: SILVER VIEWS ────────────────────────────────────
    print(f"\n[PHASE 2/3] SILVER VIEWS — SQL views over Bronze (zero-copy)")
    print("-" * 50)

    silver_sampler = MemorySampler(interval=5.0)
    silver_sampler.start()
    silver_t0 = time.time()

    from .transform_silver import NvidiaSilverTransformer
    cfg.nvidia.silver_mode = "view"
    view_transformer = NvidiaSilverTransformer(spark, cfg)
    silver_results = view_transformer.transform_all()

    silver_wall = time.time() - silver_t0
    silver_mem = silver_sampler.stop()

    results.append(PhaseResult(
        phase="silver_views_total", table="ALL",
        wall_s=round(silver_wall, 2),
        rows=sum(v for v in silver_results.values() if isinstance(v, int) and v > 0),
        peak_rss_mb=silver_sampler.peak_mb(),
        memory_samples=silver_mem,
    ))

    print(f"\nSilver views total: {silver_wall:.1f}s")
    for tbl, cnt in silver_results.items():
        print(f"  {tbl}: {cnt:,} rows")

    gc.collect()

    # ── PHASE 3: GOLD CONSTRUCTION ─────────────────────────────────
    print(f"\n[PHASE 3/3] GOLD CONSTRUCTION")
    print("-" * 50)
    tracker_g = BenchmarkTracker("full-benchmark-gold")

    gold_sampler = MemorySampler(interval=5.0)
    gold_sampler.start()
    gold_t0 = time.time()

    from .build_gold import NvidiaGoldBuilder
    cfg.nvidia.gold_mode = "materialized"
    builder = NvidiaGoldBuilder(spark, cfg, tracker=tracker_g)
    gold_results = builder.build_all()
    gold_wall = time.time() - gold_t0
    gold_mem = gold_sampler.stop()

    for step in tracker_g.steps:
        results.append(PhaseResult(
            phase="gold", table=step.table,
            wall_s=step.wall_s, rows=step.rows_out,
            peak_rss_mb=step.peak_rss_mb,
        ))

    results.append(PhaseResult(
        phase="gold_total", table="ALL",
        wall_s=round(gold_wall, 2),
        rows=sum(v for v in gold_results.values() if v > 0),
        peak_rss_mb=gold_sampler.peak_mb(),
        memory_samples=gold_mem,
    ))

    # ── GOLD STORAGE OVERHEAD CHECK ────────────────────────────────
    print(f"\n[OVERHEAD CHECK] Measuring Gold storage vs raw data...")
    try:
        gold_bytes = 0
        for tbl_name in gold_results:
            if gold_results[tbl_name] <= 0:
                continue
            try:
                tbl_full = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.{tbl_name}"
                files = spark.sql(
                    f"SELECT file_path, file_size_in_bytes FROM {tbl_full}.files"
                ).collect()
                tbl_bytes = sum(f.file_size_in_bytes for f in files)
                gold_bytes += tbl_bytes
                print(f"    {tbl_name}: {tbl_bytes / 1073741824:.2f} GB")
            except Exception as e:
                print(f"    {tbl_name}: could not measure ({e})")

        # Raw dataset size
        raw_bytes = 0
        for subdir in ["lidar/lidar_top_360fov", "labels/egomotion", "radar",
                       "camera", "calibration", "metadata"]:
            full_path = os.path.join(cfg.nvidia.source_path, subdir)
            if os.path.exists(full_path):
                for root, dirs, files in os.walk(full_path):
                    for f in files:
                        try:
                            raw_bytes += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass

        if raw_bytes > 0:
            overhead_pct = (gold_bytes / raw_bytes) * 100
            print(f"\n  Gold storage: {gold_bytes / 1073741824:.2f} GB")
            print(f"  Raw data:     {raw_bytes / 1073741824:.2f} GB")
            print(f"  Overhead:     {overhead_pct:.1f}%")

            if overhead_pct > 20:
                print(f"\n  [PIVOT] Overhead ({overhead_pct:.1f}%) > 20% — rebuilding Gold as VIEWS")
                cfg.nvidia.gold_mode = "view"
                builder_v = NvidiaGoldBuilder(spark, cfg)
                gold_results_v = builder_v.build_all()
                print(f"  Gold views created: {gold_results_v}")
                results.append(PhaseResult(
                    phase="gold_pivot_to_views", table="ALL",
                    wall_s=0, rows=sum(v for v in gold_results_v.values() if v > 0),
                ))
            else:
                print(f"  [OK] Overhead {overhead_pct:.1f}% <= 20% — keeping materialized Gold")
        else:
            print("  Could not measure raw data size")
    except Exception as e:
        print(f"  Overhead check failed: {e}")

    overall_mem = overall_sampler.stop()
    pipeline_wall = time.time() - pipeline_t0

    spark.stop()

    summary = {
        "total_pipeline_s": round(pipeline_wall, 2),
        "bronze_register_s": round(bronze_wall, 2),
        "silver_views_s": round(silver_wall, 2),
        "gold_s": round(gold_wall, 2),
        "peak_rss_mb": max(
            bronze_sampler.peak_mb(),
            silver_sampler.peak_mb(),
            gold_sampler.peak_mb(),
        ),
        "overall_memory_samples": overall_mem,
    }

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE: {pipeline_wall:.1f}s total")
    print(f"  Bronze register: {bronze_wall:.1f}s | Silver views: {silver_wall:.1f}s | Gold: {gold_wall:.1f}s")
    print(f"  Peak RSS: {summary['peak_rss_mb']:.0f} MB")
    print(f"{'='*70}")

    return results, summary


# ── Query benchmarks ──────────────────────────────────────────────────
def _time_query(spark, fn: Callable, warmup: int = 1, runs: int = 5,
                use_count: bool = False) -> Tuple[float, int, List[float]]:
    for _ in range(warmup):
        df = fn()
        df.count() if use_count else df.collect()

    times = []
    row_count = 0
    for _ in range(runs):
        t0 = time.time()
        df = fn()
        if use_count:
            row_count = df.count()
        else:
            rows = df.collect()
            row_count = len(rows)
        times.append(time.time() - t0)

    times.sort()
    return times[len(times) // 2], row_count, times


def run_query_benchmarks(fuse_root: str = FUSE_ROOT) -> List[QueryBenchResult]:
    """Run query latency benchmarks across all tiers."""
    from .config import NvidiaPipelineConfig, build_spark_session

    cfg = NvidiaPipelineConfig()
    spark = build_spark_session(cfg, "nvidia-query-benchmark")

    CAT = cfg.spark_catalog_name
    NS_B = cfg.nvidia.namespace_bronze
    NS_S = cfg.nvidia.namespace_silver
    NS_G = cfg.nvidia.namespace_gold

    results: List[QueryBenchResult] = []

    print("\n[QUERY BENCHMARKS] Warming JVM...")
    for _ in range(3):
        try:
            spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.clip_index").collect()
        except Exception:
            pass
    print("  JVM warm.\n")

    # ── 1. Raw NFS vs Bronze vs Silver (egomotion) ────────────────
    print("=" * 70)
    print("BENCH 1: Raw NFS vs Bronze vs Silver (egomotion)")
    print("=" * 70)

    # 1a: Raw NFS — Spark reads parquets directly from FUSE mount
    ego_fuse = os.path.join(fuse_root, "labels", "egomotion")
    if os.path.isdir(ego_fuse):
        chunk_dirs = sorted(
            d for d in globmod.glob(os.path.join(ego_fuse, "*.zip"))
            if os.path.isdir(d)
        )[:10]  # Sample 10 chunks for raw NFS test
        if chunk_dirs:
            def raw_ego():
                return spark.read.parquet(*[f"file://{d}" for d in chunk_dirs])

            sampler = MemorySampler(interval=2.0)
            sampler.start()
            try:
                med, rows, times = _time_query(spark, raw_ego, use_count=True)
                sampler.stop()
                print(f"  Raw NFS (10 dirs): {med:.3f}s, {rows:,} rows, RSS {sampler.peak_mb():.0f} MB")
                results.append(QueryBenchResult(
                    "Raw NFS egomotion (10 dirs)", "raw",
                    "Direct Spark read of FUSE-exposed parquets",
                    med, rows, 5, times, sampler.peak_mb()))
            except Exception as e:
                sampler.stop()
                print(f"  Raw NFS: FAILED ({e})")

    # 1b: Bronze (Iceberg via add_files)
    def bronze_ego():
        return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.egomotion")
    try:
        sampler = MemorySampler(interval=2.0)
        sampler.start()
        med, rows, times = _time_query(spark, bronze_ego)
        sampler.stop()
        cnt = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.egomotion").collect()[0][0]
        print(f"  Bronze Iceberg: {med:.3f}s, {cnt:,} rows, RSS {sampler.peak_mb():.0f} MB")
        results.append(QueryBenchResult(
            "Bronze egomotion count", "bronze",
            "Iceberg table via add_files (FUSE data)", med, cnt, 5, times, sampler.peak_mb()))
    except Exception as e:
        print(f"  Bronze: FAILED ({e})")

    # 1c: Silver view
    def silver_ego():
        return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_S}.egomotion")
    try:
        sampler = MemorySampler(interval=2.0)
        sampler.start()
        med, rows, times = _time_query(spark, silver_ego)
        sampler.stop()
        cnt = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_S}.egomotion").collect()[0][0]
        print(f"  Silver view:  {med:.3f}s, {cnt:,} rows, RSS {sampler.peak_mb():.0f} MB")
        results.append(QueryBenchResult(
            "Silver egomotion count", "silver",
            "View over Bronze (+ clip_id)", med, cnt, 5, times, sampler.peak_mb()))
    except Exception as e:
        print(f"  Silver: FAILED ({e})")

    # ── 2. Gold vs ad-hoc join ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCH 2: Gold Materialized vs Ad-Hoc Silver Join")
    print("=" * 70)

    lidar_tbl = "lidar_decoded" if cfg.nvidia.lidar_mode == "decoded" else "lidar"

    # Gold lidar_with_ego
    med_g = 0
    try:
        def gold_lidar():
            return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego")
        med_g, _, times_g = _time_query(spark, gold_lidar)
        cnt_g = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego").collect()[0][0]
        print(f"  Gold lidar_with_ego: {med_g:.3f}s, {cnt_g:,} rows")
        results.append(QueryBenchResult(
            "Gold lidar_with_ego count", "gold",
            "Pre-materialized table", med_g, cnt_g, 5, times_g))
    except Exception as e:
        print(f"  Gold lidar_with_ego: FAILED ({e})")

    # Silver equivalent join
    try:
        def silver_lidar_join():
            return spark.sql(f"""
                SELECT count(*) FROM (
                    SELECT l.clip_id, l.reference_timestamp, e.timestamp as ego_ts
                    FROM {CAT}.{NS_S}.{lidar_tbl} l
                    LEFT JOIN {CAT}.{NS_S}.egomotion e ON l.clip_id = e.clip_id
                )
            """)
        med_s, _, times_s = _time_query(spark, silver_lidar_join)
        speedup = med_s / med_g if med_g > 0 else float("inf")
        print(f"  Silver lidar+ego join: {med_s:.3f}s (speedup: {speedup:.1f}x)")
        results.append(QueryBenchResult(
            "Silver lidar+ego join count", "silver",
            "Ad-hoc join", med_s, 0, 5, times_s,
            extra={"speedup_vs_gold": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Silver join: FAILED ({e})")

    # ── 3. Partition pruning on radar_ego_fusion ───────────────────
    print(f"\n{'='*70}")
    print("BENCH 3: Partition Pruning")
    print("=" * 70)
    try:
        tbl = f"{CAT}.{NS_G}.radar_ego_fusion"
        sensors = [r.sensor_name for r in spark.sql(
            f"SELECT DISTINCT sensor_name FROM {tbl}"
        ).collect()]

        def full_scan():
            return spark.sql(f"SELECT count(*) FROM {tbl}")
        med_full, _, times_full = _time_query(spark, full_scan)
        total = spark.sql(f"SELECT count(*) FROM {tbl}").collect()[0][0]
        print(f"  Full scan ({len(sensors)} partitions): {med_full:.3f}s, {total:,} rows")
        results.append(QueryBenchResult(
            "Radar full scan", "gold", f"All {len(sensors)} partitions",
            med_full, total, 5, times_full))

        target = sensors[0]
        def single_part():
            return spark.sql(f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{target}'")
        med_single, _, times_single = _time_query(spark, single_part)
        single_cnt = spark.sql(
            f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{target}'"
        ).collect()[0][0]
        speedup = med_full / med_single if med_single > 0 else float("inf")
        print(f"  Single partition: {med_single:.3f}s, {single_cnt:,} rows (speedup: {speedup:.1f}x)")
        results.append(QueryBenchResult(
            "Radar single partition", "gold",
            f"Pruned to 1/{len(sensors)}", med_single, single_cnt, 5, times_single,
            extra={"speedup_vs_full": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Partition pruning: FAILED ({e})")

    # ── 4. Aggregation ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCH 4: Aggregation Patterns")
    print("=" * 70)
    agg_queries = [
        ("Radar stats by sensor (partitioned)", "gold",
         f"SELECT sensor_name, count(*), avg(distance) FROM {CAT}.{NS_G}.radar_ego_fusion GROUP BY sensor_name"),
        ("Radar stats by clip (non-partition)", "gold",
         f"SELECT clip_id, count(*), avg(distance) FROM {CAT}.{NS_G}.radar_ego_fusion GROUP BY clip_id"),
        ("Clip summary by split", "gold",
         f"SELECT split, count(*) FROM {CAT}.{NS_G}.sensor_fusion_clip GROUP BY split"),
    ]
    for name, tier, sql in agg_queries:
        try:
            def mk_q(s=sql):
                return lambda: spark.sql(s)
            med, rows, times = _time_query(spark, mk_q())
            print(f"  {name}: {med:.3f}s, {rows} groups")
            results.append(QueryBenchResult(name, tier, sql[:80], med, rows, 5, times))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── 5. Tier comparison ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BENCH 5: Same Logical Query Across Tiers")
    print("=" * 70)
    try:
        bronze_tables = [r[1] for r in spark.sql(f"SHOW TABLES IN {CAT}.{NS_B}").collect()]
        radar_t = next((t for t in bronze_tables if t.startswith("radar_")), None)
        if radar_t:
            sensor_name = radar_t.replace("radar_", "", 1)
            tier_queries = [
                ("Bronze", f"SELECT count(*) FROM {CAT}.{NS_B}.{radar_t}"),
                ("Silver", f"SELECT count(*) FROM {CAT}.{NS_S}.{radar_t}"),
                ("Gold (filtered)", f"SELECT count(*) FROM {CAT}.{NS_G}.radar_ego_fusion WHERE sensor_name = '{sensor_name}'"),
            ]
            for label, sql in tier_queries:
                def mk_q(s=sql):
                    return lambda: spark.sql(s)
                try:
                    med, _, times = _time_query(spark, mk_q())
                    cnt = spark.sql(sql).collect()[0][0]
                    print(f"  {label} ({radar_t}): {med:.3f}s, {cnt:,} rows")
                    results.append(QueryBenchResult(
                        f"{label} {radar_t}", label.split()[0].lower(),
                        sql[:80], med, cnt, 5, times))
                except Exception as e:
                    print(f"  {label}: FAILED ({e})")
    except Exception as e:
        print(f"  Tier comparison: FAILED ({e})")

    # ── 6. Full table scan latency by data volume ──────────────────
    print(f"\n{'='*70}")
    print("BENCH 6: Full Table Scan Latency (small → large)")
    print("=" * 70)
    scan_tables = [
        ("clip_index", "bronze", f"{CAT}.{NS_B}.clip_index"),
        ("egomotion", "bronze", f"{CAT}.{NS_B}.egomotion"),
        ("lidar", "bronze", f"{CAT}.{NS_B}.lidar"),
    ]
    for name, tier, tbl in scan_tables:
        try:
            def mk_q(t=tbl):
                return lambda: spark.sql(f"SELECT count(*) FROM {t}")
            sampler = MemorySampler(interval=2.0)
            sampler.start()
            med, _, times = _time_query(spark, mk_q())
            sampler.stop()
            cnt = spark.sql(f"SELECT count(*) FROM {tbl}").collect()[0][0]
            print(f"  {name}: {med:.3f}s, {cnt:,} rows, RSS {sampler.peak_mb():.0f} MB")
            results.append(QueryBenchResult(
                f"Full scan {name}", tier,
                f"count(*) on {name}", med, cnt, 5, times, sampler.peak_mb()))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    spark.stop()
    return results


# ── Main ───────────────────────────────────────────────────────────────
def main():
    output_path = os.environ.get("BENCHMARK_OUTPUT", "/tmp/nvidia_full_benchmark.json")

    ingestion_results, pipeline_summary = run_ingestion_with_benchmarks()
    query_results = run_query_benchmarks()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": pipeline_summary,
        "ingestion_phases": [asdict(r) for r in ingestion_results],
        "query_benchmarks": [asdict(r) for r in query_results],
    }

    print(f"\n{'='*70}")
    print("INGESTION PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Phase':<25} {'Table':<30} {'Wall(s)':>10} {'Rows':>12} {'RSS(MB)':>9}")
    print("-" * 90)
    for r in ingestion_results:
        if r.table == "ALL":
            print(f"{'>>> '+r.phase:<25} {'TOTAL':<30} {r.wall_s:>10.1f} "
                  f"{r.rows:>12,} {r.peak_rss_mb:>9.0f}")
        else:
            print(f"{r.phase:<25} {r.table:<30} {r.wall_s:>10.1f} "
                  f"{r.rows:>12,} {r.peak_rss_mb:>9.0f}")

    print(f"\n{'='*70}")
    print("QUERY LATENCY BENCHMARKS")
    print(f"{'='*70}")
    print(f"{'Name':<45} {'Tier':<8} {'Median(s)':>10} {'Rows':>12}")
    print("-" * 80)
    for r in query_results:
        print(f"{r.name:<45} {r.tier:<8} {r.median_s:>10.3f} {r.row_count:>12,}")
        if r.extra:
            for k, v in r.extra.items():
                print(f"  {'':45} {k}: {v}")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
