#!/usr/bin/env python3
"""
Continuation benchmark: picks up where full_benchmark.py left off.

Already done:
  - clip_index, data_collection, sensor_presence (bare parquets)
  - camera_intrinsics, sensor_extrinsics, vehicle_dimensions (calibration)
  - egomotion (101M rows, 340 dirs)
  - lidar (partial: ~33/340 dirs, 1M rows)

Remaining:
  1. Radar (19 sensors, small files — fast via batched symlinks)
  2. Camera (14 tables via symlink filtering)
  3. Lidar resume (append dirs 33+ to existing table)
  4. Silver views
  5. Gold tables
  6. Query benchmarks
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

os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 32g pyspark-shell")

FUSE_ROOT = "/tmp/nvidia-fuse"


class MemorySampler:
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


def main():
    from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces
    from .register_bronze import BronzeRegistrar
    from .benchmark import BenchmarkTracker

    output_path = "/tmp/nvidia_continuation_benchmark.json"

    cfg = NvidiaPipelineConfig()
    cfg.nvidia.silver_mode = "view"
    cfg.nvidia.gold_mode = "materialized"

    results: List[PhaseResult] = []
    bronze_results: Dict[str, int] = {}
    overall_sampler = MemorySampler(interval=10.0)
    overall_sampler.start()
    pipeline_t0 = time.time()

    print(f"\n{'='*70}")
    print("CONTINUATION BENCHMARK: Radar + Camera + Lidar Resume + Silver + Gold")
    print(f"{'='*70}\n")

    spark = build_spark_session(cfg, app_name="nvidia-continuation-benchmark")
    create_namespaces(spark, cfg)

    CAT = cfg.spark_catalog_name
    NS_B = cfg.nvidia.namespace_bronze
    NS_S = cfg.nvidia.namespace_silver
    NS_G = cfg.nvidia.namespace_gold

    tracker = BenchmarkTracker("continuation-bronze")
    registrar = BronzeRegistrar(spark, cfg, tracker=tracker)

    # ── RADAR REGISTRATION (19 sensors, small files) ─────────────────
    print("[PHASE 1] RADAR REGISTRATION")
    print("-" * 50)
    radar_t0 = time.time()

    radar_fuse = os.path.join(FUSE_ROOT, "radar")
    if os.path.isdir(radar_fuse):
        for sensor in sorted(os.listdir(radar_fuse)):
            sensor_path = os.path.join(radar_fuse, sensor)
            if os.path.isdir(sensor_path):
                safe = sensor.replace("-", "_")
                table_name = f"radar_{safe}"
                try:
                    rows = registrar.register_fuse_sensor(
                        table_name, f"radar/{sensor}", fuse_root=FUSE_ROOT,
                    )
                    bronze_results[table_name] = rows
                except Exception as e:
                    print(f"  [ERROR] {table_name}: {e}")
                    bronze_results[table_name] = 0

    radar_wall = time.time() - radar_t0
    results.append(PhaseResult(
        phase="radar_registration", table="ALL",
        wall_s=round(radar_wall, 2),
        rows=sum(v for v in bronze_results.values() if v > 0),
    ))
    print(f"\nRadar registration: {radar_wall:.1f}s")
    for t, c in bronze_results.items():
        print(f"  {t}: {c:,} rows")

    gc.collect()

    # ── CAMERA REGISTRATION (14 tables via symlink filtering) ────────
    print(f"\n[PHASE 2] CAMERA REGISTRATION")
    print("-" * 50)
    camera_t0 = time.time()
    cam_results: Dict[str, int] = {}

    cam_fuse = os.path.join(FUSE_ROOT, "camera")
    if os.path.isdir(cam_fuse):
        for sensor in sorted(os.listdir(cam_fuse)):
            sensor_dir = os.path.join(cam_fuse, sensor)
            if not os.path.isdir(sensor_dir):
                continue

            safe = sensor.replace("-", "_")

            for suffix, label in [
                (".timestamps.parquet", f"cam_{safe}_ts"),
                (".blurred_boxes.parquet", f"cam_{safe}_blur"),
            ]:
                print(f"[REGISTER] {label}")
                if tracker:
                    tracker.begin("bronze", label)

                # Use the registrar's batched symlink approach
                try:
                    rows = registrar.register_fuse_sensor(
                        label, f"camera/{sensor}",
                        suffix_filter=suffix, fuse_root=FUSE_ROOT,
                    )
                    cam_results[label] = rows
                except Exception as e:
                    print(f"  [ERROR] {label}: {e}")
                    cam_results[label] = 0

    camera_wall = time.time() - camera_t0
    results.append(PhaseResult(
        phase="camera_registration", table="ALL",
        wall_s=round(camera_wall, 2),
        rows=sum(v for v in cam_results.values() if v > 0),
    ))
    print(f"\nCamera registration: {camera_wall:.1f}s")
    for t, c in cam_results.items():
        print(f"  {t}: {c:,} rows")
    bronze_results.update(cam_results)

    gc.collect()

    # ── NOTE: Lidar resume deferred to after Silver/Gold/Queries ─────
    # Lidar is extremely slow (~2.7 min/dir × 340 dirs = ~15h).
    # We proceed with Silver/Gold/Queries using partial lidar data (33/340 dirs),
    # then optionally resume lidar at the end.
    lidar_t0 = time.time()
    lidar_wall = 0.0

    gc.collect()

    # ── SILVER VIEWS ──────────────────────────────────────────────────
    print(f"\n[PHASE 4] SILVER VIEWS — SQL views over Bronze")
    print("-" * 50)
    silver_t0 = time.time()

    from .transform_silver import NvidiaSilverTransformer
    cfg.nvidia.silver_mode = "view"
    view_transformer = NvidiaSilverTransformer(spark, cfg)
    silver_results = view_transformer.transform_all()

    silver_wall = time.time() - silver_t0
    results.append(PhaseResult(
        phase="silver_views", table="ALL",
        wall_s=round(silver_wall, 2),
        rows=sum(v for v in silver_results.values() if isinstance(v, int) and v > 0),
    ))
    print(f"\nSilver views: {silver_wall:.1f}s")
    for t, c in silver_results.items():
        print(f"  {t}: {c}")

    gc.collect()

    # ── GOLD CONSTRUCTION ─────────────────────────────────────────────
    print(f"\n[PHASE 5] GOLD CONSTRUCTION")
    print("-" * 50)
    gold_t0 = time.time()

    from .build_gold import NvidiaGoldBuilder
    cfg.nvidia.gold_mode = "materialized"
    gold_tracker = BenchmarkTracker("continuation-gold")
    builder = NvidiaGoldBuilder(spark, cfg, tracker=gold_tracker)
    gold_results = builder.build_all()

    gold_wall = time.time() - gold_t0
    results.append(PhaseResult(
        phase="gold", table="ALL",
        wall_s=round(gold_wall, 2),
        rows=sum(v for v in gold_results.values() if v > 0),
    ))
    print(f"\nGold construction: {gold_wall:.1f}s")
    for t, c in gold_results.items():
        print(f"  {t}: {c}")

    # Gold storage overhead check
    print(f"\n[OVERHEAD CHECK]")
    try:
        gold_bytes = 0
        for tbl_name in gold_results:
            if gold_results[tbl_name] <= 0:
                continue
            try:
                tbl_full = f"{CAT}.{NS_G}.{tbl_name}"
                files = spark.sql(
                    f"SELECT file_size_in_bytes FROM {tbl_full}.files"
                ).collect()
                tbl_bytes = sum(f.file_size_in_bytes for f in files)
                gold_bytes += tbl_bytes
                print(f"  {tbl_name}: {tbl_bytes / 1073741824:.2f} GB")
            except Exception as e:
                print(f"  {tbl_name}: could not measure ({e})")

        # Rough raw dataset size (avoid walking NFS)
        raw_gb_estimate = 4500  # ~4.5 TB based on dataset description
        overhead_pct = (gold_bytes / (raw_gb_estimate * 1073741824)) * 100
        print(f"\n  Gold storage: {gold_bytes / 1073741824:.2f} GB")
        print(f"  Raw data (est): ~{raw_gb_estimate} GB")
        print(f"  Overhead: ~{overhead_pct:.1f}%")

        if overhead_pct > 20:
            print(f"\n  [PIVOT] Overhead > 20% — rebuilding Gold as VIEWS")
            cfg.nvidia.gold_mode = "view"
            builder_v = NvidiaGoldBuilder(spark, cfg)
            builder_v.build_all()
    except Exception as e:
        print(f"  Overhead check failed: {e}")

    gc.collect()

    # ── QUERY BENCHMARKS ──────────────────────────────────────────────
    print(f"\n[PHASE 6] QUERY BENCHMARKS")
    print("=" * 70)

    query_results: List[QueryBenchResult] = []

    # Warm JVM
    for _ in range(3):
        try:
            spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.clip_index").collect()
        except Exception:
            pass

    # BENCH 1: Raw NFS vs Bronze vs Silver (egomotion)
    print("\nBENCH 1: Raw NFS vs Bronze vs Silver (egomotion)")
    print("-" * 50)

    ego_fuse = os.path.join(FUSE_ROOT, "labels", "egomotion")
    if os.path.isdir(ego_fuse):
        chunk_dirs = sorted(
            d for d in globmod.glob(os.path.join(ego_fuse, "*.zip"))
            if os.path.isdir(d)
        )[:10]
        if chunk_dirs:
            def raw_ego():
                return spark.read.parquet(*[f"file://{d}" for d in chunk_dirs])
            sampler = MemorySampler(interval=2.0)
            sampler.start()
            try:
                med, rows, times = _time_query(spark, raw_ego, use_count=True)
                sampler.stop()
                print(f"  Raw NFS (10 dirs): {med:.3f}s, {rows:,} rows, RSS {sampler.peak_mb():.0f} MB")
                query_results.append(QueryBenchResult(
                    "Raw NFS egomotion (10 dirs)", "raw",
                    "Direct Spark read of FUSE-exposed parquets",
                    med, rows, 5, times, sampler.peak_mb()))
            except Exception as e:
                sampler.stop()
                print(f"  Raw NFS: FAILED ({e})")

    for tier_name, tier_ns in [("Bronze", NS_B), ("Silver", NS_S)]:
        try:
            def mk_q(ns=tier_ns):
                return lambda: spark.sql(f"SELECT count(*) FROM {CAT}.{ns}.egomotion")
            sampler = MemorySampler(interval=2.0)
            sampler.start()
            med, _, times = _time_query(spark, mk_q())
            sampler.stop()
            cnt = spark.sql(f"SELECT count(*) FROM {CAT}.{tier_ns}.egomotion").collect()[0][0]
            print(f"  {tier_name}: {med:.3f}s, {cnt:,} rows, RSS {sampler.peak_mb():.0f} MB")
            query_results.append(QueryBenchResult(
                f"{tier_name} egomotion count", tier_name.lower(),
                f"Iceberg {tier_name.lower()} count(*)",
                med, cnt, 5, times, sampler.peak_mb()))
        except Exception as e:
            print(f"  {tier_name}: FAILED ({e})")

    # BENCH 2: Gold vs ad-hoc join
    print(f"\nBENCH 2: Gold Materialized vs Ad-Hoc Silver Join")
    print("-" * 50)
    med_g = 0
    try:
        def gold_lidar():
            return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego")
        med_g, _, times_g = _time_query(spark, gold_lidar)
        cnt_g = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego").collect()[0][0]
        print(f"  Gold lidar_with_ego: {med_g:.3f}s, {cnt_g:,} rows")
        query_results.append(QueryBenchResult(
            "Gold lidar_with_ego count", "gold",
            "Pre-materialized", med_g, cnt_g, 5, times_g))
    except Exception as e:
        print(f"  Gold lidar_with_ego: FAILED ({e})")

    try:
        lidar_tbl = "lidar_decoded" if cfg.nvidia.lidar_mode == "decoded" else "lidar"
        def silver_join():
            return spark.sql(f"""
                SELECT count(*) FROM (
                    SELECT l.clip_id, l.reference_timestamp
                    FROM {CAT}.{NS_S}.{lidar_tbl} l
                    LEFT JOIN {CAT}.{NS_S}.egomotion e ON l.clip_id = e.clip_id
                )
            """)
        med_s, _, times_s = _time_query(spark, silver_join)
        speedup = med_s / med_g if med_g > 0 else float("inf")
        print(f"  Silver lidar+ego join: {med_s:.3f}s (speedup: {speedup:.1f}x)")
        query_results.append(QueryBenchResult(
            "Silver lidar+ego join", "silver",
            "Ad-hoc join", med_s, 0, 5, times_s,
            extra={"speedup_vs_gold": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Silver join: FAILED ({e})")

    # BENCH 3: Partition pruning on radar_ego_fusion
    print(f"\nBENCH 3: Partition Pruning (radar_ego_fusion)")
    print("-" * 50)
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
        query_results.append(QueryBenchResult(
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
        print(f"  Single partition ({target}): {med_single:.3f}s, {single_cnt:,} rows (speedup: {speedup:.1f}x)")
        query_results.append(QueryBenchResult(
            "Radar single partition", "gold",
            f"Pruned to 1/{len(sensors)}", med_single, single_cnt, 5, times_single,
            extra={"speedup_vs_full": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Partition pruning: FAILED ({e})")

    # BENCH 4: Aggregation patterns
    print(f"\nBENCH 4: Aggregation Patterns")
    print("-" * 50)
    agg_queries = [
        ("Clip summary by split", "gold",
         f"SELECT split, count(*) FROM {CAT}.{NS_G}.sensor_fusion_clip GROUP BY split"),
    ]
    # Only add radar queries if radar_ego_fusion exists
    try:
        spark.sql(f"SELECT 1 FROM {CAT}.{NS_G}.radar_ego_fusion LIMIT 1").collect()
        agg_queries.insert(0, ("Radar stats by sensor", "gold",
            f"SELECT sensor_name, count(*), avg(distance) FROM {CAT}.{NS_G}.radar_ego_fusion GROUP BY sensor_name"))
    except Exception:
        pass

    for name, tier, sql in agg_queries:
        try:
            def mk_q(s=sql):
                return lambda: spark.sql(s)
            med, rows, times = _time_query(spark, mk_q())
            print(f"  {name}: {med:.3f}s, {rows} groups")
            query_results.append(QueryBenchResult(name, tier, sql[:80], med, rows, 5, times))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # BENCH 5: Tier comparison (same radar sensor across tiers)
    print(f"\nBENCH 5: Same Query Across Tiers")
    print("-" * 50)
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
                    query_results.append(QueryBenchResult(
                        f"{label} {radar_t}", label.split()[0].lower(),
                        sql[:80], med, cnt, 5, times))
                except Exception as e:
                    print(f"  {label}: FAILED ({e})")
    except Exception as e:
        print(f"  Tier comparison: FAILED ({e})")

    # BENCH 6: Full table scan latency
    print(f"\nBENCH 6: Full Table Scan Latency (small -> large)")
    print("-" * 50)
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
            query_results.append(QueryBenchResult(
                f"Full scan {name}", tier,
                f"count(*) on {name}", med, cnt, 5, times, sampler.peak_mb()))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── FINAL REPORT ─────────────────────────────────────────────────
    overall_mem = overall_sampler.stop()
    pipeline_wall = time.time() - pipeline_t0

    spark.stop()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline": {
            "total_pipeline_s": round(pipeline_wall, 2),
            "radar_s": round(radar_wall, 2),
            "camera_s": round(camera_wall, 2),
            "silver_views_s": round(silver_wall, 2),
            "gold_s": round(gold_wall, 2),
            "peak_rss_mb": overall_sampler.peak_mb(),
            "note": "lidar partial (33/340 dirs, ~1M rows) - deferred due to ~15h estimated time",
        },
        "bronze_results": bronze_results,
        "silver_results": {k: v for k, v in silver_results.items()},
        "gold_results": {k: v for k, v in gold_results.items()},
        "ingestion_phases": [asdict(r) for r in results],
        "query_benchmarks": [asdict(r) for r in query_results],
    }

    print(f"\n{'='*70}")
    print("INGESTION PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Phase':<30} {'Table':<25} {'Wall(s)':>10} {'Rows':>15}")
    print("-" * 85)
    for r in results:
        print(f"{r.phase:<30} {r.table:<25} {r.wall_s:>10.1f} {r.rows:>15,}")

    print(f"\n{'='*70}")
    print("QUERY LATENCY BENCHMARKS")
    print(f"{'='*70}")
    print(f"{'Name':<45} {'Tier':<8} {'Median(s)':>10} {'Rows':>15}")
    print("-" * 83)
    for r in query_results:
        print(f"{r.name:<45} {r.tier:<8} {r.median_s:>10.3f} {r.row_count:>15,}")
        if r.extra:
            for k, v in r.extra.items():
                print(f"  {'':45} {k}: {v}")

    print(f"\n{'='*70}")
    print(f"TOTAL PIPELINE: {pipeline_wall:.1f}s ({pipeline_wall/60:.1f} min)")
    print(f"  Radar: {radar_wall:.1f}s | Camera: {camera_wall:.1f}s")
    print(f"  Silver: {silver_wall:.1f}s | Gold: {gold_wall:.1f}s")
    print(f"  NOTE: Lidar partial (33/340 dirs) — full registration deferred")
    print(f"  Peak RSS: {overall_sampler.peak_mb():.0f} MB")
    print(f"{'='*70}")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
