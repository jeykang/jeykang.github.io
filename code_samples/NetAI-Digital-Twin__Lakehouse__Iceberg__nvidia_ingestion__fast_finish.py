#!/usr/bin/env python3
"""
Fast-finish: Silver views + Gold (sensor_fusion_clip + lidar_with_ego materialized,
radar_ego_fusion as view) + benchmarks. Skips camera registration entirely —
use whatever is already in Bronze.
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
OUTPUT_PATH = "/tmp/nvidia_fast_benchmark.json"


class MemorySampler:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread: Optional[threading.Thread] = None

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
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            self.samples.append((round(time.time() - self._t0, 1), round(rss, 1)))
            self._stop.wait(self.interval)

    def peak_mb(self) -> float:
        return max((s[1] for s in self.samples), default=0.0)


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


def _time_query(spark, fn: Callable, warmup: int = 1, runs: int = 3,
                use_count: bool = True) -> Tuple[float, int, List[float]]:
    for _ in range(warmup):
        try:
            df = fn()
            df.count() if use_count else df.collect()
        except Exception:
            pass
    times, row_count = [], 0
    for _ in range(runs):
        t0 = time.time()
        df = fn()
        row_count = df.count() if use_count else len(df.collect())
        times.append(time.time() - t0)
    times.sort()
    return times[len(times) // 2], row_count, times


def main():
    from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces
    from .benchmark import BenchmarkTracker
    from .transform_silver import NvidiaSilverTransformer
    from .build_gold import NvidiaGoldBuilder

    cfg = NvidiaPipelineConfig()
    cfg.nvidia.silver_mode = "view"
    # Use view for radar_ego_fusion (11.3B rows — too large to materialize quickly)
    # Override per-table inside build_gold by setting mode to materialized but
    # wrapping radar_ego_fusion separately.
    cfg.nvidia.gold_mode = "materialized"

    t_start = time.time()
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timings_s": {},
        "bronze_table_counts": {},
        "silver_views": {},
        "gold_tables": {},
        "query_benchmarks": [],
    }

    print(f"\n{'='*70}")
    print("FAST-FINISH: Silver → Gold (views for radar) → Benchmarks")
    print(f"{'='*70}\n")

    spark = build_spark_session(cfg, app_name="nvidia-fast-finish")
    create_namespaces(spark, cfg)

    CAT = cfg.spark_catalog_name
    NS_B = cfg.nvidia.namespace_bronze
    NS_S = cfg.nvidia.namespace_silver
    NS_G = cfg.nvidia.namespace_gold

    # Inventory existing Bronze tables
    existing_bronze = {r[1] for r in spark.sql(f"SHOW TABLES IN {CAT}.{NS_B}").collect()}
    print(f"Bronze tables: {len(existing_bronze)}")
    for t in sorted(existing_bronze):
        print(f"  {t}")

    # ── PHASE 1: SILVER VIEWS ─────────────────────────────────────────
    print(f"\n[PHASE 1] SILVER VIEWS — zero-copy SQL views over Bronze")
    print("-" * 50)
    silver_t0 = time.time()

    transformer = NvidiaSilverTransformer(spark, cfg)
    silver_results = transformer.transform_all()

    silver_wall = time.time() - silver_t0
    report["silver_views"] = {k: v for k, v in silver_results.items()}
    report["timings_s"]["silver_s"] = round(silver_wall, 2)
    ok = sum(1 for v in silver_results.values() if isinstance(v, int) and v >= 0)
    print(f"\nSilver: {silver_wall:.1f}s — {ok} views created")

    gc.collect()

    # ── PHASE 2: GOLD — sensor_fusion_clip + lidar_with_ego (materialized) ─
    print(f"\n[PHASE 2] GOLD — materialized: sensor_fusion_clip + lidar_with_ego")
    print("-" * 50)
    gold_t0 = time.time()
    gold_results: Dict[str, int] = {}

    tracker = BenchmarkTracker("fast-finish-gold")
    builder = NvidiaGoldBuilder(spark, cfg, tracker=tracker)

    # sensor_fusion_clip: small (310K rows), fast
    try:
        gold_results["sensor_fusion_clip"] = builder.build_sensor_fusion_clip()
    except Exception as e:
        print(f"  [ERROR] sensor_fusion_clip: {e}")
        gold_results["sensor_fusion_clip"] = -1

    # lidar_with_ego: only 1M lidar rows (partial), fast join with 101M ego
    try:
        gold_results["lidar_with_ego"] = builder.build_lidar_with_ego()
    except Exception as e:
        print(f"  [ERROR] lidar_with_ego: {e}")
        gold_results["lidar_with_ego"] = -1

    # radar_ego_fusion: 11.3B rows — too large to materialize, use VIEW instead
    print(f"\n[PHASE 2b] GOLD — view: radar_ego_fusion (11.3B rows, view avoids materialization)")
    cfg.nvidia.gold_mode = "view"
    builder_v = NvidiaGoldBuilder(spark, cfg)
    try:
        gold_results["radar_ego_fusion"] = builder_v.build_radar_ego_fusion()
    except Exception as e:
        print(f"  [ERROR] radar_ego_fusion view: {e}")
        gold_results["radar_ego_fusion"] = -1

    gold_wall = time.time() - gold_t0
    report["gold_tables"] = {k: v for k, v in gold_results.items()}
    report["timings_s"]["gold_s"] = round(gold_wall, 2)
    print(f"\nGold: {gold_wall:.1f}s")
    for t, c in gold_results.items():
        print(f"  {t}: {c:,}" if isinstance(c, int) and c >= 0 else f"  {t}: {c}")

    # Storage overhead for materialized tables only
    try:
        mat_bytes = 0
        for tbl_name in ["sensor_fusion_clip", "lidar_with_ego"]:
            if gold_results.get(tbl_name, -1) > 0:
                files = spark.sql(
                    f"SELECT file_size_in_bytes FROM {CAT}.{NS_G}.{tbl_name}.files"
                ).collect()
                tb = sum(f.file_size_in_bytes for f in files)
                mat_bytes += tb
                print(f"  {tbl_name} storage: {tb/1073741824:.3f} GB")
        report["gold_materialized_gb"] = round(mat_bytes / 1073741824, 4)
        print(f"  Total materialized Gold: {mat_bytes/1073741824:.3f} GB (Bronze = 0 bytes extra)")
    except Exception as e:
        print(f"  Storage check: {e}")

    gc.collect()

    # ── PHASE 3: BENCHMARKS ───────────────────────────────────────────
    print(f"\n[PHASE 3] QUERY BENCHMARKS")
    print("=" * 70)
    bench_t0 = time.time()
    qresults: List[QueryBenchResult] = []

    # Warmup JVM
    for _ in range(2):
        try:
            spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.clip_index").collect()
        except Exception:
            pass
    print("  JVM warmed.\n")

    # ── BENCH 1: Raw FUSE vs Bronze vs Silver — egomotion ─────────────
    print("BENCH 1: Raw FUSE vs Bronze vs Silver — egomotion count(*)")
    print("-" * 60)
    ego_fuse = os.path.join(FUSE_ROOT, "labels", "egomotion")
    if os.path.isdir(ego_fuse):
        chunk_dirs = sorted(
            d for d in globmod.glob(os.path.join(ego_fuse, "*.zip"))
            if os.path.isdir(d)
        )[:10]
        if chunk_dirs:
            uris = [f"file://{d}" for d in chunk_dirs]
            def raw_ego(): return spark.read.parquet(*uris)
            sm = MemorySampler(2.0); sm.start()
            try:
                med, rows, times = _time_query(spark, raw_ego, runs=3)
                sm.stop()
                print(f"  Raw FUSE (10 dirs):  {med:.3f}s | {rows:>15,} rows | RSS {sm.peak_mb():.0f} MB")
                qresults.append(QueryBenchResult("Raw FUSE egomotion (10 dirs)", "raw",
                    "spark.read.parquet on FUSE", med, rows, 3, times, sm.peak_mb()))
            except Exception as e:
                sm.stop(); print(f"  Raw FUSE: FAILED ({e})")

    for label, ns in [("Bronze", NS_B), ("Silver", NS_S)]:
        try:
            def mk(n=ns): return lambda: spark.sql(f"SELECT count(*) FROM {CAT}.{n}.egomotion")
            sm = MemorySampler(2.0); sm.start()
            med, _, times = _time_query(spark, mk(), runs=3)
            sm.stop()
            cnt = spark.sql(f"SELECT count(*) FROM {CAT}.{ns}.egomotion").collect()[0][0]
            print(f"  {label} Iceberg:       {med:.3f}s | {cnt:>15,} rows | RSS {sm.peak_mb():.0f} MB")
            qresults.append(QueryBenchResult(f"{label} egomotion count(*)", label.lower(),
                f"Iceberg {label.lower()} count", med, cnt, 3, times, sm.peak_mb()))
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    # ── BENCH 2: Gold lidar_with_ego vs Silver ad-hoc join ────────────
    print(f"\nBENCH 2: Gold Materialized vs Silver Ad-Hoc Join — lidar+ego")
    print("-" * 60)
    med_gold = 0.0
    if gold_results.get("lidar_with_ego", -1) > 0:
        try:
            def gold_lwe(): return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego")
            med_gold, _, tg = _time_query(spark, gold_lwe, runs=3)
            cnt_g = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego").collect()[0][0]
            print(f"  Gold lidar_with_ego:      {med_gold:.3f}s | {cnt_g:>12,} rows")
            qresults.append(QueryBenchResult("Gold lidar_with_ego count(*)", "gold",
                "Pre-joined materialized", med_gold, cnt_g, 3, tg))
        except Exception as e:
            print(f"  Gold lidar_with_ego: FAILED ({e})")

    try:
        def slj(): return spark.sql(f"""
            SELECT count(*) FROM (
                SELECT l.clip_id FROM {CAT}.{NS_S}.lidar l
                LEFT JOIN {CAT}.{NS_S}.egomotion e ON l.clip_id = e.clip_id
            )""")
        med_sj, _, tsj = _time_query(spark, slj, runs=3)
        cnt_sj = spark.sql(f"""SELECT count(*) FROM (
            SELECT l.clip_id FROM {CAT}.{NS_S}.lidar l
            LEFT JOIN {CAT}.{NS_S}.egomotion e ON l.clip_id = e.clip_id
        )""").collect()[0][0]
        speedup = med_sj / med_gold if med_gold > 0 else float("inf")
        print(f"  Silver lidar+ego join:    {med_sj:.3f}s | {cnt_sj:>12,} rows | Gold speedup: {speedup:.1f}x")
        qresults.append(QueryBenchResult("Silver lidar+ego ad-hoc join", "silver",
            "Ad-hoc join at query time", med_sj, cnt_sj, 3, tsj,
            extra={"gold_speedup": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Silver join: FAILED ({e})")

    # ── BENCH 3: Partition pruning — radar_ego_fusion (view) ──────────
    print(f"\nBENCH 3: Partition Pruning — radar_ego_fusion (view, 11.3B rows)")
    print("-" * 60)
    if gold_results.get("radar_ego_fusion", -1) > 0:
        try:
            tbl = f"{CAT}.{NS_G}.radar_ego_fusion"
            sensors = [r.sensor_name for r in spark.sql(
                f"SELECT DISTINCT sensor_name FROM {tbl} LIMIT 100").collect()]
            n_sensors = len(sensors)

            def full_r(): return spark.sql(f"SELECT count(*) FROM {tbl}")
            sm = MemorySampler(2.0); sm.start()
            med_full, _, tf = _time_query(spark, full_r, warmup=0, runs=2)
            sm.stop()
            total = spark.sql(f"SELECT count(*) FROM {tbl}").collect()[0][0]
            print(f"  Full view ({n_sensors} sensors): {med_full:.3f}s | {total:>15,} rows | RSS {sm.peak_mb():.0f} MB")
            qresults.append(QueryBenchResult(f"Radar full scan view ({n_sensors} sensors)", "gold",
                "Full UNION ALL view", med_full, total, 2, tf, sm.peak_mb()))

            target = sensors[0]
            def single_r(): return spark.sql(
                f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{target}'")
            sm2 = MemorySampler(2.0); sm2.start()
            med_s, _, ts = _time_query(spark, single_r, runs=3)
            sm2.stop()
            scnt = spark.sql(
                f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{target}'"
            ).collect()[0][0]
            pu = med_full / med_s if med_s > 0 else float("inf")
            print(f"  Single sensor filter:   {med_s:.3f}s | {scnt:>15,} rows | speedup: {pu:.1f}x | RSS {sm2.peak_mb():.0f} MB")
            qresults.append(QueryBenchResult("Radar single-sensor filter", "gold",
                f"WHERE sensor_name='{target}'", med_s, scnt, 3, ts, sm2.peak_mb(),
                extra={"speedup_vs_full": f"{pu:.1f}x", "sensor": target}))
        except Exception as e:
            print(f"  Radar partition pruning: FAILED ({e})")

    # ── BENCH 4: Aggregation ──────────────────────────────────────────
    print(f"\nBENCH 4: Aggregation Patterns")
    print("-" * 60)
    agg_tests: List[Tuple[str, str, str]] = []
    if gold_results.get("sensor_fusion_clip", -1) > 0:
        agg_tests.append(("Clip count by split", "gold",
            f"SELECT split, count(*) FROM {CAT}.{NS_G}.sensor_fusion_clip GROUP BY split"))
    if gold_results.get("radar_ego_fusion", -1) > 0:
        agg_tests.append(("Radar returns per sensor (view agg)", "gold",
            f"SELECT sensor_name, count(*) FROM {CAT}.{NS_G}.radar_ego_fusion GROUP BY sensor_name ORDER BY 2 DESC"))

    for name, tier, sql in agg_tests:
        try:
            def mk(s=sql): return lambda: spark.sql(s)
            med, nrows, times = _time_query(spark, mk(), runs=3, use_count=False)
            print(f"  {name}: {med:.3f}s | {nrows} groups")
            qresults.append(QueryBenchResult(name, tier, sql[:80], med, nrows, 3, times))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    # ── BENCH 5: Tier comparison — same radar sensor across Bronze/Silver/Gold ─
    print(f"\nBENCH 5: Same Query Across Tiers — radar sensor count")
    print("-" * 60)
    try:
        bronze_tables = [r[1] for r in spark.sql(f"SHOW TABLES IN {CAT}.{NS_B}").collect()]
        radar_t = next((t for t in sorted(bronze_tables) if t.startswith("radar_")), None)
        if radar_t:
            sensor_name = radar_t.replace("radar_", "", 1)
            tier_q = [
                ("Bronze", f"SELECT count(*) FROM {CAT}.{NS_B}.{radar_t}"),
                ("Silver", f"SELECT count(*) FROM {CAT}.{NS_S}.{radar_t}"),
            ]
            if gold_results.get("radar_ego_fusion", -1) > 0:
                tier_q.append(
                    ("Gold (view, filter)", f"SELECT count(*) FROM {CAT}.{NS_G}.radar_ego_fusion WHERE sensor_name = '{sensor_name}'")
                )
            for label, sql in tier_q:
                def mk(s=sql): return lambda: spark.sql(s)
                try:
                    med, _, times = _time_query(spark, mk(), runs=3)
                    cnt = spark.sql(sql).collect()[0][0]
                    print(f"  {label:<22} ({radar_t[:28]}): {med:.3f}s | {cnt:>12,} rows")
                    qresults.append(QueryBenchResult(f"{label} {radar_t}", label.split()[0].lower(),
                        sql[:80], med, cnt, 3, times))
                except Exception as e:
                    print(f"  {label}: FAILED ({e})")
    except Exception as e:
        print(f"  Tier comparison: FAILED ({e})")

    # ── BENCH 6: Full scan latency by table size ───────────────────────
    print(f"\nBENCH 6: Full Scan count(*) — small → large")
    print("-" * 60)
    scan_items = [
        ("clip_index", "bronze", f"{CAT}.{NS_B}.clip_index"),
        ("egomotion (101M)", "bronze", f"{CAT}.{NS_B}.egomotion"),
        ("radar_front_center_imaging_lrr_1 (2.17B)", "bronze",
         f"{CAT}.{NS_B}.radar_radar_front_center_imaging_lrr_1"),
    ]
    for name, tier, tbl in scan_items:
        try:
            def mk(t=tbl): return lambda: spark.sql(f"SELECT count(*) FROM {t}")
            sm = MemorySampler(2.0); sm.start()
            med, _, times = _time_query(spark, mk(), runs=3)
            sm.stop()
            cnt = spark.sql(f"SELECT count(*) FROM {tbl}").collect()[0][0]
            print(f"  {name}: {med:.3f}s | {cnt:>15,} rows | RSS {sm.peak_mb():.0f} MB")
            qresults.append(QueryBenchResult(f"Full scan {name}", tier,
                f"count(*)", med, cnt, 3, times, sm.peak_mb()))
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    bench_wall = time.time() - bench_t0
    report["timings_s"]["benchmarks_s"] = round(bench_wall, 2)
    report["timings_s"]["total_s"] = round(time.time() - t_start, 2)
    report["query_benchmarks"] = [asdict(r) for r in qresults]

    spark.stop()

    # ── PRINT FINAL REPORT ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL BENCHMARK REPORT")
    print(f"{'='*70}")

    t = report["timings_s"]
    print(f"\nTimings:")
    print(f"  Silver views:   {t.get('silver_s',0):.1f}s")
    print(f"  Gold tables:    {t.get('gold_s',0):.1f}s")
    print(f"  Benchmarks:     {t.get('benchmarks_s',0):.1f}s")
    print(f"  Total:          {t.get('total_s',0):.1f}s ({t.get('total_s',0)/60:.1f} min)")

    print(f"\nGold tables:")
    for k, v in gold_results.items():
        mode = "view" if k == "radar_ego_fusion" else "materialized"
        print(f"  {k} [{mode}]: {v:,}" if isinstance(v, int) and v >= 0 else f"  {k}: {v}")
    print(f"  Materialized storage: {report.get('gold_materialized_gb', 'N/A')} GB")

    print(f"\nQuery Benchmarks:")
    print(f"  {'Name':<50} {'Tier':<8} {'Median(s)':>10} {'Rows':>18}")
    print("  " + "-"*90)
    for r in qresults:
        extra = "  ".join(f"{k}={v}" for k, v in r.extra.items())
        print(f"  {r.name:<50} {r.tier:<8} {r.median_s:>10.3f} {r.row_count:>18,}  {extra}")

    print(f"\n{'='*70}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
