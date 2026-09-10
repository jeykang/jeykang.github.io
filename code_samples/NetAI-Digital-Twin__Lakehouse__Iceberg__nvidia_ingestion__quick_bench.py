#!/usr/bin/env python3
"""
Quick benchmark: Gold views (all 3) + query benchmarks.
Silver views already exist in the catalog from the prior run.
All Gold tables are created as VIEWS to avoid multi-hour materializations.
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

    cfg = NvidiaPipelineConfig()
    cfg.nvidia.silver_mode = "view"
    cfg.nvidia.gold_mode = "view"  # ALL Gold as views — no materialization

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
    print("QUICK BENCH: Gold Views + Query Benchmarks")
    print(f"{'='*70}\n")

    spark = build_spark_session(cfg, app_name="nvidia-quick-bench")
    create_namespaces(spark, cfg)

    CAT = cfg.spark_catalog_name
    NS_B = cfg.nvidia.namespace_bronze
    NS_S = cfg.nvidia.namespace_silver
    NS_G = cfg.nvidia.namespace_gold

    # Inventory Bronze
    existing_bronze = sorted(r[1] for r in spark.sql(f"SHOW TABLES IN {CAT}.{NS_B}").collect())
    print(f"Bronze tables: {len(existing_bronze)}")

    # Silver views exist from prior run — SHOW TABLES doesn't list views in Iceberg,
    # so derive the list from Bronze tables (Silver is 1:1 with Bronze)
    existing_silver = existing_bronze  # Silver views mirror Bronze tables
    print(f"Silver views: {len(existing_silver)} (derived from Bronze, created in prior run)")

    # ── GOLD VIEWS ────────────────────────────────────────────────────
    print(f"\n[PHASE 1] GOLD VIEWS — all 3 as zero-copy SQL views")
    print("-" * 50)
    gold_t0 = time.time()
    gold_results: Dict[str, int] = {}

    # 1. sensor_fusion_clip: positional merge of 4 metadata tables
    #    Use SQL to avoid the monotonically_increasing_id() issue.
    #    Just do a simple SELECT * with all columns from clip_index as base.
    print("[GOLD] sensor_fusion_clip (view)")
    try:
        # Simple approach: clip_index has the core columns, join others on clip_id
        # Check if all tables have clip_id
        ci_cols = [f.name for f in spark.table(f"{CAT}.{NS_S}.clip_index").schema]
        dc_cols = [f.name for f in spark.table(f"{CAT}.{NS_S}.data_collection").schema]
        sp_cols = [f.name for f in spark.table(f"{CAT}.{NS_S}.sensor_presence").schema]
        vd_cols = [f.name for f in spark.table(f"{CAT}.{NS_S}.vehicle_dimensions").schema]

        # data_collection unique cols (not in clip_index)
        dc_extra = [c for c in dc_cols if c not in ci_cols]
        sp_extra = [c for c in sp_cols if c not in ci_cols]
        vd_extra = [c for c in vd_cols if c not in ci_cols]

        dc_select = ", ".join(f"dc.{c}" for c in dc_extra) if dc_extra else ""
        sp_select = ", ".join(f"sp.{c}" for c in sp_extra) if sp_extra else ""
        vd_select = ", ".join(f"vd.{c}" for c in vd_extra) if vd_extra else ""

        extra_cols = ", ".join(filter(None, [dc_select, sp_select, vd_select]))
        extra_clause = f", {extra_cols}" if extra_cols else ""

        # If there's no common join key other than position, use ROW_NUMBER()
        # Actually, clip_index and data_collection have same row count (310,895)
        # and are aligned by row position in the Nvidia dataset.
        # Use a window function approach.
        sql = f"""
            SELECT ci.*{extra_clause}
            FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS _rn FROM {CAT}.{NS_S}.clip_index) ci
            LEFT JOIN (SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS _rn FROM {CAT}.{NS_S}.data_collection) dc ON ci._rn = dc._rn
            LEFT JOIN (SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS _rn FROM {CAT}.{NS_S}.sensor_presence) sp ON ci._rn = sp._rn
            LEFT JOIN (SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS _rn FROM {CAT}.{NS_S}.vehicle_dimensions) vd ON ci._rn = vd._rn
        """
        # Actually ROW_NUMBER with ORDER BY NULL is non-deterministic in Spark.
        # Since these are positional, and the tables may not have a reliable
        # join key, let's just skip sensor_fusion_clip for now and note it
        # requires materialization (uses monotonically_increasing_id).
        raise RuntimeError("sensor_fusion_clip requires materialization (positional join) — skipped in view mode")
    except Exception as e:
        print(f"  [SKIP] sensor_fusion_clip: {e}")
        gold_results["sensor_fusion_clip"] = -1

    # 2. lidar_with_ego: view joining lidar with aggregated egomotion
    print("[GOLD] lidar_with_ego (view)")
    try:
        lidar_tbl = "lidar_decoded" if "lidar_decoded" in existing_bronze else "lidar"
        sql = (
            f"SELECT l.*, e.ego_sample_count, e.ego_trajectory "
            f"FROM {CAT}.{NS_S}.{lidar_tbl} l "
            f"LEFT JOIN ("
            f"  SELECT clip_id, "
            f"    count(*) AS ego_sample_count, "
            f"    collect_list(struct(timestamp, x, y, z, qw, qx, qy, qz)) AS ego_trajectory "
            f"  FROM {CAT}.{NS_S}.egomotion "
            f"  GROUP BY clip_id"
            f") e ON l.clip_id = e.clip_id"
        )
        full = f"{CAT}.{NS_G}.lidar_with_ego"
        spark.sql(f"CREATE OR REPLACE VIEW {full} AS {sql}")
        rows = spark.table(full).count()
        gold_results["lidar_with_ego"] = rows
        print(f"  [VIEW] {full}: {rows:,} rows")
    except Exception as e:
        print(f"  [ERROR] lidar_with_ego: {e}")
        gold_results["lidar_with_ego"] = -1

    # 3. radar_ego_fusion: UNION ALL of all radar sensors + ego count
    print("[GOLD] radar_ego_fusion (view)")
    try:
        # Derive radar tables from Bronze (Silver views mirror Bronze 1:1)
        radar_tables = sorted(t for t in existing_bronze if t.startswith("radar_"))
        print(f"  Found {len(radar_tables)} radar tables in silver")

        unions = []
        for rt in radar_tables:
            cols = [f.name for f in spark.table(f"{CAT}.{NS_S}.{rt}").schema]
            has_sensor = "sensor_name" in cols
            if has_sensor:
                unions.append(f"SELECT * FROM {CAT}.{NS_S}.{rt}")
            else:
                sensor_name = rt.replace("radar_", "", 1)
                unions.append(f"SELECT *, '{sensor_name}' AS sensor_name FROM {CAT}.{NS_S}.{rt}")
        union_sql = " UNION ALL ".join(unions)

        sql = (
            f"SELECT r.*, e.ego_sample_count "
            f"FROM ({union_sql}) r "
            f"LEFT JOIN ("
            f"  SELECT clip_id, count(*) AS ego_sample_count "
            f"  FROM {CAT}.{NS_S}.egomotion "
            f"  GROUP BY clip_id"
            f") e ON r.clip_id = e.clip_id"
        )
        full = f"{CAT}.{NS_G}.radar_ego_fusion"
        spark.sql(f"CREATE OR REPLACE VIEW {full} AS {sql}")
        # Don't count — 11.3B rows would take too long
        gold_results["radar_ego_fusion"] = 0  # view created, count skipped
        print(f"  [VIEW] {full}: created (count skipped — 11.3B rows)")
    except Exception as e:
        print(f"  [ERROR] radar_ego_fusion: {e}")
        gold_results["radar_ego_fusion"] = -1

    gold_wall = time.time() - gold_t0
    report["gold_tables"] = gold_results
    report["timings_s"]["gold_views_s"] = round(gold_wall, 2)
    print(f"\nGold views: {gold_wall:.1f}s")

    gc.collect()

    # ── PHASE 2: BENCHMARKS ───────────────────────────────────────────
    print(f"\n[PHASE 2] QUERY BENCHMARKS")
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

    # ── BENCH 2: Gold lidar_with_ego view vs Silver ad-hoc join ──────
    print(f"\nBENCH 2: Gold View vs Silver Ad-Hoc Join — lidar+ego")
    print("-" * 60)
    med_gold = 0.0
    if gold_results.get("lidar_with_ego", -1) > 0:
        try:
            def gold_lwe(): return spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego")
            med_gold, _, tg = _time_query(spark, gold_lwe, runs=3)
            cnt_g = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_G}.lidar_with_ego").collect()[0][0]
            print(f"  Gold lidar_with_ego (view): {med_gold:.3f}s | {cnt_g:>12,} rows")
            qresults.append(QueryBenchResult("Gold lidar_with_ego count(*)", "gold",
                "Pre-joined view", med_gold, cnt_g, 3, tg))
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
        print(f"  Silver lidar+ego join:    {med_sj:.3f}s | {cnt_sj:>12,} rows | Gold/Silver: {speedup:.1f}x")
        qresults.append(QueryBenchResult("Silver lidar+ego ad-hoc join", "silver",
            "Ad-hoc join at query time", med_sj, cnt_sj, 3, tsj,
            extra={"gold_vs_silver": f"{speedup:.1f}x"}))
    except Exception as e:
        print(f"  Silver join: FAILED ({e})")

    # ── BENCH 3: Radar single-sensor filter vs full count ────────────
    print(f"\nBENCH 3: Radar View — single sensor filter vs full scan")
    print("-" * 60)
    if gold_results.get("radar_ego_fusion", 0) >= 0:
        try:
            tbl = f"{CAT}.{NS_G}.radar_ego_fusion"
            # Get a sample sensor name from bronze (mirrors silver)
            first_radar = next(t for t in existing_bronze if t.startswith("radar_"))
            sample_sensor = first_radar.replace("radar_", "", 1)

            # Single sensor via Gold view
            def single_r(): return spark.sql(
                f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{sample_sensor}'")
            sm2 = MemorySampler(2.0); sm2.start()
            med_s, _, ts = _time_query(spark, single_r, runs=3)
            sm2.stop()
            scnt = spark.sql(
                f"SELECT count(*) FROM {tbl} WHERE sensor_name = '{sample_sensor}'"
            ).collect()[0][0]
            print(f"  Gold view single sensor: {med_s:.3f}s | {scnt:>15,} rows | RSS {sm2.peak_mb():.0f} MB | sensor={sample_sensor}")
            qresults.append(QueryBenchResult("Radar single-sensor filter (Gold view)", "gold",
                f"WHERE sensor_name='{sample_sensor}'", med_s, scnt, 3, ts, sm2.peak_mb(),
                extra={"sensor": sample_sensor}))

            # Same sensor direct from Bronze
            def bronze_r(): return spark.sql(
                f"SELECT count(*) FROM {CAT}.{NS_B}.{first_radar}")
            sm3 = MemorySampler(2.0); sm3.start()
            med_b, _, tb = _time_query(spark, bronze_r, runs=3)
            sm3.stop()
            bcnt = spark.sql(f"SELECT count(*) FROM {CAT}.{NS_B}.{first_radar}").collect()[0][0]
            overhead = med_s / med_b if med_b > 0 else float("inf")
            print(f"  Bronze direct sensor:    {med_b:.3f}s | {bcnt:>15,} rows | RSS {sm3.peak_mb():.0f} MB")
            print(f"  View overhead: {overhead:.2f}x (view adds ego join + UNION filter)")
            qresults.append(QueryBenchResult(f"Bronze direct {first_radar}", "bronze",
                "Direct table scan", med_b, bcnt, 3, tb, sm3.peak_mb(),
                extra={"gold_view_overhead": f"{overhead:.2f}x"}))
        except Exception as e:
            print(f"  Radar benchmarks: FAILED ({e})")

    # ── BENCH 4: Tier comparison — same radar sensor across tiers ─────
    print(f"\nBENCH 4: Same Query Across Tiers — radar sensor count")
    print("-" * 60)
    try:
        radar_t = next((t for t in sorted(existing_bronze) if t.startswith("radar_")), None)
        if radar_t:
            sensor_name = radar_t.replace("radar_", "", 1)
            tier_q = [
                ("Bronze", f"SELECT count(*) FROM {CAT}.{NS_B}.{radar_t}"),
                ("Silver", f"SELECT count(*) FROM {CAT}.{NS_S}.{radar_t}"),
            ]
            if gold_results.get("radar_ego_fusion", 0) >= 0:
                tier_q.append(
                    ("Gold (view+filter)", f"SELECT count(*) FROM {CAT}.{NS_G}.radar_ego_fusion WHERE sensor_name = '{sensor_name}'")
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

    # ── BENCH 5: Full scan latency by table size ──────────────────────
    print(f"\nBENCH 5: Full Scan count(*) — small → large")
    print("-" * 60)
    scan_items = [
        ("clip_index (311K)", "bronze", f"{CAT}.{NS_B}.clip_index"),
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

    # ── BENCH 6: Aggregation on Gold view ─────────────────────────────
    print(f"\nBENCH 6: Aggregation on Gold Views")
    print("-" * 60)
    if gold_results.get("radar_ego_fusion", 0) >= 0:
        try:
            agg_sql = f"SELECT sensor_name, count(*) AS cnt FROM {CAT}.{NS_G}.radar_ego_fusion GROUP BY sensor_name ORDER BY cnt DESC"
            def agg_fn(): return spark.sql(agg_sql)
            sm = MemorySampler(2.0); sm.start()
            med, nrows, times = _time_query(spark, agg_fn, runs=2, use_count=False)
            sm.stop()
            print(f"  Radar returns per sensor (GROUP BY on 11.3B view): {med:.3f}s | {nrows} groups | RSS {sm.peak_mb():.0f} MB")
            qresults.append(QueryBenchResult("Radar GROUP BY sensor_name (11.3B view)", "gold",
                "GROUP BY on UNION ALL view", med, nrows, 2, times, sm.peak_mb()))
        except Exception as e:
            print(f"  Radar aggregation: FAILED ({e})")

    bench_wall = time.time() - bench_t0
    report["timings_s"]["gold_views_s"] = round(gold_wall, 2)
    report["timings_s"]["benchmarks_s"] = round(bench_wall, 2)
    report["timings_s"]["total_s"] = round(time.time() - t_start, 2)
    # Include previously measured Silver time
    report["timings_s"]["silver_s"] = 14.1  # from prior run
    report["query_benchmarks"] = [asdict(r) for r in qresults]

    spark.stop()

    # ── PRINT FINAL REPORT ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL BENCHMARK REPORT")
    print(f"{'='*70}")

    t = report["timings_s"]
    print(f"\nTimings:")
    print(f"  Silver views (prior):  {t.get('silver_s',0):.1f}s (31 views)")
    print(f"  Gold views:            {t.get('gold_views_s',0):.1f}s")
    print(f"  Benchmarks:            {t.get('benchmarks_s',0):.1f}s")
    print(f"  Total (this run):      {t.get('total_s',0):.1f}s ({t.get('total_s',0)/60:.1f} min)")

    print(f"\nGold tables:")
    for k, v in gold_results.items():
        print(f"  {k} [view]: {v:,}" if isinstance(v, int) and v >= 0 else f"  {k}: {v}")

    print(f"\nQuery Benchmarks:")
    print(f"  {'Name':<55} {'Tier':<8} {'Median(s)':>10} {'Rows':>18}")
    print("  " + "-" * 95)
    for r in qresults:
        extra = "  ".join(f"{k}={v}" for k, v in r.extra.items())
        print(f"  {r.name:<55} {r.tier:<8} {r.median_s:>10.3f} {r.row_count:>18,}  {extra}")

    print(f"\n{'='*70}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
