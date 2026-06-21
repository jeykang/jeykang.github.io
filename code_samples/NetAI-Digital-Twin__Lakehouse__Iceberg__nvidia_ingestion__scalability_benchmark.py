#!/usr/bin/env python3
"""
Scalability benchmark for the Nvidia PhysicalAI ingestion pipeline.

Runs the full Bronze -> Silver -> Gold pipeline at increasing data scales.
Each scale level runs as a **separate subprocess** to guarantee a fresh
JVM (avoids accumulated heap pressure across levels).

Usage (inside spark-iceberg container):
    python3 -m nvidia_ingestion.scalability_benchmark
    python3 -m nvidia_ingestion.scalability_benchmark --max-level 6
    python3 -m nvidia_ingestion.scalability_benchmark --driver-memory 8g

Internal: launched by the orchestrator with --run-level <N> for each level.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ScalePoint:
    """Measurements for a single scale level."""
    level: int
    max_chunks: int
    max_clips: int
    phase: str
    wall_s: float = 0.0
    cpu_user_s: float = 0.0
    cpu_sys_s: float = 0.0
    peak_rss_mb: float = 0.0
    rows_out: int = 0
    tables_ok: int = 0
    tables_fail: int = 0
    error: str = ""


# (max_chunks, max_clips_per_chunk)
SCALE_LEVELS = [
    (1, 2),      # Level 0  - baseline
    (1, 5),      # Level 1
    (1, 10),     # Level 2
    (1, 25),     # Level 3
    (1, 50),     # Level 4
    (1, 60),     # Level 5
    (1, 75),     # Level 6
    (1, 100),    # Level 7  - full chunk
    (2, 100),    # Level 8
    (5, 100),    # Level 9
    (10, 100),   # Level 10
    (20, 100),   # Level 9
]


# ---------------------------------------------------------------------------
# Worker: runs one scale level inside a fresh process
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def _cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (ru.ru_utime, ru.ru_stime)


def _run_one_phase(phase_name, run_fn, spark, cfg, level, max_chunks, max_clips):
    sp = ScalePoint(level=level, max_chunks=max_chunks, max_clips=max_clips,
                    phase=phase_name)
    try:
        t0 = time.perf_counter()
        c0 = _cpu()
        res = run_fn(spark, cfg)
        sp.wall_s = round(time.perf_counter() - t0, 2)
        c1 = _cpu()
        sp.cpu_user_s = round(c1[0] - c0[0], 2)
        sp.cpu_sys_s = round(c1[1] - c0[1], 2)
        sp.peak_rss_mb = round(_rss_mb(), 1)
        sp.rows_out = sum(v for v in res.values() if v >= 0)
        sp.tables_ok = sum(1 for v in res.values() if v >= 0)
        sp.tables_fail = sum(1 for v in res.values() if v < 0)
    except Exception as e:
        sp.error = str(e)[:300]
        sp.peak_rss_mb = round(_rss_mb(), 1)
    return sp


def run_single_level(level: int, driver_memory: str, output_file: str):
    """Execute one scale level (called in a subprocess)."""
    if level >= len(SCALE_LEVELS):
        return
    max_chunks, max_clips = SCALE_LEVELS[level]

    from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces
    from .benchmark import BenchmarkTracker
    from .ingest_bronze import NvidiaBronzeIngester
    from .transform_silver import NvidiaSilverTransformer
    from .build_gold import NvidiaGoldBuilder

    ns_suffix = f"_scale_{level}"
    cfg = NvidiaPipelineConfig()
    cfg.nvidia.lidar_mode = "blob"
    cfg.nvidia.driver_memory = driver_memory
    cfg.nvidia.max_zip_chunks = max_chunks
    cfg.nvidia.max_clips_per_chunk = max_clips
    cfg.nvidia.namespace_bronze = f"nvidia_bronze{ns_suffix}"
    cfg.nvidia.namespace_silver = f"nvidia_silver{ns_suffix}"
    cfg.nvidia.namespace_gold = f"nvidia_gold{ns_suffix}"

    spark = build_spark_session(cfg, f"nvidia-scale-{level}")
    create_namespaces(spark, cfg)

    results: List[ScalePoint] = []

    # Bronze
    def run_bronze(sp, cfg):
        tracker = BenchmarkTracker(f"scale-{level}-bronze")
        return NvidiaBronzeIngester(sp, cfg, tracker).ingest_all()
    sp = _run_one_phase("bronze", run_bronze, spark, cfg, level, max_chunks, max_clips)
    results.append(sp)
    print(f"  Bronze: {sp.wall_s}s, {sp.rows_out} rows, RSS {sp.peak_rss_mb}MB"
          + (f" ERROR: {sp.error}" if sp.error else ""), flush=True)
    if sp.error:
        _save_and_cleanup(results, output_file, spark, cfg)
        return

    # Silver
    def run_silver(sp, cfg):
        tracker = BenchmarkTracker(f"scale-{level}-silver")
        return NvidiaSilverTransformer(sp, cfg, tracker).transform_all()
    sp = _run_one_phase("silver", run_silver, spark, cfg, level, max_chunks, max_clips)
    results.append(sp)
    print(f"  Silver: {sp.wall_s}s, {sp.rows_out} rows, RSS {sp.peak_rss_mb}MB"
          + (f" ERROR: {sp.error}" if sp.error else ""), flush=True)
    if sp.error:
        _save_and_cleanup(results, output_file, spark, cfg)
        return

    # Gold
    def run_gold(sp, cfg):
        tracker = BenchmarkTracker(f"scale-{level}-gold")
        return NvidiaGoldBuilder(sp, cfg, tracker).build_all()
    sp = _run_one_phase("gold", run_gold, spark, cfg, level, max_chunks, max_clips)
    results.append(sp)
    print(f"  Gold: {sp.wall_s}s, {sp.rows_out} rows, RSS {sp.peak_rss_mb}MB"
          + (f" ERROR: {sp.error}" if sp.error else ""), flush=True)

    # Total
    total_sp = ScalePoint(
        level=level, max_chunks=max_chunks, max_clips=max_clips,
        phase="total",
        wall_s=round(sum(r.wall_s for r in results), 2),
        cpu_user_s=round(sum(r.cpu_user_s for r in results), 2),
        cpu_sys_s=round(sum(r.cpu_sys_s for r in results), 2),
        peak_rss_mb=round(_rss_mb(), 1),
        rows_out=sum(r.rows_out for r in results),
        tables_ok=sum(r.tables_ok for r in results),
        tables_fail=sum(r.tables_fail for r in results),
    )
    results.append(total_sp)

    _save_and_cleanup(results, output_file, spark, cfg)


def _save_and_cleanup(results, output_file, spark, cfg):
    """Clean up namespaces, write results, stop Spark."""
    cat = cfg.spark_catalog_name
    for ns in [cfg.nvidia.namespace_bronze, cfg.nvidia.namespace_silver,
               cfg.nvidia.namespace_gold]:
        try:
            tables = spark.sql(f"SHOW TABLES IN {cat}.{ns}").collect()
            for row in tables:
                spark.sql(f"DROP TABLE IF EXISTS {cat}.{ns}.{row[1]} PURGE")
            spark.sql(f"DROP NAMESPACE IF EXISTS {cat}.{ns}")
        except Exception:
            pass
    try:
        spark.stop()
    except Exception:
        pass

    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


# ---------------------------------------------------------------------------
# Orchestrator: spawns subprocesses
# ---------------------------------------------------------------------------

def orchestrate(args):
    """Main loop - spawns a fresh subprocess per scale level."""
    output = args.output
    all_results: List[dict] = []

    print("=" * 70)
    print("NVIDIA PHYSICALAI - SCALABILITY BENCHMARK")
    print(f"  Levels: {args.start_level} to {min(args.max_level, len(SCALE_LEVELS))-1}")
    print(f"  Driver memory: {args.driver_memory}")
    print("=" * 70, flush=True)

    for i in range(args.start_level, min(args.max_level, len(SCALE_LEVELS))):
        mc, ml = SCALE_LEVELS[i]
        print(f"\n{'#' * 60}")
        print(f"# Level {i}: max_chunks={mc}, max_clips={ml}")
        print(f"{'#' * 60}", flush=True)

        level_file = f"/tmp/_scale_level_{i}.json"
        if os.path.exists(level_file):
            os.unlink(level_file)

        cmd = [
            sys.executable, "-m", "nvidia_ingestion.scalability_benchmark",
            "--run-level", str(i),
            "--driver-memory", args.driver_memory,
            "--level-output", level_file,
        ]
        env = os.environ.copy()
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=env, timeout=3600,
                stdout=sys.stdout, stderr=sys.stderr,
            )
        except subprocess.TimeoutExpired:
            print(f"\n!!! Level {i} timed out after 3600s - stopping", flush=True)
            all_results.append(asdict(ScalePoint(
                level=i, max_chunks=mc, max_clips=ml,
                phase="timeout", error="timeout 3600s"
            )))
            break
        elapsed = time.time() - t0

        if os.path.exists(level_file):
            with open(level_file) as f:
                level_results = json.load(f)
            all_results.extend(level_results)
            os.unlink(level_file)

            errors = [r for r in level_results if r.get("error")]
            totals = [r for r in level_results if r["phase"] == "total"]
            if totals:
                t = totals[0]
                print(f"\n>>> L{i} TOTAL: {t['wall_s']}s, {t['rows_out']} rows, "
                      f"RSS {t['peak_rss_mb']}MB", flush=True)
            if errors:
                print(f"\n!!! Level {i} had errors - stopping sweep", flush=True)
                break
        else:
            print(f"\n!!! Level {i} crashed (exit={proc.returncode}, "
                  f"{elapsed:.0f}s) - stopping sweep", flush=True)
            all_results.append(asdict(ScalePoint(
                level=i, max_chunks=mc, max_clips=ml,
                phase="crash", error=f"exit code {proc.returncode}"
            )))
            break

    # Save combined results
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[RESULTS] Saved to {output}")

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"{'Lvl':>3} {'Chnk':>4} {'Clip':>4} {'Phase':>8} "
          f"{'Wall(s)':>8} {'CPU_U':>7} {'CPU_S':>7} {'RSS_MB':>8} "
          f"{'Rows':>12} {'OK':>3} {'Fail':>4}")
    print("-" * 100)
    for r in all_results:
        err = " ERR" if r.get("error") else ""
        print(f"{r['level']:>3} {r['max_chunks']:>4} {r['max_clips']:>4} "
              f"{r['phase']:>8} {r['wall_s']:>8.1f} {r['cpu_user_s']:>7.1f} "
              f"{r['cpu_sys_s']:>7.1f} {r['peak_rss_mb']:>8.1f} "
              f"{r['rows_out']:>12} {r['tables_ok']:>3} {r['tables_fail']:>4}"
              f"{err}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Nvidia PhysicalAI scalability benchmark"
    )
    parser.add_argument("--max-level", type=int, default=len(SCALE_LEVELS))
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--driver-memory", type=str, default="4g")
    parser.add_argument("--output", type=str,
                        default="/tmp/nvidia_scalability_results.json")
    # Internal: run a single level (used by subprocess)
    parser.add_argument("--run-level", type=int, default=None)
    parser.add_argument("--level-output", type=str, default=None)
    args = parser.parse_args()

    if args.run_level is not None:
        run_single_level(args.run_level, args.driver_memory,
                         args.level_output or f"/tmp/_scale_level_{args.run_level}.json")
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
