#!/usr/bin/env python3
"""
Multi-scale ingestion benchmark: proves linear scaling for petabyte projection.

Strategy: parallel-extract parquet files from NFS zips to local /tmp,
register via batched add_files() (zero-copy metadata-only), create
medallion views, benchmark query latency, teardown, repeat at next scale.

Optimizations for TB-scale within 3 hours:
  - Parallel extraction (ThreadPoolExecutor, 8 workers) to saturate NFS
  - Parquet validation removes truncated files from NFS timeouts
  - Resumable extraction (-n flag skips already-extracted files)
  - Batched add_files() avoids JVM OOM on large file sets
  - Persistent symlink dirs survive until after query phase

Usage (inside spark-iceberg container):
    python -u nvidia_ingestion/scalability_bench.py [--scales 2,10,50,200]
"""

import argparse
import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 32g pyspark-shell"
from pyspark.sql import SparkSession


# ── Constants ────────────────────────────────────────────────────────
NFS_ROOT = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
EXTRACT_ROOT = "/tmp/nvidia-extract"
REPORT_FILE = "/tmp/scalability_report.json"
CAT = "iceberg"
ADD_FILES_BATCH = 5_000  # max files per add_files() call
EXTRACT_WORKERS = 2      # parallel zip extractions per sensor
EXTRACT_TIMEOUT = 1200   # seconds per zip extraction
SENSOR_WORKERS = 4       # sensors extracted in parallel


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── Spark session ────────────────────────────────────────────────────
def build_spark():
    return (
        SparkSession.builder.appName("nvidia-scalability-bench")
        .config("spark.sql.defaultCatalog", "iceberg")
        .config("spark.sql.catalog.iceberg",
                "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.catalog-impl",
                "org.apache.iceberg.rest.RESTCatalog")
        .config("spark.sql.catalog.iceberg.uri",
                "http://polaris:8181/api/catalog")
        .config("spark.sql.catalog.iceberg.warehouse", "lakehouse_catalog")
        .config("spark.sql.catalog.iceberg.io-impl",
                "org.apache.iceberg.io.ResolvingFileIO")
        .config("spark.sql.catalog.iceberg.s3.endpoint", "http://minio:9000")
        .config("spark.sql.catalog.iceberg.s3.path-style-access", "true")
        .config("spark.sql.catalog.iceberg.s3.access-key-id", "minioadmin")
        .config("spark.sql.catalog.iceberg.s3.secret-access-key", "minioadmin")
        .config("spark.sql.catalog.iceberg.oauth2-server-uri",
                "http://polaris:8181/api/catalog/v1/oauth/tokens")
        .config("spark.sql.catalog.iceberg.credential", "root:s3cr3t")
        .config("spark.sql.catalog.iceberg.scope", "PRINCIPAL_ROLE:ALL")
        .config("spark.sql.catalog.iceberg.oauth2.server-uri",
                "http://polaris:8181/api/catalog/v1/oauth/tokens")
        .config("spark.sql.catalog.iceberg.oauth2.credential", "root:s3cr3t")
        .config("spark.sql.catalog.iceberg.oauth2.scope", "PRINCIPAL_ROLE:ALL")
        # Hadoop S3A
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )


# ── Extract helpers ──────────────────────────────────────────────────
def _extract_one_zip(zf_path: str, extract_to: str,
                     suffix: str) -> Tuple[str, bool, str]:
    """Extract a single zip. Returns (basename, success, error_msg)."""
    basename = os.path.basename(zf_path)
    try:
        subprocess.run(
            ["unzip", "-n", "-j", "-d", extract_to, zf_path, f"*{suffix}"],
            capture_output=True, text=True, timeout=EXTRACT_TIMEOUT
        )
        return basename, True, ""
    except subprocess.TimeoutExpired:
        return basename, False, "timeout"
    except Exception as e:
        return basename, False, str(e)


def validate_parquets(directory: str, suffix: str = ".parquet") -> Tuple[int, int]:
    """Validate parquet files, remove corrupted ones.
    Returns (valid_count, removed_count)."""
    files = glob.glob(os.path.join(directory, f"*{suffix}"))
    valid = 0
    removed = 0
    for f in files:
        try:
            sz = os.path.getsize(f)
            if sz < 12:
                os.remove(f)
                removed += 1
                continue
            with open(f, "rb") as fh:
                magic_head = fh.read(4)
                fh.seek(-4, 2)
                magic_tail = fh.read(4)
            if magic_head == b"PAR1" and magic_tail == b"PAR1":
                valid += 1
            else:
                os.remove(f)
                removed += 1
        except Exception:
            try:
                os.remove(f)
            except OSError:
                pass
            removed += 1
    return valid, removed


def extract_sensor(sensor_dir: str, extract_to: str,
                   max_chunks: int, suffix: str = ".parquet") -> Tuple[int, float, int]:
    """Extract parquet files from zips using parallel workers.
    Returns (num_valid_files, elapsed_s, bytes_extracted)."""
    t0 = time.time()
    os.makedirs(extract_to, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(sensor_dir, "*.zip")))
    if max_chunks and len(zips) > max_chunks:
        zips = zips[:max_chunks]

    if not zips:
        return 0, 0.0, 0

    # Parallel extraction
    timeouts = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        futures = {
            pool.submit(_extract_one_zip, zf, extract_to, suffix): zf
            for zf in zips
        }
        for fut in as_completed(futures):
            basename, ok, err = fut.result()
            if not ok:
                if err == "timeout":
                    timeouts += 1
                else:
                    errors += 1

    if timeouts:
        print(f"    [WARN] {timeouts} zip(s) timed out", flush=True)
    if errors:
        print(f"    [WARN] {errors} zip(s) errored", flush=True)

    # Validate — remove truncated parquets
    valid_count, removed = validate_parquets(extract_to, suffix)
    if removed:
        print(f"    [INFO] removed {removed} corrupted parquet(s)", flush=True)

    # Tally bytes
    valid_files = glob.glob(os.path.join(extract_to, f"*{suffix}"))
    total_bytes = sum(os.path.getsize(f) for f in valid_files)

    return valid_count, time.time() - t0, total_bytes


# ── Registration ─────────────────────────────────────────────────────
def register_table(spark, ns: str, table: str, parquet_path: str,
                   suffix: str = ".parquet") -> Tuple[int, float]:
    """Register parquet files as Iceberg table. Returns (rows, elapsed_s)."""
    t0 = time.time()
    full = f"{CAT}.{ns}.{table}"

    # Collect files
    if os.path.isdir(parquet_path):
        files = sorted(glob.glob(
            os.path.join(parquet_path, f"**/*{suffix}"), recursive=True
        ))
    else:
        files = [parquet_path]

    if not files:
        return 0, time.time() - t0

    # Create empty table from first file's schema
    sample_dir = tempfile.mkdtemp(prefix="pq_schema_")
    os.symlink(files[0], os.path.join(sample_dir, "sample.parquet"))
    df = spark.read.parquet(f"file://{sample_dir}")
    df.limit(0).writeTo(full).using("iceberg").tableProperty(
        "format-version", "2"
    ).createOrReplace()
    shutil.rmtree(sample_dir)

    # Batch add_files() — persistent dirs under extract path survive until teardown
    total = len(files)
    num_batches = (total + ADD_FILES_BATCH - 1) // ADD_FILES_BATCH

    for i in range(0, total, ADD_FILES_BATCH):
        batch = files[i:i + ADD_FILES_BATCH]
        batch_num = i // ADD_FILES_BATCH + 1

        if num_batches == 1:
            reg_dir = os.path.dirname(files[0])
        else:
            reg_dir = os.path.join(parquet_path, f"_batch_{batch_num}")
            os.makedirs(reg_dir, exist_ok=True)
            for pf in batch:
                link = os.path.join(reg_dir, os.path.basename(pf))
                if not os.path.exists(link):
                    os.symlink(pf, link)

        spark.sql(
            f"CALL {CAT}.system.add_files("
            f"  table => '{full}',"
            f"  source_table => '`parquet`.`file://{reg_dir}`'"
            f")"
        )
        if num_batches > 1:
            print(f"      add_files batch {batch_num}/{num_batches}: "
                  f"{len(batch)} files", flush=True)

    count = spark.table(full).count()
    return count, time.time() - t0


# ── Silver/Gold views ────────────────────────────────────────────────
CLIP_RE = (
    "regexp_extract(input_file_name(), "
    "'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', 1)"
)


def create_views(spark, ns_bronze, ns_silver, ns_gold,
                 bronze_tables, radar_tables):
    """Create Silver + Gold views. Returns elapsed_s."""
    t0 = time.time()

    for table in bronze_tables:
        bronze_full = f"{CAT}.{ns_bronze}.{table}"
        silver_full = f"{CAT}.{ns_silver}.{table}"
        cols = [f.name for f in spark.table(bronze_full).schema.fields]
        if "clip_id" in cols:
            sql = f"SELECT * FROM {bronze_full}"
        else:
            sql = f"SELECT *, {CLIP_RE} AS clip_id FROM {bronze_full}"
        spark.sql(f"CREATE OR REPLACE VIEW {silver_full} AS {sql}")

    # Gold: egomotion_summary
    try:
        spark.table(f"{CAT}.{ns_silver}.egomotion")
        sql = (
            f"SELECT clip_id, count(*) AS ego_count "
            f"FROM {CAT}.{ns_silver}.egomotion "
            f"WHERE clip_id != '' GROUP BY clip_id"
        )
        spark.sql(f"CREATE OR REPLACE VIEW {CAT}.{ns_gold}.egomotion_summary AS {sql}")
    except Exception:
        pass

    # Gold: radar_ego_fusion
    if radar_tables:
        unions = [
            f"SELECT *, '{rt}' AS sensor_name FROM {CAT}.{ns_silver}.{rt}"
            for rt in radar_tables
        ]
        union_sql = " UNION ALL ".join(unions)
        try:
            sql = (
                f"SELECT r.*, e.ego_count "
                f"FROM ({union_sql}) r "
                f"LEFT JOIN ("
                f"  SELECT clip_id, count(*) AS ego_count "
                f"  FROM {CAT}.{ns_silver}.egomotion GROUP BY clip_id"
                f") e ON r.clip_id = e.clip_id"
            )
            spark.sql(f"CREATE OR REPLACE VIEW {CAT}.{ns_gold}.radar_ego_fusion AS {sql}")
        except Exception:
            pass

    # Gold: sensor_fusion_clip
    try:
        spark.table(f"{CAT}.{ns_silver}.clip_index")
        spark.sql(
            f"CREATE OR REPLACE VIEW {CAT}.{ns_gold}.sensor_fusion_clip "
            f"AS SELECT * FROM {CAT}.{ns_silver}.clip_index"
        )
    except Exception:
        pass

    return time.time() - t0


# ── Query benchmarks ─────────────────────────────────────────────────
def run_queries(spark, ns_bronze, ns_silver, ns_gold,
                bronze_tables, radar_tables) -> Dict:
    results = {}

    def timed_query(name, sql):
        try:
            t0 = time.time()
            rows = spark.sql(sql).collect()
            results[name] = {
                "elapsed_s": round(time.time() - t0, 4),
                "rows": len(rows),
            }
        except Exception as e:
            results[name] = {"elapsed_s": -1, "error": str(e)[:200]}

    # Bronze tier
    if "clip_index" in bronze_tables:
        timed_query("bronze_clip_count",
                    f"SELECT count(*) FROM {CAT}.{ns_bronze}.clip_index")

    radar_t = next((t for t in bronze_tables if "radar" in t), None)
    if radar_t:
        timed_query("bronze_radar_count",
                    f"SELECT count(*) FROM {CAT}.{ns_bronze}.{radar_t}")

    # Silver tier
    if "clip_index" in bronze_tables:
        timed_query("silver_clip_count",
                    f"SELECT count(*) FROM {CAT}.{ns_silver}.clip_index")

    if radar_t:
        timed_query("silver_radar_clip_agg",
                    f"SELECT clip_id, count(*) AS n FROM {CAT}.{ns_silver}.{radar_t} "
                    f"WHERE clip_id != '' GROUP BY clip_id LIMIT 20")

    # Gold tier
    for gold_table in ["egomotion_summary", "radar_ego_fusion",
                       "sensor_fusion_clip"]:
        try:
            spark.table(f"{CAT}.{ns_gold}.{gold_table}")
            timed_query(f"gold_{gold_table}_count",
                        f"SELECT count(*) FROM {CAT}.{ns_gold}.{gold_table}")
            timed_query(f"gold_{gold_table}_sample",
                        f"SELECT * FROM {CAT}.{ns_gold}.{gold_table} LIMIT 10")
        except Exception:
            pass

    return results


# ── Teardown ─────────────────────────────────────────────────────────
def teardown(spark, ns_bronze, ns_silver, ns_gold):
    for ns in [ns_gold, ns_silver, ns_bronze]:
        try:
            views = [r[1] for r in
                     spark.sql(f"SHOW VIEWS IN {CAT}.{ns}").collect()]
            for v in views:
                spark.sql(f"DROP VIEW IF EXISTS {CAT}.{ns}.{v}")
        except Exception:
            pass
        try:
            tables = [r[1] for r in
                      spark.sql(f"SHOW TABLES IN {CAT}.{ns}").collect()]
            for t in tables:
                spark.sql(f"DROP TABLE IF EXISTS {CAT}.{ns}.{t} PURGE")
        except Exception:
            pass


# ── Single scale-factor run ──────────────────────────────────────────
def run_scale_factor(spark, max_chunks: int) -> Dict:
    ns_bronze = f"nvidia_bench_bronze_{max_chunks}"
    ns_silver = f"nvidia_bench_silver_{max_chunks}"
    ns_gold = f"nvidia_bench_gold_{max_chunks}"

    print(f"\n{'#'*70}")
    print(f"# SCALE FACTOR: {max_chunks} chunks per sensor")
    print(f"# Namespaces: {ns_bronze} / {ns_silver} / {ns_gold}")
    print(f"{'#'*70}\n")

    for ns in [ns_bronze, ns_silver, ns_gold]:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CAT}.{ns}")

    result = {
        "max_chunks": max_chunks,
        "tables": {},
        "total_extract_s": 0,
        "total_register_s": 0,
        "total_view_s": 0,
        "total_rows": 0,
        "total_files": 0,
        "total_bytes": 0,
        "rss_mb_peak": 0,
        "queries": {},
    }

    all_bronze_tables = []
    radar_tables = []

    # ── Phase 1: Bare Parquets (direct NFS read, no extraction) ─
    print("[PHASE 1] Register bare Parquet tables from NFS")
    bare_tables = {
        "clip_index": f"{NFS_ROOT}/clip_index.parquet",
        "data_collection": f"{NFS_ROOT}/metadata/data_collection.parquet",
        "sensor_presence": f"{NFS_ROOT}/metadata/sensor_presence.parquet",
        "camera_intrinsics": f"{NFS_ROOT}/calibration/camera_intrinsics",
        "sensor_extrinsics": f"{NFS_ROOT}/calibration/sensor_extrinsics",
        "vehicle_dimensions": f"{NFS_ROOT}/calibration/vehicle_dimensions",
    }

    for table, nfs_path in bare_tables.items():
        if not os.path.exists(nfs_path):
            print(f"  [SKIP] {table}: path not found")
            continue
        print(f"  [{table}]...", end="", flush=True)
        try:
            rows, elapsed = register_table(spark, ns_bronze, table, nfs_path)
            result["tables"][table] = {
                "rows": rows, "register_s": round(elapsed, 2),
                "type": "bare_nfs",
            }
            result["total_register_s"] += elapsed
            result["total_rows"] += rows
            all_bronze_tables.append(table)
            print(f" {rows:,} rows in {elapsed:.1f}s")
        except Exception as e:
            print(f" FAILED: {e}")

    # ── Phase 2: Extract ALL sensors in parallel, then register ─
    print(f"\n[PHASE 2] Extract & register sensor data "
          f"({max_chunks} chunks/sensor)")

    extract_base = os.path.join(EXTRACT_ROOT, f"scale_{max_chunks}")
    os.makedirs(extract_base, exist_ok=True)

    # Sensor definitions
    sensors = []

    ego_dir = os.path.join(NFS_ROOT, "labels", "egomotion")
    if os.path.isdir(ego_dir):
        sensors.append(("egomotion", ego_dir, ".parquet"))

    # All radar sensors
    radar_dir = os.path.join(NFS_ROOT, "radar")
    if os.path.isdir(radar_dir):
        for sub in sorted(os.listdir(radar_dir)):
            full = os.path.join(radar_dir, sub)
            if os.path.isdir(full):
                sensors.append((sub.replace("-", "_"), full, ".parquet"))

    print(f"  {len(sensors)} sensors to process")

    # Phase 2a: Extract ALL sensors in parallel (cross-sensor parallelism)
    # This maximizes NFS throughput by issuing concurrent reads across
    # different zip directories simultaneously.
    print(f"  [EXTRACT] Starting parallel extraction across all sensors...")
    extract_t0 = time.time()

    def _extract_sensor_task(args):
        table, nfs_dir, suffix = args
        sensor_extract = os.path.join(extract_base, table)
        nf, et, nb = extract_sensor(nfs_dir, sensor_extract, max_chunks, suffix)
        return table, nf, et, nb

    sensor_results = {}
    # Use up to 4 concurrent sensor extractions (each with 8 internal workers)
    # to saturate NFS without overwhelming it
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_extract_sensor_task, s): s[0]
            for s in sensors
        }
        for fut in as_completed(futures):
            table, nf, et, nb = fut.result()
            sensor_results[table] = (nf, et, nb)
            print(f"    [{table}] {nf} files ({nb/1048576:.0f} MB) "
                  f"in {et:.1f}s", flush=True)

    extract_wall = time.time() - extract_t0
    print(f"  [EXTRACT] All sensors extracted in {extract_wall:.1f}s wall-clock")

    # Phase 2b: Register each sensor sequentially (Spark is single-JVM)
    print(f"  [REGISTER] Registering sensors...")
    for table, nfs_dir, suffix in sensors:
        nf, et, nb = sensor_results.get(table, (0, 0, 0))
        result["total_extract_s"] += et
        result["total_files"] += nf
        result["total_bytes"] += nb

        if nf == 0:
            continue

        sensor_extract = os.path.join(extract_base, table)
        print(f"  [{table}] registering...", end="", flush=True)
        try:
            rows, reg_time = register_table(
                spark, ns_bronze, table, sensor_extract, suffix
            )
            result["tables"][table] = {
                "rows": rows,
                "register_s": round(reg_time, 2),
                "extract_s": round(et, 2),
                "files": nf,
                "bytes": nb,
                "type": "zip_extract",
            }
            result["total_register_s"] += reg_time
            result["total_rows"] += rows
            all_bronze_tables.append(table)
            if table.startswith("radar_"):
                radar_tables.append(table)
            print(f" {rows:,} rows in {reg_time:.1f}s")
        except Exception as e:
            print(f" FAILED: {e}")

    result["rss_mb_after_register"] = round(rss_mb(), 1)

    # ── Phase 3: Views ──────────────────────────────────────────
    print(f"\n[PHASE 3] Create Silver + Gold views")
    view_elapsed = create_views(
        spark, ns_bronze, ns_silver, ns_gold, all_bronze_tables, radar_tables
    )
    result["total_view_s"] = round(view_elapsed, 2)
    print(f"  Views created in {view_elapsed:.1f}s "
          f"({len(all_bronze_tables)} bronze → silver, gold joins)")

    # ── Phase 4: Query benchmark ────────────────────────────────
    print(f"\n[PHASE 4] Query latency benchmarks")

    # Warm-up
    run_queries(spark, ns_bronze, ns_silver, ns_gold,
                all_bronze_tables, radar_tables)
    print("  (warm-up done)")

    # 3 timed runs → median
    all_runs = []
    for run_num in range(3):
        qr = run_queries(spark, ns_bronze, ns_silver, ns_gold,
                         all_bronze_tables, radar_tables)
        all_runs.append(qr)
        print(f"  Run {run_num+1}/3 done")

    median_queries = {}
    for qname in all_runs[0]:
        latencies = sorted(
            r[qname]["elapsed_s"] for r in all_runs
            if qname in r and r[qname]["elapsed_s"] >= 0
        )
        if latencies:
            mid = len(latencies) // 2
            median_queries[qname] = {
                "median_s": latencies[mid],
                "min_s": latencies[0],
                "max_s": latencies[-1],
            }
    result["queries"] = median_queries
    result["rss_mb_peak"] = round(rss_mb(), 1)

    print(f"\n  {'Query':<35} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for qname, ql in sorted(median_queries.items()):
        print(f"  {qname:<35} {ql['median_s']:>7.3f}s "
              f"{ql['min_s']:>7.3f}s {ql['max_s']:>7.3f}s")

    # ── Cleanup ─────────────────────────────────────────────────
    print(f"\n[TEARDOWN] scale={max_chunks}")
    teardown(spark, ns_bronze, ns_silver, ns_gold)
    # Clean extracted data to free disk
    shutil.rmtree(extract_base, ignore_errors=True)
    # Clean any leftover symlink temp dirs
    for d in glob.glob("/tmp/pq_*"):
        shutil.rmtree(d, ignore_errors=True)
    print("  Done")

    return result


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=str, default="2,10,50,200",
                        help="Comma-separated chunk counts per sensor")
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",")]

    print(f"\n{'='*70}")
    print("NVIDIA PHYSICALAI — MULTI-SCALE SCALABILITY BENCHMARK")
    print(f"  Scale factors (chunks/sensor): {scales}")
    print(f"  NFS source: {NFS_ROOT}")
    print(f"  Extract target: {EXTRACT_ROOT}")
    print(f"  Strategy: parallel zip extract → add_files() → views → queries → teardown")
    print(f"  Parallel extraction workers: {EXTRACT_WORKERS}")
    print(f"{'='*70}")

    spark = build_spark()
    all_results = []

    for sf in scales:
        t0 = time.time()
        try:
            result = run_scale_factor(spark, sf)
            result["wall_clock_s"] = round(time.time() - t0, 1)
            all_results.append(result)
            print(f"\n  Scale {sf}: {result['wall_clock_s']}s total, "
                  f"{result['total_rows']:,} rows, "
                  f"{result['total_bytes']/1073741824:.1f} GB")
        except Exception as e:
            print(f"\n[ERROR] Scale factor {sf} failed: {e}")
            import traceback
            traceback.print_exc()

    spark.stop()

    # ── Scalability Analysis ─────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nfs_root": NFS_ROOT,
        "strategy": "parallel_zip_extract_to_local + batched_add_files",
        "extract_workers": EXTRACT_WORKERS,
        "scale_factors": all_results,
        "projections": {},
    }

    if len(all_results) >= 2:
        # Linear regression on multiple points (least squares)
        def linear_fit(points):
            n = len(points)
            sx = sum(p[0] for p in points)
            sy = sum(p[1] for p in points)
            sxx = sum(p[0]**2 for p in points)
            sxy = sum(p[0]*p[1] for p in points)
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-12:
                return 0, 0
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            return slope, intercept

        # Registration scaling: rows → register time
        reg_points = [(r["total_rows"], r["total_register_s"])
                      for r in all_results if r["total_rows"] > 0]
        if len(reg_points) >= 2:
            reg_slope, reg_intercept = linear_fit(reg_points)

            # Extraction scaling: bytes → extract time
            ext_points = [(r["total_bytes"], r["total_extract_s"])
                          for r in all_results if r["total_bytes"] > 0]
            ext_slope, ext_intercept = linear_fit(ext_points) if len(ext_points) >= 2 else (0, 0)

            # Full dataset: ~119 TB
            full_rows_est = 20_000_000_000  # ~20B rows estimated
            full_bytes = 119 * 1024**4

            # Scale from observed data
            if all_results:
                biggest = max(all_results, key=lambda r: r["total_rows"])
                if biggest["total_rows"] > 0:
                    row_scale = full_rows_est / biggest["total_rows"]
                    byte_scale = full_bytes / biggest["total_bytes"] if biggest["total_bytes"] > 0 else 0

            proj_register_s = reg_slope * full_rows_est + reg_intercept
            proj_extract_s = ext_slope * full_bytes + ext_intercept if ext_slope else 0

            report["projections"]["full_dataset_119TB"] = {
                "estimated_rows": full_rows_est,
                "estimated_bytes_TB": 119,
                "projected_register_h": round(proj_register_s / 3600, 2),
                "projected_extract_h": round(proj_extract_s / 3600, 2),
                "projected_total_h": round(
                    (proj_register_s + proj_extract_s) / 3600, 2
                ),
                "register_slope_s_per_row": reg_slope,
                "extract_slope_s_per_byte": ext_slope,
                "note": "Linear extrapolation from multi-point benchmark",
            }

            # Petabyte projection (10× full dataset)
            pb_rows = full_rows_est * 10
            pb_bytes = full_bytes * 10
            report["projections"]["petabyte_1PB"] = {
                "estimated_rows": pb_rows,
                "estimated_bytes_PB": 1.19,
                "projected_register_h": round(
                    (reg_slope * pb_rows + reg_intercept) / 3600, 2
                ),
                "projected_extract_h": round(
                    (ext_slope * pb_bytes + ext_intercept) / 3600, 2
                ) if ext_slope else "N/A",
                "note": "10x full dataset projection — linear scaling assumed",
            }

        # Query scaling analysis
        if all_results[0].get("queries") and all_results[-1].get("queries"):
            q_first = all_results[0]["queries"]
            q_last = all_results[-1]["queries"]
            data_scale = (all_results[-1]["total_rows"] /
                          all_results[0]["total_rows"]
                          if all_results[0]["total_rows"] else 1)
            query_scaling = {}
            for qname in q_first:
                if qname in q_last:
                    s_lat = q_first[qname]["median_s"]
                    l_lat = q_last[qname]["median_s"]
                    if s_lat > 0:
                        ratio = l_lat / s_lat
                        query_scaling[qname] = {
                            "small_scale_s": round(s_lat, 4),
                            "large_scale_s": round(l_lat, 4),
                            "data_scale_factor": round(data_scale, 1),
                            "latency_ratio": round(ratio, 3),
                            "scaling": ("constant" if ratio < 2.0
                                        else "sublinear" if ratio < data_scale * 0.5
                                        else "linear"),
                        }
            report["query_scaling_analysis"] = query_scaling

    # ── Print Summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SCALABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scale':>8} {'Rows':>14} {'Files':>8} {'Data_GB':>9} "
          f"{'Extract_s':>10} {'Register_s':>11} {'Views_s':>8} "
          f"{'RSS_MB':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['max_chunks']:>8} {r['total_rows']:>14,} "
              f"{r['total_files']:>8,} "
              f"{r['total_bytes']/1073741824:>8.1f} "
              f"{r['total_extract_s']:>9.1f}s "
              f"{r['total_register_s']:>10.1f}s "
              f"{r['total_view_s']:>7.1f}s "
              f"{r['rss_mb_peak']:>7.0f}")

    if report.get("projections"):
        print(f"\nPROJECTED TIMES:")
        for label, proj in report["projections"].items():
            if "projected_total_h" in proj:
                print(f"  {label}: {proj['projected_total_h']:.1f}h total "
                      f"(register: {proj['projected_register_h']:.1f}h, "
                      f"extract: {proj['projected_extract_h']:.1f}h)")

    if report.get("query_scaling_analysis"):
        print(f"\nQUERY SCALING ({all_results[0]['max_chunks']}→"
              f"{all_results[-1]['max_chunks']} chunks):")
        for qname, qs in sorted(report["query_scaling_analysis"].items()):
            print(f"  {qname:<35} {qs['small_scale_s']:.3f}s → "
                  f"{qs['large_scale_s']:.3f}s "
                  f"(ratio: {qs['latency_ratio']:.2f}× — {qs['scaling']})")

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
