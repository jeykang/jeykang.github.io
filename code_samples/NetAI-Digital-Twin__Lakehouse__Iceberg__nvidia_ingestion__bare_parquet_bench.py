#!/usr/bin/env python3
"""
Bare-parquet scalability benchmark — no extraction, no zip dependencies.

Registers subsets of the calibration parquet files (3,116 files per sensor)
at increasing scale factors to prove linear add_files() scaling and constant
query latency. Works directly from NFS without extraction.

Usage (via spark-submit inside spark-iceberg container):
    spark-submit --driver-memory 4g nvidia_ingestion/bare_parquet_bench.py
"""

import glob
import json
import os
import resource
import shutil
import tempfile
import time
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession

# ── Constants ────────────────────────────────────────────────────────
NFS_ROOT = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
REPORT_FILE = "/tmp/bare_parquet_scalability.json"
CAT = "iceberg"
ADD_FILES_BATCH = 5_000

# Scale factors: number of parquet files per table to register
SCALES = [50, 200, 500, 1000, 2000, 3116]

# Tables with bare parquet files (no extraction needed)
BARE_TABLES = {
    "camera_intrinsics": f"{NFS_ROOT}/calibration/camera_intrinsics",
    "sensor_extrinsics": f"{NFS_ROOT}/calibration/sensor_extrinsics",
    "vehicle_dimensions": f"{NFS_ROOT}/calibration/vehicle_dimensions",
}

METADATA_TABLES = {
    "data_collection": f"{NFS_ROOT}/metadata/data_collection.parquet",
    "sensor_presence": f"{NFS_ROOT}/metadata/sensor_presence.parquet",
}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_spark():
    return (
        SparkSession.builder.appName("nvidia-bare-parquet-bench")
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


def register_table_subset(spark, ns: str, table: str, source_dir: str,
                          max_files: int) -> Tuple[int, float, int]:
    """Register a subset of parquet files. Returns (rows, elapsed_s, file_count)."""
    t0 = time.time()
    full = f"{CAT}.{ns}.{table}"

    # Collect up to max_files parquets
    if os.path.isdir(source_dir):
        all_files = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
    elif source_dir.endswith(".parquet"):
        all_files = [source_dir]
    else:
        return 0, 0, 0

    files = all_files[:max_files]
    if not files:
        return 0, 0, 0

    # Create empty table from schema
    sample_dir = tempfile.mkdtemp(prefix="pq_schema_")
    os.symlink(files[0], os.path.join(sample_dir, "sample.parquet"))
    df = spark.read.parquet(f"file://{sample_dir}")
    df.limit(0).writeTo(full).using("iceberg").tableProperty(
        "format-version", "2"
    ).createOrReplace()
    shutil.rmtree(sample_dir)

    # Batch add_files
    total = len(files)
    for i in range(0, total, ADD_FILES_BATCH):
        batch = files[i:i + ADD_FILES_BATCH]
        batch_num = i // ADD_FILES_BATCH + 1

        # Create persistent symlink dir for this batch
        batch_dir = tempfile.mkdtemp(prefix=f"pq_{table}_b{batch_num}_")
        for pf in batch:
            link = os.path.join(batch_dir, os.path.basename(pf))
            if not os.path.exists(link):
                os.symlink(pf, link)

        spark.sql(
            f"CALL {CAT}.system.add_files("
            f"  table => '{full}',"
            f"  source_table => '`parquet`.`file://{batch_dir}`'"
            f")"
        )

    count = spark.table(full).count()
    return count, time.time() - t0, len(files)


CLIP_RE = (
    "regexp_extract(input_file_name(), "
    "'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', 1)"
)


def run_scale(spark, scale: int) -> Dict:
    ns_bronze = f"nvidia_bare_bronze_{scale}"
    ns_silver = f"nvidia_bare_silver_{scale}"
    ns_gold = f"nvidia_bare_gold_{scale}"

    print(f"\n{'='*60}")
    print(f"SCALE: {scale} files per table")
    print(f"{'='*60}")

    for ns in [ns_bronze, ns_silver, ns_gold]:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CAT}.{ns}")

    result = {
        "scale": scale,
        "tables": {},
        "total_rows": 0,
        "total_register_s": 0,
        "total_files": 0,
        "rss_mb": 0,
        "queries": {},
    }

    all_tables = []

    # Register calibration tables
    for table, path in BARE_TABLES.items():
        print(f"  [{table}] registering {scale} files...", end="", flush=True)
        try:
            rows, elapsed, nf = register_table_subset(
                spark, ns_bronze, table, path, scale
            )
            result["tables"][table] = {
                "rows": rows, "register_s": round(elapsed, 2), "files": nf,
            }
            result["total_rows"] += rows
            result["total_register_s"] += elapsed
            result["total_files"] += nf
            all_tables.append(table)
            print(f" {rows:,} rows, {nf} files in {elapsed:.1f}s")
        except Exception as e:
            print(f" FAILED: {e}")

    # Register metadata tables (single files, constant across scales)
    for table, path in METADATA_TABLES.items():
        print(f"  [{table}]...", end="", flush=True)
        try:
            rows, elapsed, nf = register_table_subset(
                spark, ns_bronze, table, path, 1
            )
            result["tables"][table] = {
                "rows": rows, "register_s": round(elapsed, 2), "files": nf,
            }
            result["total_rows"] += rows
            result["total_register_s"] += elapsed
            result["total_files"] += nf
            all_tables.append(table)
            print(f" {rows:,} rows in {elapsed:.1f}s")
        except Exception as e:
            print(f" FAILED: {e}")

    # Silver views
    vt0 = time.time()
    for table in all_tables:
        bronze_full = f"{CAT}.{ns_bronze}.{table}"
        silver_full = f"{CAT}.{ns_silver}.{table}"
        cols = [f.name for f in spark.table(bronze_full).schema.fields]
        if "clip_id" in cols:
            sql = f"SELECT * FROM {bronze_full}"
        else:
            sql = f"SELECT *, {CLIP_RE} AS clip_id FROM {bronze_full}"
        spark.sql(f"CREATE OR REPLACE VIEW {silver_full} AS {sql}")

    # Gold view: cross-calibration join
    calib_tables = [t for t in all_tables if t in BARE_TABLES]
    if len(calib_tables) >= 2:
        unions = [
            f"SELECT *, '{t}' AS calib_type FROM {CAT}.{ns_silver}.{t}"
            for t in calib_tables
        ]
        union_sql = " UNION ALL ".join(unions)
        spark.sql(
            f"CREATE OR REPLACE VIEW {CAT}.{ns_gold}.calibration_union "
            f"AS {union_sql}"
        )

    result["view_creation_s"] = round(time.time() - vt0, 2)
    print(f"  Views created in {result['view_creation_s']:.1f}s")

    # Query benchmarks (4 runs, take median of last 3)
    def timed_query(name, sql):
        try:
            t0 = time.time()
            rows = spark.sql(sql).collect()
            return round(time.time() - t0, 4), len(rows)
        except Exception:
            return -1, 0

    queries = {
        "bronze_count": f"SELECT count(*) FROM {CAT}.{ns_bronze}.{calib_tables[0]}" if calib_tables else None,
        "bronze_sample": f"SELECT * FROM {CAT}.{ns_bronze}.{calib_tables[0]} LIMIT 100" if calib_tables else None,
        "silver_count": f"SELECT count(*) FROM {CAT}.{ns_silver}.{calib_tables[0]}" if calib_tables else None,
        "silver_clip_agg": f"SELECT clip_id, count(*) FROM {CAT}.{ns_silver}.{calib_tables[0]} WHERE clip_id != '' GROUP BY clip_id LIMIT 20" if calib_tables else None,
        "gold_union_count": f"SELECT count(*) FROM {CAT}.{ns_gold}.calibration_union",
        "gold_union_sample": f"SELECT * FROM {CAT}.{ns_gold}.calibration_union LIMIT 100",
    }

    # Warm-up
    for name, sql in queries.items():
        if sql:
            timed_query(name, sql)
    print("  (warm-up done)")

    # 3 timed runs
    all_runs = []
    for run in range(3):
        run_results = {}
        for name, sql in queries.items():
            if sql:
                lat, nrows = timed_query(name, sql)
                run_results[name] = lat
        all_runs.append(run_results)
        print(f"  Run {run+1}/3 done")

    # Median
    median_queries = {}
    for name in queries:
        if queries[name] is None:
            continue
        lats = sorted(r.get(name, -1) for r in all_runs if r.get(name, -1) >= 0)
        if lats:
            mid = len(lats) // 2
            median_queries[name] = {
                "median_s": lats[mid],
                "min_s": lats[0],
                "max_s": lats[-1],
            }
    result["queries"] = median_queries

    print(f"\n  {'Query':<25} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*50}")
    for qname, ql in sorted(median_queries.items()):
        print(f"  {qname:<25} {ql['median_s']:>7.3f}s "
              f"{ql['min_s']:>7.3f}s {ql['max_s']:>7.3f}s")

    result["rss_mb"] = round(rss_mb(), 1)

    # Teardown
    print(f"  [TEARDOWN]")
    for ns in [ns_gold, ns_silver, ns_bronze]:
        try:
            for r in spark.sql(f"SHOW VIEWS IN {CAT}.{ns}").collect():
                spark.sql(f"DROP VIEW IF EXISTS {CAT}.{ns}.{r[1]}")
        except Exception:
            pass
        try:
            for r in spark.sql(f"SHOW TABLES IN {CAT}.{ns}").collect():
                spark.sql(f"DROP TABLE IF EXISTS {CAT}.{ns}.{r[1]} PURGE")
        except Exception:
            pass

    # Clean temp symlink dirs
    for d in glob.glob("/tmp/pq_*"):
        shutil.rmtree(d, ignore_errors=True)

    return result


def main():
    print(f"\n{'#'*60}")
    print("BARE PARQUET SCALABILITY BENCHMARK")
    print(f"Scales: {SCALES}")
    print(f"Tables: {list(BARE_TABLES.keys())} + {list(METADATA_TABLES.keys())}")
    print(f"{'#'*60}")

    spark = build_spark()
    all_results = []

    for scale in SCALES:
        try:
            result = run_scale(spark, scale)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Scale {scale} failed: {e}")
            import traceback
            traceback.print_exc()

    spark.stop()

    # Analysis
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "strategy": "bare_parquet_add_files_from_nfs",
        "scales": all_results,
    }

    # Linear regression
    if len(all_results) >= 2:
        points = [(r["total_files"], r["total_register_s"])
                  for r in all_results if r["total_files"] > 0]
        if len(points) >= 2:
            n = len(points)
            sx = sum(p[0] for p in points)
            sy = sum(p[1] for p in points)
            sxx = sum(p[0]**2 for p in points)
            sxy = sum(p[0]*p[1] for p in points)
            denom = n * sxx - sx * sx
            if abs(denom) > 1e-12:
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
                report["regression"] = {
                    "slope_s_per_file": slope,
                    "intercept_s": intercept,
                    "files_per_second": round(1/slope, 1) if slope > 0 else 0,
                }
                # Project to full dataset (20 sensors × ~300K files = 6M files)
                full_files = 6_000_000
                pb_files = 60_000_000
                report["projections"] = {
                    "full_119TB": {
                        "files": full_files,
                        "projected_register_h": round((slope * full_files + intercept) / 3600, 2),
                    },
                    "petabyte_1PB": {
                        "files": pb_files,
                        "projected_register_h": round((slope * pb_files + intercept) / 3600, 2),
                    },
                }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scale':>8} {'Files':>8} {'Rows':>14} {'Register_s':>11} {'RSS_MB':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['scale']:>8} {r['total_files']:>8} "
              f"{r['total_rows']:>14,} "
              f"{r['total_register_s']:>10.1f}s "
              f"{r['rss_mb']:>7.0f}")

    if report.get("regression"):
        reg = report["regression"]
        print(f"\nRegression: {reg['files_per_second']} files/s")

    if report.get("projections"):
        for label, proj in report["projections"].items():
            print(f"  {label}: {proj['projected_register_h']:.1f}h "
                  f"for {proj['files']:,} files")

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
