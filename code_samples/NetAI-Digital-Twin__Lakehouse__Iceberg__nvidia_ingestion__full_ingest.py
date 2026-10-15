#!/usr/bin/env python3
"""
Full zero-copy ingestion: mount zips via ratarmount, register via add_files().

Runs INSIDE the spark-iceberg container with FUSE support.

Usage:
    python nvidia_ingestion/full_ingest.py [--max-sensors N] [--max-chunks N]
"""

import argparse
import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── PySpark imports ──────────────────────────────────────────────────
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"
from pyspark.sql import SparkSession


# ── Constants ────────────────────────────────────────────────────────
NFS_ROOT = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
FUSE_ROOT = "/tmp/nvidia-fuse"
BENCHMARK_FILE = "/tmp/full_ingest_benchmark.json"


# ── Benchmark tracking ───────────────────────────────────────────────
@dataclass
class StepMetric:
    name: str
    phase: str
    start: float = 0.0
    elapsed: float = 0.0
    rows: int = 0
    files: int = 0
    bytes_total: int = 0
    rss_mb: float = 0.0


class Benchmark:
    def __init__(self):
        self.steps: List[StepMetric] = []
        self.t0 = time.time()

    def _rss_mb(self) -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def begin(self, phase: str, name: str) -> StepMetric:
        m = StepMetric(name=name, phase=phase, start=time.time())
        self.steps.append(m)
        return m

    def end(self, m: StepMetric, rows=0, files=0, bytes_total=0):
        m.elapsed = time.time() - m.start
        m.rows = rows
        m.files = files
        m.bytes_total = bytes_total
        m.rss_mb = self._rss_mb()

    def total_elapsed(self) -> float:
        return time.time() - self.t0

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "FULL INGESTION BENCHMARK SUMMARY",
            f"{'='*70}",
            f"{'Phase':<12} {'Table':<30} {'Rows':>10} {'Files':>7} {'Time':>8} {'RSS_MB':>8}",
            "-" * 70,
        ]
        for s in self.steps:
            lines.append(
                f"{s.phase:<12} {s.name:<30} {s.rows:>10} {s.files:>7} "
                f"{s.elapsed:>7.1f}s {s.rss_mb:>7.0f}"
            )
        lines.append("-" * 70)
        lines.append(f"Total elapsed: {self.total_elapsed():.1f}s")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "total_elapsed_s": self.total_elapsed(),
            "steps": [
                {
                    "phase": s.phase,
                    "name": s.name,
                    "elapsed_s": s.elapsed,
                    "rows": s.rows,
                    "files": s.files,
                    "bytes_total": s.bytes_total,
                    "rss_mb": s.rss_mb,
                }
                for s in self.steps
            ],
        }


# ── FUSE mount helpers ───────────────────────────────────────────────
def mount_sensor(sensor_dir: str, mount_point: str,
                 max_chunks: int = 0,
                 batch_size: int = 100) -> Tuple[bool, int]:
    """Mount a sensor directory's zip files via ratarmount in batches.

    For large directories (3000+ zips), mounts in batches using separate
    mount points and collects all Parquet files via symlinks.

    Returns (success, num_parquet_files).
    """
    if os.path.ismount(mount_point):
        parquets = glob.glob(os.path.join(mount_point, "**/*.parquet"),
                             recursive=True)
        return True, len(parquets)

    os.makedirs(mount_point, exist_ok=True)

    # Collect zip files
    zips = sorted(glob.glob(os.path.join(sensor_dir, "*.zip")))
    if not zips:
        return False, 0

    if max_chunks and len(zips) > max_chunks:
        zips = zips[:max_chunks]

    if len(zips) <= batch_size:
        # Small enough for a single mount
        cmd = ["ratarmount"] + zips + [mount_point]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                print(f"    [FAIL] ratarmount: {result.stderr[:500]}")
                return False, 0
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT] ratarmount >60min")
            return False, 0

        parquets = glob.glob(os.path.join(mount_point, "**/*.parquet"),
                             recursive=True)
        return True, len(parquets)

    # Large directory: mount in batches, collect via symlinks
    # Use a staging area with sub-mounts
    import tempfile
    staging = tempfile.mkdtemp(prefix="fuse_staging_")
    all_parquets = []

    for i in range(0, len(zips), batch_size):
        batch = zips[i:i + batch_size]
        batch_mp = os.path.join(staging, f"batch_{i // batch_size:04d}")
        os.makedirs(batch_mp, exist_ok=True)

        batch_num = i // batch_size + 1
        total_batches = (len(zips) + batch_size - 1) // batch_size
        print(f"    Mounting batch {batch_num}/{total_batches} "
              f"({len(batch)} zips)...", flush=True)

        cmd = ["ratarmount"] + batch + [batch_mp]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                print(f"    [WARN] batch {batch_num} failed: "
                      f"{result.stderr[:300]}")
                continue
        except subprocess.TimeoutExpired:
            print(f"    [WARN] batch {batch_num} timeout")
            continue

        batch_parquets = glob.glob(
            os.path.join(batch_mp, "**/*.parquet"), recursive=True
        )
        all_parquets.extend(batch_parquets)
        print(f"    Batch {batch_num}: {len(batch_parquets)} parquet files")

    if not all_parquets:
        return False, 0

    # Create symlinks in the mount_point for all Parquet files
    for pf in all_parquets:
        link_name = os.path.join(mount_point, os.path.basename(pf))
        if not os.path.exists(link_name):
            os.symlink(pf, link_name)

    return True, len(all_parquets)


def unmount_all():
    """Unmount all FUSE mounts under FUSE_ROOT."""
    if not os.path.isdir(FUSE_ROOT):
        return
    for root, dirs, _ in os.walk(FUSE_ROOT, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            if os.path.ismount(path):
                subprocess.run(["fusermount", "-u", path],
                               capture_output=True)


# ── Spark session ────────────────────────────────────────────────────
def build_spark():
    return (
        SparkSession.builder.appName("nvidia-full-ingest")
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
        # Tuning
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .getOrCreate()
    )


# ── Registration helpers ─────────────────────────────────────────────
CAT = "iceberg"
NS_BRONZE = "nvidia_bronze"
NS_SILVER = "nvidia_silver"
NS_GOLD = "nvidia_gold"


ADD_FILES_BATCH = 10_000  # max files per add_files() call to avoid JVM OOM


def register_table(spark, table: str, parquet_uri: str,
                   suffix_filter: str = ".parquet") -> int:
    """Create Iceberg table from Parquet via add_files() (zero-copy).

    For directories with mixed content (e.g. .mp4 + .parquet), creates a
    temp directory with symlinks to only the Parquet files, then registers
    from there.  For large directories (>ADD_FILES_BATCH files), batches
    the add_files() calls to avoid JVM heap exhaustion.
    """
    import tempfile

    full = f"{CAT}.{NS_BRONZE}.{table}"

    # Check if directory has non-parquet files that would confuse add_files()
    actual_path = parquet_uri.replace("file://", "")
    needs_filter = False
    if os.path.isdir(actual_path):
        all_files = os.listdir(actual_path)
        non_parquet = [f for f in all_files
                       if not f.endswith(suffix_filter) and not f.startswith(".")]
        if non_parquet:
            needs_filter = True

    if needs_filter:
        # Collect only matching files
        parquet_files = sorted(glob.glob(
            os.path.join(actual_path, f"**/*{suffix_filter}"), recursive=True
        ))
        if not parquet_files:
            return 0
    else:
        # Collect all parquet files in the directory
        if os.path.isdir(actual_path):
            parquet_files = sorted(glob.glob(
                os.path.join(actual_path, "**/*.parquet"), recursive=True
            ))
        else:
            parquet_files = [actual_path]

    # Create the empty Iceberg table from schema of first file
    sample_dir = tempfile.mkdtemp(prefix=f"pq_schema_{table}_")
    os.symlink(parquet_files[0], os.path.join(sample_dir, "sample.parquet"))
    df = spark.read.parquet(f"file://{sample_dir}")
    df.limit(0).writeTo(full).using("iceberg").tableProperty(
        "format-version", "2"
    ).createOrReplace()
    shutil.rmtree(sample_dir)

    # Batch the add_files() calls
    total_files = len(parquet_files)
    num_batches = (total_files + ADD_FILES_BATCH - 1) // ADD_FILES_BATCH
    print(f"    Registering {total_files} files in {num_batches} "
          f"add_files() batch(es)...", flush=True)

    for i in range(0, total_files, ADD_FILES_BATCH):
        batch = parquet_files[i:i + ADD_FILES_BATCH]
        batch_num = i // ADD_FILES_BATCH + 1

        # Create temp dir with symlinks for this batch
        batch_dir = tempfile.mkdtemp(prefix=f"pq_{table}_b{batch_num}_")
        for pf in batch:
            link_name = os.path.join(batch_dir, os.path.basename(pf))
            if not os.path.exists(link_name):
                os.symlink(pf, link_name)
        batch_uri = f"file://{batch_dir}"

        t0 = time.time()
        spark.sql(
            f"CALL {CAT}.system.add_files("
            f"  table => '{full}',"
            f"  source_table => '`parquet`.`{batch_uri}`'"
            f")"
        )
        elapsed = time.time() - t0
        print(f"    Batch {batch_num}/{num_batches}: "
              f"{len(batch)} files in {elapsed:.1f}s", flush=True)

        # Clean up batch symlinks
        shutil.rmtree(batch_dir, ignore_errors=True)

    return spark.table(full).count()


def append_to_table(spark, table: str, parquet_uri: str) -> int:
    """Append files to existing Iceberg table via add_files()."""
    full = f"{CAT}.{NS_BRONZE}.{table}"
    spark.sql(
        f"CALL {CAT}.system.add_files("
        f"  table => '{full}',"
        f"  source_table => '`parquet`.`{parquet_uri}`'"
        f")"
    )
    return spark.table(full).count()


# ── Silver views ─────────────────────────────────────────────────────
CLIP_RE = (
    "regexp_extract(input_file_name(), "
    "'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', 1)"
)


def create_silver_views(spark, bronze_tables: List[str]):
    """Create Silver views that add clip_id via input_file_name()."""
    for table in bronze_tables:
        bronze_full = f"{CAT}.{NS_BRONZE}.{table}"
        silver_full = f"{CAT}.{NS_SILVER}.{table}"

        # Check if clip_id already exists
        cols = [f.name for f in spark.table(bronze_full).schema.fields]
        if "clip_id" in cols:
            sql = f"SELECT * FROM {bronze_full}"
        else:
            sql = f"SELECT *, {CLIP_RE} AS clip_id FROM {bronze_full}"

        spark.sql(f"CREATE OR REPLACE VIEW {silver_full} AS {sql}")
        count = spark.table(silver_full).count()
        print(f"  [SILVER] {silver_full}: {count} rows (view)")


# ── Gold views ───────────────────────────────────────────────────────
def create_gold_views(spark, radar_tables: List[str]):
    """Create Gold views for analytics (zero storage)."""
    # 1. lidar_with_ego (if lidar and egomotion exist)
    try:
        spark.table(f"{CAT}.{NS_SILVER}.lidar")
        spark.table(f"{CAT}.{NS_SILVER}.egomotion")
        sql = (
            f"SELECT l.*, e.ego_count "
            f"FROM {CAT}.{NS_SILVER}.lidar l "
            f"LEFT JOIN ("
            f"  SELECT clip_id, count(*) AS ego_count "
            f"  FROM {CAT}.{NS_SILVER}.egomotion GROUP BY clip_id"
            f") e ON l.clip_id = e.clip_id"
        )
        spark.sql(f"CREATE OR REPLACE VIEW {CAT}.{NS_GOLD}.lidar_with_ego AS {sql}")
        print(f"  [GOLD] lidar_with_ego view created")
    except Exception as e:
        print(f"  [GOLD] lidar_with_ego skipped: {e}")

    # 2. radar_ego_fusion
    if radar_tables:
        unions = []
        for rt in radar_tables:
            sensor_name = rt.replace("radar_", "", 1)
            unions.append(
                f"SELECT *, '{sensor_name}' AS sensor_name "
                f"FROM {CAT}.{NS_SILVER}.{rt}"
            )
        union_sql = " UNION ALL ".join(unions)
        try:
            sql = (
                f"SELECT r.*, e.ego_count "
                f"FROM ({union_sql}) r "
                f"LEFT JOIN ("
                f"  SELECT clip_id, count(*) AS ego_count "
                f"  FROM {CAT}.{NS_SILVER}.egomotion GROUP BY clip_id"
                f") e ON r.clip_id = e.clip_id"
            )
            spark.sql(f"CREATE OR REPLACE VIEW {CAT}.{NS_GOLD}.radar_ego_fusion AS {sql}")
            print(f"  [GOLD] radar_ego_fusion view created ({len(radar_tables)} radar tables)")
        except Exception as e:
            print(f"  [GOLD] radar_ego_fusion skipped: {e}")

    # 3. sensor_fusion_clip (materialized — uses positional join)
    try:
        spark.table(f"{CAT}.{NS_SILVER}.clip_index")
        sql = f"SELECT * FROM {CAT}.{NS_SILVER}.clip_index"
        spark.sql(f"CREATE OR REPLACE VIEW {CAT}.{NS_GOLD}.sensor_fusion_clip AS {sql}")
        print(f"  [GOLD] sensor_fusion_clip view created")
    except Exception as e:
        print(f"  [GOLD] sensor_fusion_clip skipped: {e}")


# ── Query benchmarks ─────────────────────────────────────────────────
def benchmark_queries(spark, bench: Benchmark, all_tables: List[str]):
    """Run representative queries across tiers and measure latency."""
    queries = {
        # Bronze: raw scan
        "bronze_clip_count": f"SELECT count(*) FROM {CAT}.{NS_BRONZE}.clip_index",
        "bronze_radar_sample": next(
            (f"SELECT * FROM {CAT}.{NS_BRONZE}.{t} LIMIT 100"
             for t in all_tables if t.startswith("radar_")),
            None,
        ),
        # Silver: view with clip_id derivation
        "silver_clip_count": f"SELECT count(*) FROM {CAT}.{NS_SILVER}.clip_index",
        "silver_clip_filter": (
            f"SELECT count(*) FROM {CAT}.{NS_SILVER}.clip_index "
            f"WHERE clip_id IS NOT NULL"
        ),
    }

    # Add gold queries if tables exist
    for gold_table in ["lidar_with_ego", "radar_ego_fusion", "sensor_fusion_clip"]:
        try:
            spark.table(f"{CAT}.{NS_GOLD}.{gold_table}")
            queries[f"gold_{gold_table}_count"] = (
                f"SELECT count(*) FROM {CAT}.{NS_GOLD}.{gold_table}"
            )
            queries[f"gold_{gold_table}_sample"] = (
                f"SELECT * FROM {CAT}.{NS_GOLD}.{gold_table} LIMIT 10"
            )
        except Exception:
            pass

    print(f"\n{'='*70}")
    print("QUERY LATENCY BENCHMARKS")
    print(f"{'='*70}")

    for name, sql in queries.items():
        if sql is None:
            continue
        m = bench.begin("query", name)
        try:
            result = spark.sql(sql).collect()
            rows = len(result)
            bench.end(m, rows=rows)
            print(f"  {name}: {m.elapsed:.3f}s ({rows} rows)")
        except Exception as e:
            bench.end(m)
            print(f"  {name}: FAILED ({e})")


# ── Main orchestrator ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sensors", type=int, default=0,
                        help="Limit number of sensor dirs to mount (0=all)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Limit zip chunks per sensor (0=all)")
    parser.add_argument("--skip-fuse", action="store_true",
                        help="Skip FUSE mounting (use existing mounts)")
    parser.add_argument("--skip-register", action="store_true",
                        help="Skip registration (use existing tables)")
    args = parser.parse_args()

    bench = Benchmark()

    print("\n" + "=" * 70)
    print("NVIDIA PHYSICALAI DATASET — FULL ZERO-COPY INGESTION")
    print(f"  NFS source:   {NFS_ROOT}")
    print(f"  FUSE root:    {FUSE_ROOT}")
    print(f"  max_sensors:  {args.max_sensors or 'ALL'}")
    print(f"  max_chunks:   {args.max_chunks or 'ALL'}")
    print("=" * 70)

    # ── Discover sensors ──────────────────────────────────────────
    sensor_dirs: Dict[str, str] = {}  # table_name → nfs_dir

    # Radar
    radar_dir = os.path.join(NFS_ROOT, "radar")
    if os.path.isdir(radar_dir):
        for sub in sorted(os.listdir(radar_dir)):
            full = os.path.join(radar_dir, sub)
            if os.path.isdir(full):
                safe = sub.replace("-", "_")
                # Avoid double-prefix: "radar_front_center_srr_0" not "radar_radar_front_center_srr_0"
                sensor_dirs[safe] = full

    # Lidar
    lidar_dir = os.path.join(NFS_ROOT, "lidar", "lidar_top_360fov")
    if os.path.isdir(lidar_dir):
        sensor_dirs["lidar"] = lidar_dir

    # Egomotion
    ego_dir = os.path.join(NFS_ROOT, "labels", "egomotion")
    if os.path.isdir(ego_dir):
        sensor_dirs["egomotion"] = ego_dir

    # Camera — each sensor has .timestamps.parquet and .blurred_boxes.parquet
    # plus .mp4 videos. We register timestamps and blur as separate tables.
    cam_dir = os.path.join(NFS_ROOT, "camera")
    if os.path.isdir(cam_dir):
        for sub in sorted(os.listdir(cam_dir)):
            full = os.path.join(cam_dir, sub)
            if os.path.isdir(full):
                safe = sub.replace("-", "_")
                sensor_dirs[f"cam_{safe}_ts"] = (full, ".timestamps.parquet")
                sensor_dirs[f"cam_{safe}_blur"] = (full, ".blurred_boxes.parquet")

    if args.max_sensors:
        # Prioritize egomotion and lidar (needed for gold views), then radar
        priority = ["egomotion", "lidar"]
        keys = [k for k in priority if k in sensor_dirs]
        remaining = [k for k in sorted(sensor_dirs.keys()) if k not in keys]
        keys.extend(remaining)
        keys = keys[:args.max_sensors]
        sensor_dirs = {k: sensor_dirs[k] for k in keys}

    print(f"\nDiscovered {len(sensor_dirs)} sensor entries")
    for name in sorted(sensor_dirs.keys()):
        entry = sensor_dirs[name]
        path = entry[0] if isinstance(entry, tuple) else entry
        print(f"  {name}: {path}")

    # ── Build Spark session ───────────────────────────────────────
    print("\n[SPARK] Building session...")
    spark = build_spark()
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CAT}.{NS_BRONZE}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CAT}.{NS_SILVER}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CAT}.{NS_GOLD}")

    all_bronze_tables = []
    radar_tables = []

    if not args.skip_register:
        # ── Phase 1: Register bare Parquets (no FUSE needed) ──────
        print(f"\n{'='*70}")
        print("[PHASE 1] Register bare Parquet tables from NFS")
        print(f"{'='*70}")

        bare_tables = {
            "clip_index": f"{NFS_ROOT}/clip_index.parquet",
            "data_collection": f"{NFS_ROOT}/metadata/data_collection.parquet",
            "sensor_presence": f"{NFS_ROOT}/metadata/sensor_presence.parquet",
        }
        cal_tables = {
            "camera_intrinsics": f"{NFS_ROOT}/calibration/camera_intrinsics",
            "sensor_extrinsics": f"{NFS_ROOT}/calibration/sensor_extrinsics",
            "vehicle_dimensions": f"{NFS_ROOT}/calibration/vehicle_dimensions",
        }

        for table, nfs_path in {**bare_tables, **cal_tables}.items():
            m = bench.begin("bronze", table)
            print(f"\n  [{table}]")
            if not os.path.exists(nfs_path):
                print(f"    [SKIP] {nfs_path} not found")
                bench.end(m)
                continue
            try:
                uri = f"file://{nfs_path}"
                rows = register_table(spark, table, uri)
                bench.end(m, rows=rows)
                all_bronze_tables.append(table)
                print(f"    [OK] {rows} rows in {m.elapsed:.1f}s")
            except Exception as e:
                bench.end(m)
                print(f"    [FAIL] {e}")

        # ── Phase 2: Mount & register sensor data ─────────────────
        print(f"\n{'='*70}")
        print("[PHASE 2] Mount zip archives & register sensor data")
        print(f"{'='*70}")

        # Group camera entries by source dir (same dir, different suffix)
        mounted_dirs = {}  # nfs_dir → mount_point

        for table in sorted(sensor_dirs.keys()):
            entry = sensor_dirs[table]
            # Unpack: tuple = (dir, suffix_filter), str = plain dir
            if isinstance(entry, tuple):
                nfs_dir, suffix_filter = entry
            else:
                nfs_dir, suffix_filter = entry, ".parquet"

            # Mount (reuse if same dir already mounted)
            mount_key = nfs_dir
            if mount_key not in mounted_dirs:
                mount_point = os.path.join(FUSE_ROOT, os.path.basename(nfs_dir))
                m_mount = bench.begin("mount", os.path.basename(nfs_dir))

                if not args.skip_fuse:
                    print(f"\n  [{table}] Mounting {nfs_dir}...")
                    ok, num_files = mount_sensor(nfs_dir, mount_point,
                                                 max_chunks=args.max_chunks)
                    bench.end(m_mount, files=num_files)
                    if not ok:
                        print(f"    [SKIP] Mount failed")
                        continue
                    print(f"    [MOUNT] {num_files} files in {m_mount.elapsed:.1f}s")
                else:
                    bench.end(m_mount)
                    if not os.path.ismount(mount_point):
                        print(f"    [SKIP] {mount_point} not mounted")
                        continue

                mounted_dirs[mount_key] = mount_point
            else:
                mount_point = mounted_dirs[mount_key]

            # Register in Iceberg
            m_reg = bench.begin("bronze", table)
            print(f"  [{table}] Registering via add_files()...")
            try:
                uri = f"file://{mount_point}"
                rows = register_table(spark, table, uri,
                                      suffix_filter=suffix_filter)
                bench.end(m_reg, rows=rows)
                all_bronze_tables.append(table)
                if table.startswith("radar_"):
                    radar_tables.append(table)
                print(f"    [OK] {rows} rows in {m_reg.elapsed:.1f}s")
            except Exception as e:
                bench.end(m_reg)
                print(f"    [FAIL] {e}")
    else:
        # Discover existing bronze tables
        try:
            existing = [
                r[1] for r in
                spark.sql(f"SHOW TABLES IN {CAT}.{NS_BRONZE}").collect()
            ]
            all_bronze_tables = existing
            radar_tables = [t for t in existing if t.startswith("radar_")]
        except Exception:
            pass

    # ── Phase 3: Silver views ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("[PHASE 3] Create Silver views (clip_id via input_file_name)")
    print(f"{'='*70}")
    m = bench.begin("silver", "all_views")
    create_silver_views(spark, all_bronze_tables)
    bench.end(m)

    # ── Phase 4: Gold views ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("[PHASE 4] Create Gold views (zero storage)")
    print(f"{'='*70}")
    m = bench.begin("gold", "all_views")
    create_gold_views(spark, radar_tables)
    bench.end(m)

    # ── Phase 5: Query benchmarks ─────────────────────────────────
    print(f"\n{'='*70}")
    print("[PHASE 5] Query latency benchmarks")
    print(f"{'='*70}")
    benchmark_queries(spark, bench, all_bronze_tables)

    # ── Summary ───────────────────────────────────────────────────
    print(bench.summary())

    # Save benchmark to JSON
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(bench.to_json(), f, indent=2)
    print(f"\nBenchmark saved to {BENCHMARK_FILE}")

    spark.stop()


if __name__ == "__main__":
    main()
