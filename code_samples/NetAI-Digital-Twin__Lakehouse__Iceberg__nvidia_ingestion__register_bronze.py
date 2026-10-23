"""
True zero-copy Bronze registration via Iceberg's add_files() procedure.

Creates Bronze Iceberg tables by registering existing Parquet files
without rewriting any data.  Only Iceberg metadata (manifests, snapshots)
is created.  Works with:

  - FUSE-mounted zip archives (Option 1 — data stays on NFS)
  - S3/MinIO paths (Option 2 — after byte-stream upload)

Since clip_id is now derived at query time by Silver views (via
input_file_name()), Bronze tables no longer need it.  This means ALL
tables — including sensor data — can use add_files() for true zero-copy
registration.

Runs INSIDE the Spark container (via docker exec).

Usage:
    python -m nvidia_ingestion.register_bronze nfs
    python -m nvidia_ingestion.register_bronze fuse
    python -m nvidia_ingestion.register_bronze s3
    python -m nvidia_ingestion.register_bronze fuse --max-chunks 2
"""

import argparse
import glob
import os
from typing import Dict, List, Optional

from .benchmark import BenchmarkTracker
from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------

# Bare parquet files (not in zips) — on NFS directly
BARE_PARQUET_TABLES = {
    "clip_index": "clip_index.parquet",
    "data_collection": "metadata/data_collection.parquet",
    "sensor_presence": "metadata/sensor_presence.parquet",
}

CALIBRATION_TABLES = {
    "camera_intrinsics": "calibration/camera_intrinsics",
    "sensor_extrinsics": "calibration/sensor_extrinsics",
    "vehicle_dimensions": "calibration/vehicle_dimensions",
}


# ---------------------------------------------------------------------------
# Registration engine
# ---------------------------------------------------------------------------

class BronzeRegistrar:
    """Registers Parquet files into Bronze Iceberg tables via add_files().

    True zero-copy: no data is read, deserialized, or rewritten.  Only
    Iceberg manifests and snapshots are created, pointing at the existing
    Parquet files on FUSE or S3.
    """

    def __init__(self, spark, config: NvidiaPipelineConfig,
                 tracker: Optional[BenchmarkTracker] = None):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.ns = config.nvidia.namespace_bronze
        self.src = config.nvidia.source_path
        self.tracker = tracker
        self._limit_chunks = config.nvidia.max_zip_chunks or None

    def _full(self, table: str) -> str:
        return f"{self.cat}.{self.ns}.{table}"

    def _register(self, table: str, parquet_uri: str) -> int:
        """Create a table from schema and register files via add_files().

        parquet_uri: a file:// or s3a:// path to a Parquet file or directory.
        """
        full = self._full(table)
        # Create empty table with matching schema
        df = self.spark.read.parquet(parquet_uri)
        df.limit(0).writeTo(full).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()
        # Register the actual data files (zero-copy — metadata only)
        self.spark.sql(
            f"CALL {self.cat}.system.add_files("
            f"  table => '{full}',"
            f"  source_table => '`parquet`.`{parquet_uri}`'"
            f")"
        )
        return self.spark.table(full).count()

    def _append_files(self, table: str, parquet_uri: str) -> None:
        """Append additional files to an existing table via add_files().

        Does NOT count rows — counting the whole table after every append
        compounds quadratically (O(n²) FUSE file opens as dirs accumulate).
        Callers do a single count() at the very end.
        """
        full = self._full(table)
        self.spark.sql(
            f"CALL {self.cat}.system.add_files("
            f"  table => '{full}',"
            f"  source_table => '`parquet`.`{parquet_uri}`'"
            f")"
        )

    # -- Bare Parquet registration ---------------------------------------------

    def register_bare_parquets(self, mode: str = "fuse") -> Dict[str, int]:
        """Register bare Parquet files (not inside zips)."""
        results = {}
        for table, rel_path in BARE_PARQUET_TABLES.items():
            print(f"[REGISTER] {table}")
            if self.tracker:
                self.tracker.begin("bronze", table)

            if mode == "fuse":
                src_path = os.path.join(self.src, rel_path)
                if not os.path.exists(src_path):
                    print(f"  [SKIP] {src_path} not found")
                    results[table] = 0
                    if self.tracker:
                        self.tracker.end()
                    continue
                uri = f"file://{src_path}"
                bytes_in = os.path.getsize(src_path)
            else:
                uri = f"s3a://spark1/nvidia_bronze/{table}/"
                bytes_in = 0

            rows = self._register(table, uri)
            if self.tracker:
                self.tracker.end(rows_out=rows, bytes_in=bytes_in)
            print(f"  [DONE] {self._full(table)}: {rows} rows")
            results[table] = rows
        return results

    # -- Calibration registration ----------------------------------------------

    def register_calibration(self, mode: str = "fuse") -> Dict[str, int]:
        results = {}
        for table, rel_dir in CALIBRATION_TABLES.items():
            print(f"[REGISTER] {table}")
            if self.tracker:
                self.tracker.begin("bronze", table)

            if mode == "fuse":
                src_dir = os.path.join(self.src, rel_dir)
                if not os.path.isdir(src_dir):
                    print(f"  [SKIP] {src_dir} not found")
                    results[table] = 0
                    if self.tracker:
                        self.tracker.end()
                    continue
                uri = f"file://{src_dir}"
                parquets = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
                bytes_in = sum(os.path.getsize(p) for p in parquets)
            else:
                uri = f"s3a://spark1/nvidia_bronze/{table}/"
                bytes_in = 0

            rows = self._register(table, uri)
            if self.tracker:
                self.tracker.end(rows_out=rows, bytes_in=bytes_in)
            print(f"  [DONE] {self._full(table)}: {rows} rows")
            results[table] = rows
        return results

    # -- FUSE-based sensor registration ----------------------------------------

    def _find_fuse_dirs(self, sensor_subdir: str,
                        fuse_root: str) -> List[str]:
        """Find FUSE-mounted chunk directories for a sensor.

        With ratarmount --recursive, each zip archive in the source dir
        appears as a directory named ``<name>.zip/`` under the mount point,
        containing the zip's files.  E.g.:
            /tmp/nvidia-fuse/labels/egomotion/egomotion.chunk_0003.zip/
        """
        base = os.path.join(fuse_root, sensor_subdir)
        if not os.path.isdir(base):
            return []
        # If the directory itself contains Parquets (flat mount)
        if glob.glob(os.path.join(base, "*.parquet")):
            return [base]
        # ratarmount --recursive: subdirs named *.zip or chunk_* etc
        subdirs = sorted(
            d for d in glob.glob(os.path.join(base, "*"))
            if os.path.isdir(d)
        )
        if self._limit_chunks and subdirs:
            subdirs = subdirs[:self._limit_chunks]
        return subdirs

    def _create_batched_symlink_dirs(
        self,
        table: str,
        dirs: List[str],
        suffix_filter: str = ".parquet",
        batch_size: int = 5000,
    ) -> List[str]:
        """Create batched flat directories with symlinks to matching parquets.

        Instead of one huge directory (which stalls on large file counts),
        splits files into batches.  Returns a list of batch directory paths.
        """
        # Collect all source files
        all_files = []
        for d in dirs:
            for f in glob.glob(os.path.join(d, f"*{suffix_filter}")):
                chunk_name = os.path.basename(d).replace(".zip", "")
                basename = os.path.basename(f)
                link_name = f"{chunk_name}__{basename}"
                all_files.append((f, link_name))

        if not all_files:
            return []

        # Split into batches
        batch_dirs = []
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i + batch_size]
            batch_idx = i // batch_size
            batch_dir = f"/tmp/nvidia-fuse-flat/{table}/batch_{batch_idx:04d}"
            os.makedirs(batch_dir, exist_ok=True)

            # Clear old symlinks
            for old in os.listdir(batch_dir):
                os.remove(os.path.join(batch_dir, old))

            for src, link_name in batch:
                link_path = os.path.join(batch_dir, link_name)
                if not os.path.exists(link_path):
                    os.symlink(src, link_path)

            batch_dirs.append(batch_dir)

        print(f"    symlinks: {len(all_files)} files -> {len(batch_dirs)} batches of ~{batch_size}")
        return batch_dirs

    def register_fuse_sensor(
        self,
        table: str,
        sensor_subdir: str,
        suffix_filter: str = ".parquet",
        fuse_root: str = "/mnt/nvidia-fuse",
        batch_size: int = 5000,
    ) -> int:
        """Register FUSE-mounted sensor Parquets via add_files().

        Strategy depends on file sizes:
        - Small files (egomotion, radar): batched flat symlink dirs
        - Large files (lidar): per-chunk-dir add_files (100 files each)
        """
        print(f"[REGISTER] {table} (add_files)")
        if self.tracker:
            self.tracker.begin("bronze", table)

        dirs = self._find_fuse_dirs(sensor_subdir, fuse_root)
        if not dirs:
            print(f"  [SKIP] no FUSE data for {sensor_subdir}")
            if self.tracker:
                self.tracker.end()
            return 0

        # If only one dir, use it directly
        if len(dirs) == 1:
            all_files = glob.glob(os.path.join(dirs[0], f"*{suffix_filter}"))
            if not all_files:
                print(f"  [SKIP] no {suffix_filter} files found")
                if self.tracker:
                    self.tracker.end()
                return 0
            uri = f"file://{dirs[0]}"
            rows = self._register(table, uri)
            if self.tracker:
                self.tracker.end(rows_out=rows)
            print(f"  [DONE] {self._full(table)}: {rows} rows (1 dir)")
            return rows

        # Check if files are large by sampling one
        sample_files = glob.glob(os.path.join(dirs[0], f"*{suffix_filter}"))
        if not sample_files:
            print(f"  [SKIP] no {suffix_filter} files in first dir")
            if self.tracker:
                self.tracker.end()
            return 0

        sample_size = os.path.getsize(sample_files[0])
        files_per_dir = len(sample_files)

        # Large files (>10 MB): use per-dir add_files (each dir has ~100 files)
        if sample_size > 10_000_000:
            print(f"    large files (~{sample_size // 1048576} MB each, "
                  f"{files_per_dir}/dir), using per-dir registration")
            return self._register_per_dir(table, dirs)

        # Small files: use batched flat symlink approach
        batch_dirs = self._create_batched_symlink_dirs(
            table, dirs, suffix_filter, batch_size
        )
        if not batch_dirs:
            print(f"  [SKIP] no {suffix_filter} files found across {len(dirs)} dirs")
            if self.tracker:
                self.tracker.end()
            return 0

        # Register first batch (creates the table)
        uri = f"file://{batch_dirs[0]}"
        rows = self._register(table, uri)

        # Append remaining batches
        for i, bd in enumerate(batch_dirs[1:], 2):
            uri = f"file://{bd}"
            self._append_files(table, uri)
            if i % 5 == 0 or i == len(batch_dirs):
                print(f"    batch {i}/{len(batch_dirs)} registered", flush=True)

        # Single count at the end (not after every append)
        rows = self.spark.table(self._full(table)).count()
        if self.tracker:
            self.tracker.end(rows_out=rows)
        print(f"  [DONE] {self._full(table)}: {rows} rows "
              f"({len(dirs)} dirs, {len(batch_dirs)} batches)")
        return rows

    def _register_per_dir(self, table: str, dirs: List[str],
                          suffix_filter: str = ".parquet") -> int:
        """Register using one add_files() call per chunk directory.

        When suffix_filter is more specific than ".parquet" (e.g. cameras
        where a chunk dir holds both timestamps + blurred_boxes parquets),
        per-chunk symlink staging dirs are built inside the source tree so
        add_files sees only the matching files. Iceberg's add_files does
        not accept glob patterns. Symlinks must persist past registration
        because Iceberg manifests store their paths.
        """
        import shutil

        staging_dirs = dirs
        if suffix_filter != ".parquet":
            suffix_tag = suffix_filter.lstrip(".").replace(".", "_")
            staging_root = os.path.join(
                self.src, ".bronze_staging", f"{table}_{suffix_tag}"
            )
            if os.path.isdir(staging_root):
                shutil.rmtree(staging_root, ignore_errors=True)
            os.makedirs(staging_root, exist_ok=True)
            staging_dirs = []
            for idx, d in enumerate(dirs):
                sdir = os.path.join(staging_root, f"chunk_{idx:04d}")
                os.makedirs(sdir, exist_ok=True)
                for f in glob.glob(os.path.join(d, f"*{suffix_filter}")):
                    if os.path.getsize(f) == 0:
                        continue
                    link = os.path.join(sdir, os.path.basename(f))
                    try:
                        os.symlink(f, link)
                    except FileExistsError:
                        pass
                if os.listdir(sdir):
                    staging_dirs.append(sdir)
                else:
                    shutil.rmtree(sdir, ignore_errors=True)
            if not staging_dirs:
                print("  [SKIP] no non-empty files after staging")
                if self.tracker:
                    self.tracker.end()
                return 0

        # Register first dir (creates the table)
        rows = self._register(table, f"file://{staging_dirs[0]}")

        # Append remaining dirs
        for i, d in enumerate(staging_dirs[1:], 2):
            try:
                self._append_files(table, f"file://{d}")
            except Exception as e:
                print(f"    [WARN] dir {i}: {e}", flush=True)
            if i % 50 == 0 or i == len(staging_dirs):
                print(f"    dir {i}/{len(staging_dirs)} registered", flush=True)

        # Single count at the end
        rows = self.spark.table(self._full(table)).count()
        if self.tracker:
            self.tracker.end(rows_out=rows)
        print(f"  [DONE] {self._full(table)}: {rows} rows ({len(dirs)} dirs)")
        return rows

    # -- NFS-based sensor registration -----------------------------------------

    def _find_nfs_chunk_dirs(self, sensor_subdir: str) -> List[str]:
        """Find extracted chunk directories for a sensor on NFS.

        NFS layout (after zip extraction):
            <source_path>/labels/egomotion/egomotion.chunk_0000/
            <source_path>/lidar/lidar_top_360fov/lidar_top_360fov.chunk_0003/
            <source_path>/radar/radar_corner_front_left_srr_0/radar_...chunk_0003/

        Each chunk dir contains per-clip parquet files.
        """
        base = os.path.join(self.src, sensor_subdir)
        if not os.path.isdir(base):
            return []
        # If the directory itself contains parquets (flat layout)
        if glob.glob(os.path.join(base, "*.parquet")):
            return [base]
        # Otherwise look for chunk subdirectories
        subdirs = sorted(
            d for d in glob.glob(os.path.join(base, "*"))
            if os.path.isdir(d)
        )
        # Filter to dirs that contain non-empty parquets.
        # Some NFS chunks may have 0-byte stub files from incomplete
        # extraction — skip those to avoid add_files() errors.
        def _has_real_parquets(d):
            parquets = glob.glob(os.path.join(d, "*.parquet"))
            return any(os.path.getsize(p) > 0 for p in parquets[:3])
        subdirs = [d for d in subdirs if _has_real_parquets(d)]
        if self._limit_chunks and subdirs:
            subdirs = subdirs[:self._limit_chunks]
        return subdirs

    def register_nfs_sensor(
        self,
        table: str,
        sensor_subdir: str,
        suffix_filter: str = ".parquet",
    ) -> int:
        """Register sensor Parquets from extracted NFS chunk dirs via add_files().

        Always uses per-chunk-dir add_files() calls (no symlinks).
        NFS mounts inside containers don't reliably support symlink-based
        batching (file sizes report as 0 through symlinks on NFS), so we
        register each chunk directory individually.  Each chunk has ~100
        files, which add_files() handles efficiently.
        """
        print(f"[REGISTER] {table} (add_files, NFS)")
        if self.tracker:
            self.tracker.begin("bronze", table)

        dirs = self._find_nfs_chunk_dirs(sensor_subdir)
        if not dirs:
            print(f"  [SKIP] no NFS data for {sensor_subdir}")
            if self.tracker:
                self.tracker.end()
            return 0

        # Filter dirs to only those containing matching files
        if suffix_filter != ".parquet":
            dirs = [d for d in dirs
                    if glob.glob(os.path.join(d, f"*{suffix_filter}"))]
            if not dirs:
                print(f"  [SKIP] no {suffix_filter} files found")
                if self.tracker:
                    self.tracker.end()
                return 0

        # Per-dir registration: one add_files() per chunk directory.
        # (_register_per_dir handles both single-dir and multi-dir cases,
        # and builds a symlink staging dir when suffix_filter scopes the
        # files to a subset of parquets in each chunk.)
        print(f"    {len(dirs)} chunk dirs, using per-dir registration")
        return self._register_per_dir(table, dirs, suffix_filter=suffix_filter)

    def register_all_nfs(self) -> Dict[str, int]:
        """Register all Bronze tables from extracted NFS directories.

        Uses source_path (config.nvidia.source_path) as the root for both
        bare parquets and sensor chunk directories.
        """
        results = {}

        # Bare Parquets (directly on NFS — same as fuse mode)
        results.update(self.register_bare_parquets(mode="fuse"))

        # Calibration directories
        results.update(self.register_calibration(mode="fuse"))

        # Egomotion
        results["egomotion"] = self.register_nfs_sensor(
            "egomotion", "labels/egomotion",
        )

        # Lidar
        results["lidar"] = self.register_nfs_sensor(
            "lidar", "lidar/lidar_top_360fov",
        )

        # Radar (discover sensors dynamically)
        radar_root = os.path.join(self.src, "radar")
        if os.path.isdir(radar_root):
            for sensor in sorted(os.listdir(radar_root)):
                sensor_path = os.path.join(radar_root, sensor)
                if os.path.isdir(sensor_path):
                    safe = sensor.replace("-", "_")
                    results[f"radar_{safe}"] = self.register_nfs_sensor(
                        f"radar_{safe}", f"radar/{sensor}",
                    )

        # Camera metadata (timestamps + blur boxes)
        cam_root = os.path.join(self.src, "camera")
        if os.path.isdir(cam_root):
            for sensor in sorted(os.listdir(cam_root)):
                sensor_path = os.path.join(cam_root, sensor)
                if os.path.isdir(sensor_path):
                    safe = sensor.replace("-", "_")
                    results[f"cam_{safe}_ts"] = self.register_nfs_sensor(
                        f"cam_{safe}_ts", f"camera/{sensor}",
                        suffix_filter=".timestamps.parquet",
                    )
                    results[f"cam_{safe}_blur"] = self.register_nfs_sensor(
                        f"cam_{safe}_blur", f"camera/{sensor}",
                        suffix_filter=".blurred_boxes.parquet",
                    )

        return results

    # -- S3-based sensor registration ------------------------------------------

    def register_s3_sensor(self, table: str, s3_prefix: str) -> int:
        """Register sensor Parquets from S3 via add_files()."""
        print(f"[REGISTER] {table} (add_files, S3)")
        if self.tracker:
            self.tracker.begin("bronze", table)

        s3_uri = f"s3a://spark1/{s3_prefix}"
        rows = self._register(table, s3_uri)

        if self.tracker:
            self.tracker.end(rows_out=rows)
        print(f"  [DONE] {self._full(table)}: {rows} rows")
        return rows

    # -- Full registration orchestrators ----------------------------------------

    def register_all_fuse(self, fuse_root: str = "/mnt/nvidia-fuse") -> Dict[str, int]:
        """Register all Bronze tables from FUSE-mounted sources."""
        results = {}

        # Bare Parquets (directly on NFS)
        results.update(self.register_bare_parquets(mode="fuse"))

        # Calibration directories
        results.update(self.register_calibration(mode="fuse"))

        # Egomotion
        results["egomotion"] = self.register_fuse_sensor(
            "egomotion", "labels/egomotion", fuse_root=fuse_root,
        )

        # Lidar
        results["lidar"] = self.register_fuse_sensor(
            "lidar", "lidar/lidar_top_360fov", fuse_root=fuse_root,
        )

        # Radar (discover sensors dynamically)
        radar_root = os.path.join(fuse_root, "radar")
        if os.path.isdir(radar_root):
            for sensor in sorted(os.listdir(radar_root)):
                sensor_path = os.path.join(radar_root, sensor)
                if os.path.isdir(sensor_path):
                    safe = sensor.replace("-", "_")
                    results[f"radar_{safe}"] = self.register_fuse_sensor(
                        f"radar_{safe}", f"radar/{sensor}",
                        fuse_root=fuse_root,
                    )

        # Camera metadata (timestamps + blur boxes)
        cam_root = os.path.join(fuse_root, "camera")
        if os.path.isdir(cam_root):
            for sensor in sorted(os.listdir(cam_root)):
                sensor_path = os.path.join(cam_root, sensor)
                if os.path.isdir(sensor_path):
                    safe = sensor.replace("-", "_")
                    results[f"cam_{safe}_ts"] = self.register_fuse_sensor(
                        f"cam_{safe}_ts", f"camera/{sensor}",
                        suffix_filter=".timestamps.parquet",
                        fuse_root=fuse_root,
                    )
                    results[f"cam_{safe}_blur"] = self.register_fuse_sensor(
                        f"cam_{safe}_blur", f"camera/{sensor}",
                        suffix_filter=".blurred_boxes.parquet",
                        fuse_root=fuse_root,
                    )

        return results

    def register_all_s3(self, s3_prefix: str = "nvidia_bronze") -> Dict[str, int]:
        """Register all Bronze tables from S3 paths (after upload)."""
        results = {}

        results.update(self.register_bare_parquets(mode="s3"))
        results.update(self.register_calibration(mode="s3"))

        # Discover uploaded sensor tables from S3
        s3_base = f"s3a://spark1/{s3_prefix}"
        try:
            hadoop_conf = self.spark._jsc.hadoopConfiguration()
            fs_class = self.spark._jvm.org.apache.hadoop.fs.FileSystem
            uri_class = self.spark._jvm.java.net.URI
            fs = fs_class.get(uri_class(s3_base), hadoop_conf)
            path_class = self.spark._jvm.org.apache.hadoop.fs.Path
            statuses = fs.listStatus(path_class(s3_base))
            subdirs = [s.getPath().getName() for s in statuses if s.isDirectory()]
        except Exception:
            subdirs = []

        skip = set(BARE_PARQUET_TABLES) | set(CALIBRATION_TABLES)
        for table_dir in sorted(subdirs):
            if table_dir in skip:
                continue
            results[table_dir] = self.register_s3_sensor(
                table_dir, f"{s3_prefix}/{table_dir}/",
            )

        return results


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_nfs_registration(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker: Optional[BenchmarkTracker] = None,
) -> Dict[str, int]:
    """Register all Bronze tables from extracted NFS directories."""
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-register-nfs")
    try:
        create_namespaces(spark, config)
        reg = BronzeRegistrar(spark, config, tracker=tracker)
        return reg.register_all_nfs()
    finally:
        spark.stop()


def run_fuse_registration(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker: Optional[BenchmarkTracker] = None,
    fuse_root: str = "/mnt/nvidia-fuse",
) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-register-fuse")
    try:
        create_namespaces(spark, config)
        reg = BronzeRegistrar(spark, config, tracker=tracker)
        return reg.register_all_fuse(fuse_root=fuse_root)
    finally:
        spark.stop()


def run_s3_registration(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker: Optional[BenchmarkTracker] = None,
    s3_prefix: str = "nvidia_bronze",
) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-register-s3")
    try:
        create_namespaces(spark, config)
        reg = BronzeRegistrar(spark, config, tracker=tracker)
        return reg.register_all_s3(s3_prefix=s3_prefix)
    finally:
        spark.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-copy Bronze registration via Iceberg add_files()"
    )
    parser.add_argument(
        "mode", choices=["nfs", "fuse", "s3"],
        help="Registration source: nfs (extracted dirs), fuse (FUSE-mounted zips), or s3 (MinIO)"
    )
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--fuse-root", default="/mnt/nvidia-fuse")
    parser.add_argument("--s3-prefix", default="nvidia_bronze")
    args = parser.parse_args()

    cfg = NvidiaPipelineConfig()
    if args.max_chunks:
        cfg.nvidia.max_zip_chunks = args.max_chunks

    tracker = BenchmarkTracker(f"nvidia-register-{args.mode}")

    if args.mode == "nfs":
        results = run_nfs_registration(cfg, tracker)
    elif args.mode == "fuse":
        results = run_fuse_registration(cfg, tracker, fuse_root=args.fuse_root)
    else:
        results = run_s3_registration(cfg, tracker, s3_prefix=args.s3_prefix)

    tracker.print_summary()
    tracker.flush()

    print(f"\n{'=' * 60}")
    print(f"BRONZE REGISTRATION SUMMARY ({args.mode.upper()})")
    print(f"{'=' * 60}")
    for table, count in results.items():
        status = "+" if count > 0 else "-"
        print(f"  {status} {table}: {count} rows")


if __name__ == "__main__":
    main()
