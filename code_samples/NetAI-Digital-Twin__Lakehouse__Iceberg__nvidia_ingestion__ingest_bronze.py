"""
Bronze Layer Ingestion for Nvidia PhysicalAI Autonomous Vehicles dataset.

Reads sensor data from NFS-mounted HuggingFace Hub zip archives (no
extraction needed — zips use STORED mode) and non-zipped Parquet files,
then writes 1:1 Iceberg tables in the nvidia_bronze namespace.

Tables produced:
  - clip_index          (from clip_index.parquet)
  - data_collection     (from metadata/data_collection.parquet)
  - sensor_presence     (from metadata/sensor_presence.parquet)
  - camera_intrinsics   (from calibration/camera_intrinsics/*.parquet)
  - sensor_extrinsics   (from calibration/sensor_extrinsics/*.parquet)
  - vehicle_dimensions  (from calibration/vehicle_dimensions/*.parquet)
  - egomotion           (from labels/egomotion/*.zip  → parquet inside)
  - lidar               (from lidar/lidar_top_360fov/*.zip → parquet)
  - radar_<name>        (from radar/<name>/*.zip → parquet)
  - camera_timestamps   (from camera/<name>/*.zip → .timestamps.parquet)
  - camera_blurred_boxes(from camera/<name>/*.zip → .blurred_boxes.parquet)
"""

import glob
import io
import os
import zipfile
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .benchmark import BenchmarkTracker
from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces

try:
    import DracoPy

    _HAS_DRACO = True
except ImportError:
    _HAS_DRACO = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_id_from_filename(name: str) -> str:
    """Extract the clip UUID from a zip-embedded filename like
    '25cd4769-5dcf-4b53-a351-bf2c5deb6124.egomotion.parquet'."""
    return name.split(".")[0]


def _read_parquets_from_zip(zip_path: str, suffix_filter: str = ".parquet",
                            add_clip_id: bool = True):
    """Yield (filename, pyarrow.Table) for every parquet inside a zip.

    When *add_clip_id* is True a ``clip_id`` column is derived from each
    filename's UUID prefix and appended to the table.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(suffix_filter):
                buf = io.BytesIO(z.read(name))
                tbl = pq.read_table(buf)
                if add_clip_id and "clip_id" not in tbl.schema.names:
                    clip_id = _clip_id_from_filename(name)
                    col = pa.array([clip_id] * tbl.num_rows, type=pa.string())
                    tbl = tbl.append_column("clip_id", col)
                yield name, tbl


def _concat_arrow_tables(tables: List[pa.Table]) -> pa.Table:
    """Concatenate a list of Arrow tables, handling schema differences."""
    if not tables:
        return pa.table({})
    schema = pa.unify_schemas([t.schema for t in tables])
    unified = [t.cast(schema) for t in tables]
    return pa.concat_tables(unified)


def _decode_draco_table(arrow_tbl: pa.Table) -> pa.Table:
    """Expand Draco-encoded point cloud blobs into x/y/z float arrays."""
    if not _HAS_DRACO:
        raise RuntimeError("DracoPy is required for lidar_mode='decoded'")
    timestamps, spins, nums, xs, ys, zs, clip_ids = [], [], [], [], [], [], []
    has_clip_id = "clip_id" in arrow_tbl.schema.names
    for i in range(arrow_tbl.num_rows):
        blob = arrow_tbl.column("draco_encoded_pointcloud")[i].as_py()
        ts = arrow_tbl.column("reference_timestamp")[i].as_py()
        si = arrow_tbl.column("spin_index")[i].as_py()
        if has_clip_id:
            clip_ids.append(arrow_tbl.column("clip_id")[i].as_py())
        try:
            import numpy as np
            mesh = DracoPy.decode(blob)
            pts = np.array(mesh.points).reshape(-1, 3)
            timestamps.append(ts)
            spins.append(si)
            nums.append(len(pts))
            xs.append(pts[:, 0].tolist())
            ys.append(pts[:, 1].tolist())
            zs.append(pts[:, 2].tolist())
        except Exception:
            timestamps.append(ts)
            spins.append(si)
            nums.append(0)
            xs.append([])
            ys.append([])
            zs.append([])
    result = pa.table({
        "reference_timestamp": pa.array(timestamps, type=pa.int64()),
        "spin_index": pa.array(spins, type=pa.int64()),
        "num_points": pa.array(nums, type=pa.int32()),
        "points_x": pa.array(xs, type=pa.list_(pa.float64())),
        "points_y": pa.array(ys, type=pa.list_(pa.float64())),
        "points_z": pa.array(zs, type=pa.list_(pa.float64())),
    })
    if has_clip_id:
        result = result.append_column("clip_id", pa.array(clip_ids, type=pa.string()))
    return result


# ---------------------------------------------------------------------------
# Bronze Ingester
# ---------------------------------------------------------------------------

class NvidiaBronzeIngester:
    """Reads the Nvidia PhysicalAI dataset into Iceberg Bronze tables."""

    def __init__(self, spark, config: NvidiaPipelineConfig,
                 tracker: Optional[BenchmarkTracker] = None):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.ns = config.nvidia.namespace_bronze
        self.src = config.nvidia.source_path
        self.tracker = tracker
        self._limit_chunks = config.nvidia.max_zip_chunks or None
        self._limit_clips = config.nvidia.max_clips_per_chunk or None

    def _full(self, table: str) -> str:
        return f"{self.cat}.{self.ns}.{table}"

    # -- zip-based ingestion helpers ----------------------------------------

    def _zip_paths(self, subdir: str) -> List[str]:
        """Return sorted list of zip file paths under <src>/<subdir>."""
        pattern = os.path.join(self.src, subdir, "*.zip")
        paths = sorted(glob.glob(pattern))
        if self._limit_chunks:
            paths = paths[: self._limit_chunks]
        return paths

    def _parquet_paths(self, subdir: str) -> List[str]:
        """Return sorted list of parquet file paths under <src>/<subdir>."""
        pattern = os.path.join(self.src, subdir, "*.parquet")
        return sorted(glob.glob(pattern))

    def _ingest_bare_parquets(self, table: str, subdir: str) -> int:
        """Ingest plain parquet files (no zip) into a Bronze table."""
        paths = self._parquet_paths(subdir)
        if not paths:
            print(f"  [SKIP] no parquet files in {subdir}")
            return 0
        if self._limit_chunks:
            paths = paths[: self._limit_chunks]

        if self.tracker:
            self.tracker.begin("bronze", table)

        bytes_in = sum(os.path.getsize(p) for p in paths)

        # Let Spark read the parquets directly (parallel across cores)
        abs_dir = os.path.join(self.src, subdir)
        df = self.spark.read.parquet(f"file://{abs_dir}")
        rows_in = df.count()

        full = self._full(table)
        df.writeTo(full).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()

        rows_out = self.spark.table(full).count()
        if self.tracker:
            self.tracker.end(rows_in=rows_in, rows_out=rows_out,
                             bytes_in=bytes_in, files=len(paths))
        print(f"  [DONE] {full}: {rows_out} rows from {len(paths)} files")
        return rows_out

    def _ingest_zipped_parquets(
        self,
        table: str,
        subdir: str,
        suffix_filter: str = ".parquet",
        transform_fn=None,
        spark_schema=None,
        compression_override: Optional[str] = None,
        batch_size: int = 20,
    ) -> int:
        """Read parquets from zip archives in batches, write to Iceberg.

        Processes *batch_size* zips at a time: extract matching parquets to a
        local temp directory (with optional transform), let Spark read them
        natively (parallel I/O across all cores), and append to the Iceberg
        table.  This avoids loading the entire dataset into a single Arrow
        table which doesn't scale beyond a few GB.
        """
        import shutil

        zips = self._zip_paths(subdir)
        if not zips:
            print(f"  [SKIP] no zips in {subdir}")
            return 0

        if self.tracker:
            self.tracker.begin("bronze", table)

        full = self._full(table)
        total_rows_in = 0
        total_rows_out = 0
        total_bytes_in = 0
        total_parquets = 0
        table_created = False
        clips_so_far = 0

        if compression_override:
            prev = self.spark.conf.get("spark.sql.parquet.compression.codec")
            self.spark.conf.set("spark.sql.parquet.compression.codec", compression_override)

        for batch_start in range(0, len(zips), batch_size):
            batch_zips = zips[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(zips) + batch_size - 1) // batch_size

            # Extract parquets from this batch of zips to a temp dir
            tmp_dir = f"/tmp/_bronze_staging_{table}"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)

            batch_tables: List[pa.Table] = []
            for zp in batch_zips:
                total_bytes_in += os.path.getsize(zp)
                for fname, tbl in _read_parquets_from_zip(zp, suffix_filter):
                    if self._limit_clips and clips_so_far >= self._limit_clips:
                        break
                    if transform_fn:
                        tbl = transform_fn(tbl)
                    batch_tables.append(tbl)
                    clips_so_far += 1
                if self._limit_clips and clips_so_far >= self._limit_clips:
                    break

            if not batch_tables:
                continue

            # Write batch to temp parquet files, then let Spark read them
            combined = _concat_arrow_tables(batch_tables)
            n_pq = len(batch_tables)
            total_parquets += n_pq
            total_rows_in += combined.num_rows
            del batch_tables

            pq.write_table(combined, os.path.join(tmp_dir, "part-0.parquet"))
            del combined
            import gc; gc.collect()

            df = self.spark.read.parquet(f"file://{tmp_dir}")

            if not table_created:
                df.writeTo(full).using("iceberg").tableProperty(
                    "format-version", "2"
                ).createOrReplace()
                table_created = True
            else:
                df.writeTo(full).append()

            print(f"    batch {batch_num}/{total_batches}: "
                  f"{n_pq} parquets from {len(batch_zips)} zips", flush=True)

            shutil.rmtree(tmp_dir, ignore_errors=True)

            if self._limit_clips and clips_so_far >= self._limit_clips:
                break

        if compression_override:
            self.spark.conf.set("spark.sql.parquet.compression.codec", prev)

        if table_created:
            total_rows_out = self.spark.table(full).count()
        if self.tracker:
            self.tracker.end(rows_in=total_rows_in, rows_out=total_rows_out,
                             bytes_in=total_bytes_in, zips=len(zips),
                             parquets=total_parquets)
        print(f"  [DONE] {full}: {total_rows_out} rows from {len(zips)} zips")
        return total_rows_out

    # -- per-table ingestors ------------------------------------------------

    def ingest_clip_index(self) -> int:
        path = os.path.join(self.src, "clip_index.parquet")
        if not os.path.exists(path):
            return 0
        if self.tracker:
            self.tracker.begin("bronze", "clip_index")
        tbl = pq.read_table(path)
        df = self.spark.createDataFrame(tbl.to_pandas())
        full = self._full("clip_index")
        df.writeTo(full).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()
        rows = self.spark.table(full).count()
        if self.tracker:
            self.tracker.end(rows_in=tbl.num_rows, rows_out=rows,
                             bytes_in=os.path.getsize(path))
        print(f"  [DONE] {full}: {rows} rows")
        return rows

    def ingest_metadata(self) -> Dict[str, int]:
        results = {}
        for name in ["data_collection", "sensor_presence"]:
            path = os.path.join(self.src, "metadata", f"{name}.parquet")
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: file not found")
                results[name] = 0
                continue
            if self.tracker:
                self.tracker.begin("bronze", name)
            tbl = pq.read_table(path)
            df = self.spark.createDataFrame(tbl.to_pandas())
            full = self._full(name)
            df.writeTo(full).using("iceberg").tableProperty(
                "format-version", "2"
            ).createOrReplace()
            rows = self.spark.table(full).count()
            if self.tracker:
                self.tracker.end(rows_in=tbl.num_rows, rows_out=rows,
                                 bytes_in=os.path.getsize(path))
            print(f"  [DONE] {full}: {rows} rows")
            results[name] = rows
        return results

    def ingest_calibration(self) -> Dict[str, int]:
        results = {}
        for cal_type in ["camera_intrinsics", "sensor_extrinsics",
                         "vehicle_dimensions"]:
            subdir = f"calibration/{cal_type}"
            if os.path.isdir(os.path.join(self.src, subdir)):
                results[cal_type] = self._ingest_bare_parquets(cal_type, subdir)
        return results

    def ingest_egomotion(self) -> int:
        return self._ingest_zipped_parquets(
            "egomotion", "labels/egomotion"
        )

    def ingest_lidar(self) -> int:
        from pyspark.sql.types import (StructType, StructField, LongType,
                                         IntegerType, DoubleType, ArrayType,
                                         StringType)
        mode = self.config.nvidia.lidar_mode
        transform = _decode_draco_table if mode == "decoded" else None
        table_name = "lidar_decoded" if mode == "decoded" else "lidar"
        decoded_schema = StructType([
            StructField("reference_timestamp", LongType()),
            StructField("spin_index", LongType()),
            StructField("num_points", IntegerType()),
            StructField("points_x", ArrayType(DoubleType())),
            StructField("points_y", ArrayType(DoubleType())),
            StructField("points_z", ArrayType(DoubleType())),
            StructField("clip_id", StringType()),
        ]) if mode == "decoded" else None
        return self._ingest_zipped_parquets(
            table_name, "lidar/lidar_top_360fov",
            transform_fn=transform, spark_schema=decoded_schema,
            # Draco blobs are already compressed; zstd on >2 GB blob
            # columns hits JVM max array size limit. Use snappy.
            compression_override="snappy",
            # Lidar zips are ~20 GB each; process 5 at a time to limit memory
            batch_size=5,
        )

    def ingest_radar(self) -> Dict[str, int]:
        import gc
        radar_dir = os.path.join(self.src, "radar")
        if not os.path.isdir(radar_dir):
            return {}
        results = {}
        for sensor in sorted(os.listdir(radar_dir)):
            sensor_path = os.path.join(radar_dir, sensor)
            if os.path.isdir(sensor_path):
                safe_name = sensor.replace("-", "_")
                results[safe_name] = self._ingest_zipped_parquets(
                    f"radar_{safe_name}", f"radar/{sensor}"
                )
                gc.collect()
        return results

    def ingest_camera_metadata(self) -> Dict[str, int]:
        """Ingest camera timestamp and blurred_boxes parquets from zips."""
        cam_dir = os.path.join(self.src, "camera")
        if not os.path.isdir(cam_dir):
            return {}
        results = {}
        for sensor in sorted(os.listdir(cam_dir)):
            sensor_path = os.path.join(cam_dir, sensor)
            if not os.path.isdir(sensor_path):
                continue
            safe = sensor.replace("-", "_")
            # timestamps
            results[f"{safe}_timestamps"] = self._ingest_zipped_parquets(
                f"cam_{safe}_ts",
                f"camera/{sensor}",
                suffix_filter=".timestamps.parquet",
            )
            # blurred boxes
            results[f"{safe}_blurred_boxes"] = self._ingest_zipped_parquets(
                f"cam_{safe}_blur",
                f"camera/{sensor}",
                suffix_filter=".blurred_boxes.parquet",
            )
        return results

    # -- orchestrator -------------------------------------------------------

    def ingest_all(self) -> Dict[str, int]:
        """Run full Bronze ingestion and return table → row-count map."""
        import gc
        results: Dict[str, int] = {}

        print("[BRONZE] clip_index")
        results["clip_index"] = self.ingest_clip_index()

        print("[BRONZE] metadata")
        results.update(self.ingest_metadata())

        print("[BRONZE] calibration")
        results.update(self.ingest_calibration())

        print("[BRONZE] egomotion")
        results["egomotion"] = self.ingest_egomotion()
        gc.collect()

        print(f"[BRONZE] lidar (mode={self.config.nvidia.lidar_mode})")
        lidar_key = ("lidar_decoded" if self.config.nvidia.lidar_mode == "decoded"
                      else "lidar")
        results[lidar_key] = self.ingest_lidar()
        gc.collect()

        print("[BRONZE] radar (all sensors)")
        results.update(self.ingest_radar())
        gc.collect()

        print("[BRONZE] camera metadata (timestamps + blurred boxes)")
        results.update(self.ingest_camera_metadata())

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bronze_ingestion(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker: Optional[BenchmarkTracker] = None,
) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-bronze")
    try:
        create_namespaces(spark, config)
        ingester = NvidiaBronzeIngester(spark, config, tracker=tracker)
        return ingester.ingest_all()
    finally:
        spark.stop()


if __name__ == "__main__":
    tracker = BenchmarkTracker("nvidia-bronze")
    results = run_bronze_ingestion(tracker=tracker)
    tracker.print_summary()
    tracker.flush()

    print("\n" + "=" * 60)
    print("NVIDIA BRONZE INGESTION SUMMARY")
    print("=" * 60)
    for table, count in results.items():
        status = "✓" if count >= 0 else "✗"
        print(f"  {status} {table}: {count} rows")
