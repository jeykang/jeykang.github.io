"""
Silver Layer Transformations for Nvidia PhysicalAI dataset.

Two modes are supported, controlled by ``NvidiaConfig.silver_mode``:

**inplace** (default)
    Reads the chunked zip archives on the decimated subset, enriches every
    Parquet inside with ``clip_id`` (derived from the filename UUID) and
    ``sensor_name`` (for radar tables), writes the enriched Parquets back to
    disk as a directory of plain ``.parquet`` files, and removes the original
    zip.  After this step the files on disk *are* the Silver data — no
    Iceberg views or tables are needed, and downstream Bronze registration
    via ``add_files()`` will pick up the enriched columns automatically.

**view** (legacy)
    Creates zero-copy SQL views over Bronze tables.  No data is duplicated;
    each Silver view is a thin ``CREATE OR REPLACE VIEW`` that adds clip_id
    at query time via ``input_file_name()`` and a UUID regex.
"""

import io
import logging
import os
import zipfile
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

from .benchmark import BenchmarkTracker
from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces

log = logging.getLogger("nvidia.silver")

# UUID regex for extracting clip_id from filenames (view mode)
_CLIP_ID_EXPR = (
    "regexp_extract(input_file_name(), "
    "'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', 1)"
)


# ---------------------------------------------------------------------------
# In-place Silver transformer  (default)
# ---------------------------------------------------------------------------

def _clip_id_from_filename(name: str) -> str:
    """Extract clip UUID from e.g. '25cd4769-…-bf2c5deb6124.egomotion.parquet'."""
    return name.split(".")[0]


class InplaceSilverTransformer:
    """Enrich Parquet files on disk — adds ``clip_id`` and ``sensor_name``
    directly into the data files, replacing the original zip archives with
    directories of enriched Parquets.

    After this runs, every sensor Parquet on the subset has ``clip_id`` baked
    in, so downstream Iceberg registration (Bronze ``add_files()``) and Gold
    joins work without runtime derivation.
    """

    # Sensor directories inside the subset that contain zipped parquets.
    _SENSOR_DIRS = {
        "labels/egomotion": {"add_clip_id": True},
        "lidar/lidar_top_360fov": {"add_clip_id": True},
    }
    # Radar sensors — also get sensor_name.
    _RADAR_SENSORS = [
        "radar_corner_front_left_srr_0",
        "radar_corner_front_left_srr_3",
        "radar_corner_front_right_srr_0",
        "radar_corner_front_right_srr_3",
        "radar_corner_rear_left_srr_0",
        "radar_corner_rear_left_srr_3",
        "radar_corner_rear_right_srr_0",
        "radar_corner_rear_right_srr_3",
        "radar_front_center_imaging_lrr_1",
        "radar_front_center_mrr_2",
        "radar_front_center_srr_0",
        "radar_rear_left_mrr_2",
        "radar_rear_left_srr_0",
        "radar_rear_right_mrr_2",
        "radar_rear_right_srr_0",
        "radar_side_left_srr_0",
        "radar_side_left_srr_3",
        "radar_side_right_srr_0",
        "radar_side_right_srr_3",
    ]
    # Camera sensors — extract timestamps + blurred_boxes parquets.
    _CAMERA_SENSORS = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ]

    def __init__(self, source_path: str,
                 tracker: Optional[BenchmarkTracker] = None):
        self.src = source_path
        self.tracker = tracker

    # -- helpers ------------------------------------------------------------

    def _enrich_zip(self, zip_path: str, out_dir: str,
                    add_clip_id: bool = True,
                    sensor_name: Optional[str] = None,
                    suffix_filter: str = ".parquet") -> int:
        """Read parquets from *zip_path*, enrich, write to *out_dir*.

        Returns the number of parquet files written.
        """
        os.makedirs(out_dir, exist_ok=True)
        written = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(suffix_filter):
                    continue
                buf = io.BytesIO(zf.read(name))
                tbl = pq.read_table(buf)

                if add_clip_id and "clip_id" not in tbl.schema.names:
                    clip_id = _clip_id_from_filename(name)
                    col = pa.array([clip_id] * tbl.num_rows, type=pa.string())
                    tbl = tbl.append_column("clip_id", col)

                if sensor_name and "sensor_name" not in tbl.schema.names:
                    col = pa.array([sensor_name] * tbl.num_rows, type=pa.string())
                    tbl = tbl.append_column("sensor_name", col)

                out_path = os.path.join(out_dir, name)
                pq.write_table(tbl, out_path)
                written += 1
        return written

    def _process_sensor_dir(self, rel_dir: str,
                            add_clip_id: bool = True,
                            sensor_name: Optional[str] = None,
                            suffix_filter: str = ".parquet") -> Dict[str, int]:
        """Process all zips in a sensor directory, enriching in-place."""
        abs_dir = os.path.join(self.src, rel_dir)
        if not os.path.isdir(abs_dir):
            log.warning("Sensor dir missing: %s", abs_dir)
            return {"files": 0, "zips": 0, "skipped": 0}

        zips = sorted(
            f for f in os.listdir(abs_dir) if f.endswith(".zip")
        )
        if not zips:
            log.info("  No zips in %s (already enriched?)", rel_dir)
            return {"files": 0, "zips": 0, "skipped": 0}

        total_files = 0
        skipped = 0
        for zname in zips:
            zip_path = os.path.join(abs_dir, zname)
            # Output directory replaces the .zip extension
            out_dir = os.path.join(abs_dir, zname.removesuffix(".zip"))

            if os.path.isdir(out_dir):
                skipped += 1
                continue

            n = self._enrich_zip(
                zip_path, out_dir,
                add_clip_id=add_clip_id,
                sensor_name=sensor_name,
                suffix_filter=suffix_filter,
            )
            total_files += n

            # Remove the zip now that enriched parquets are on disk.
            os.remove(zip_path)

        return {"files": total_files, "zips": len(zips), "skipped": skipped}

    # -- orchestrator -------------------------------------------------------

    def transform_all(self) -> Dict[str, int]:
        """Run in-place enrichment across all sensor types.

        Returns a dict of ``sensor_dir → files_written``.
        """
        results: Dict[str, int] = {}

        # Egomotion + LiDAR
        for rel_dir, opts in self._SENSOR_DIRS.items():
            label = rel_dir.split("/")[-1]
            print(f"[SILVER inplace] {label}")
            if self.tracker:
                self.tracker.begin("silver", label)
            stats = self._process_sensor_dir(
                rel_dir, add_clip_id=opts["add_clip_id"],
            )
            results[label] = stats["files"]
            if self.tracker:
                self.tracker.end(rows_in=stats["files"], rows_out=stats["files"])
            print(f"  enriched {stats['files']} files from {stats['zips']} zips"
                  f" ({stats['skipped']} already done)")

        # Radar
        for sensor in self._RADAR_SENSORS:
            rel_dir = f"radar/{sensor}"
            sensor_name = sensor.replace("radar_", "", 1)
            print(f"[SILVER inplace] {sensor}")
            if self.tracker:
                self.tracker.begin("silver", sensor)
            stats = self._process_sensor_dir(
                rel_dir, add_clip_id=True, sensor_name=sensor_name,
            )
            results[sensor] = stats["files"]
            if self.tracker:
                self.tracker.end(rows_in=stats["files"], rows_out=stats["files"])
            print(f"  enriched {stats['files']} files from {stats['zips']} zips"
                  f" ({stats['skipped']} already done)")

        # Camera metadata (timestamps + blurred boxes)
        for sensor in self._CAMERA_SENSORS:
            for suffix, label_sfx in [
                (".timestamps.parquet", "_ts"),
                (".blurred_boxes.parquet", "_blur"),
            ]:
                rel_dir = f"camera/{sensor}"
                label = f"cam_{sensor}{label_sfx}"
                print(f"[SILVER inplace] {label}")
                if self.tracker:
                    self.tracker.begin("silver", label)
                stats = self._process_sensor_dir(
                    rel_dir, add_clip_id=True, suffix_filter=suffix,
                )
                results[label] = stats["files"]
                if self.tracker:
                    self.tracker.end(rows_in=stats["files"], rows_out=stats["files"])
                print(f"  enriched {stats['files']} files from {stats['zips']} zips"
                      f" ({stats['skipped']} already done)")

        # Calibration parquets are bare (no zips, no clip_id needed) — skip.
        # Metadata parquets (clip_index, data_collection, sensor_presence)
        # already have clip_id — skip.

        return results


# ---------------------------------------------------------------------------
# View-based Silver transformer  (legacy)
# ---------------------------------------------------------------------------

class NvidiaSilverTransformer:
    """
    Creates zero-copy Silver **views** over Nvidia Bronze tables.

    No data is duplicated.  Each Silver view is a SQL alias that provides
    consistent naming, sensor_name tagging (radar), clip_id derivation
    (for zero-copy Bronze tables), and a uniform query surface for the
    Gold builder.  Only Iceberg catalog metadata is stored.
    """

    # Tables whose source Parquets already contain clip_id
    _HAS_CLIP_ID = {"clip_index", "data_collection", "sensor_presence"}

    # Calibration tables — no clip_id needed (per-sensor, not per-clip)
    _NO_CLIP_ID = {"camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"}

    def __init__(self, spark: SparkSession, config: NvidiaPipelineConfig,
                 tracker: Optional[BenchmarkTracker] = None):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.bronze_ns = config.nvidia.namespace_bronze
        self.silver_ns = config.nvidia.namespace_silver
        self.tracker = tracker

    def _bronze(self, t: str) -> str:
        return f"{self.cat}.{self.bronze_ns}.{t}"

    def _silver(self, t: str) -> str:
        return f"{self.cat}.{self.silver_ns}.{t}"

    def _create_view(self, silver_name: str, sql: str) -> int:
        """Execute a CREATE OR REPLACE VIEW and return the row count."""
        self.spark.sql(sql)
        return self.spark.table(self._silver(silver_name)).count()

    def _needs_clip_id(self, table: str) -> bool:
        if table in self._HAS_CLIP_ID or table in self._NO_CLIP_ID:
            return False
        try:
            cols = [f.name for f in self.spark.table(self._bronze(table)).schema]
            return "clip_id" not in cols
        except Exception:
            return True

    # -- per-table transforms ------------------------------------------------

    def _transform_passthrough(self, table: str) -> int:
        if self.tracker:
            self.tracker.begin("silver", table)
        sql = (
            f"CREATE OR REPLACE VIEW {self._silver(table)} "
            f"AS SELECT * FROM {self._bronze(table)}"
        )
        rows_out = self._create_view(table, sql)
        if self.tracker:
            self.tracker.end(rows_in=rows_out, rows_out=rows_out)
        print(f"  [VIEW] {self._silver(table)}: {rows_out} rows")
        return rows_out

    def _transform_with_clip_id(self, table: str) -> int:
        if self.tracker:
            self.tracker.begin("silver", table)
        sql = (
            f"CREATE OR REPLACE VIEW {self._silver(table)} "
            f"AS SELECT *, {_CLIP_ID_EXPR} AS clip_id "
            f"FROM {self._bronze(table)}"
        )
        rows_out = self._create_view(table, sql)
        if self.tracker:
            self.tracker.end(rows_in=rows_out, rows_out=rows_out)
        print(f"  [VIEW] {self._silver(table)}: {rows_out} rows (+ clip_id)")
        return rows_out

    def _transform_auto(self, table: str) -> int:
        if self._needs_clip_id(table):
            return self._transform_with_clip_id(table)
        return self._transform_passthrough(table)

    def _transform_radar(self, table: str) -> int:
        if self.tracker:
            self.tracker.begin("silver", table)
        sensor_name = table.replace("radar_", "", 1)

        if self._needs_clip_id(table):
            sql = (
                f"CREATE OR REPLACE VIEW {self._silver(table)} "
                f"AS SELECT *, '{sensor_name}' AS sensor_name, "
                f"{_CLIP_ID_EXPR} AS clip_id "
                f"FROM {self._bronze(table)}"
            )
        else:
            sql = (
                f"CREATE OR REPLACE VIEW {self._silver(table)} "
                f"AS SELECT *, '{sensor_name}' AS sensor_name "
                f"FROM {self._bronze(table)}"
            )

        rows_out = self._create_view(table, sql)
        if self.tracker:
            self.tracker.end(rows_in=rows_out, rows_out=rows_out)
        print(f"  [VIEW] {self._silver(table)}: {rows_out} rows")
        return rows_out

    def _transform_camera_meta(self, table: str) -> int:
        return self._transform_auto(table)

    # -- orchestrator --------------------------------------------------------

    def transform_all(self) -> Dict[str, int]:
        results: Dict[str, int] = {}

        for tbl in ["clip_index", "data_collection", "sensor_presence",
                     "camera_intrinsics", "sensor_extrinsics",
                     "vehicle_dimensions", "egomotion"]:
            try:
                print(f"[SILVER] {tbl}")
                results[tbl] = self._transform_auto(tbl)
            except Exception as e:
                print(f"  [ERROR] {tbl}: {e}")
                results[tbl] = -1

        lidar_tbl = ("lidar_decoded"
                      if self.config.nvidia.lidar_mode == "decoded"
                      else "lidar")
        try:
            print(f"[SILVER] {lidar_tbl}")
            results[lidar_tbl] = self._transform_auto(lidar_tbl)
        except Exception as e:
            print(f"  [ERROR] {lidar_tbl}: {e}")
            results[lidar_tbl] = -1

        try:
            bronze_tables = [
                r[1] for r in
                self.spark.sql(
                    f"SHOW TABLES IN {self.cat}.{self.bronze_ns}"
                ).collect()
            ]
        except Exception:
            bronze_tables = []

        for t in sorted(bronze_tables):
            if t.startswith("radar_"):
                try:
                    print(f"[SILVER] {t}")
                    results[t] = self._transform_radar(t)
                except Exception as e:
                    print(f"  [ERROR] {t}: {e}")
                    results[t] = -1

        for t in sorted(bronze_tables):
            if t.startswith("cam_"):
                try:
                    print(f"[SILVER] {t}")
                    results[t] = self._transform_camera_meta(t)
                except Exception as e:
                    print(f"  [ERROR] {t}: {e}")
                    results[t] = -1

        return results


# ---------------------------------------------------------------------------
# Entry point (dispatches based on silver_mode)
# ---------------------------------------------------------------------------

def run_silver_transformation(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker: Optional[BenchmarkTracker] = None,
) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()

    mode = config.nvidia.silver_mode

    if mode == "inplace":
        transformer = InplaceSilverTransformer(
            source_path=config.nvidia.source_path,
            tracker=tracker,
        )
        return transformer.transform_all()

    # Legacy view mode — needs Spark
    spark = build_spark_session(config, app_name="nvidia-silver")
    try:
        create_namespaces(spark, config)
        transformer = NvidiaSilverTransformer(spark, config, tracker=tracker)
        return transformer.transform_all()
    finally:
        spark.stop()


if __name__ == "__main__":
    tracker = BenchmarkTracker("nvidia-silver")
    results = run_silver_transformation(tracker=tracker)
    tracker.print_summary()
    tracker.flush()

    print("\n" + "=" * 60)
    print("NVIDIA SILVER TRANSFORMATION SUMMARY")
    print("=" * 60)
    for table, count in results.items():
        status = "+" if count >= 0 else "-"
        print(f"  {status} {table}: {count} files")
