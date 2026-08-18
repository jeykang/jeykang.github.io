"""Canonical-schema Bronze builder.

Reshapes the 41 zero-copy Bronze tables into 16 canonical tables matching
`kaist_schema_v2.dbml`. Run via the bottom main() or `run_canonical_build()`.

Source tables (existing Bronze, will be dropped at end):
    clip_index, data_collection, sensor_presence,
    camera_intrinsics, sensor_extrinsics, vehicle_dimensions,
    egomotion, lidar,
    radar_radar_<19 sensors>,
    cam_<7 sensors>_ts, cam_<7 sensors>_blur

Canonical tables (written to iceberg.nvidia_bronze.<TableName>):
    Session, Clip, Episode, Frame,
    Calibration, Camera, Lidar, Radar, CanBus,
    HDMap,
    Session_EgoMotion, Category, DynamicObject, Occupancy, Motion, EgoMotion

Strategy notes:
- clip_id is missing on sensor parquets — extracted from filename via
  regexp_extract(input_file_name(), UUID_RE, 1).
- frame_id is per-(clip_id, sensor_timestamp) hash — substr(sha2(...), 0, 16).
- session_id is one per chunk (340 sessions).
- Empty tables get the schema only; `.using('iceberg').createOrReplace()`
  with a 0-row DataFrame.
- Old tables are dropped only after all 16 canonical tables are built and
  row-counted, so a failure mid-build leaves both old and new tables in
  place rather than losing data.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, BooleanType, IntegerType, LongType, StringType, StructField, StructType,
)

from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UUID_RE = r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."

CAMERA_SENSORS = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

RADAR_SENSORS = [
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

CANONICAL_TABLES = [
    "Session", "Clip", "Episode", "Frame",
    "Calibration", "Camera", "Lidar", "Radar", "CanBus",
    "HDMap",
    "Session_EgoMotion", "Category", "DynamicObject", "Occupancy", "Motion", "EgoMotion",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_id(clip_id_col, ts_col):
    """sha256(clip_id + ":" + cast(ts as string)) truncated to 16 hex chars."""
    return F.substring(F.sha2(F.concat_ws(":", clip_id_col, ts_col.cast("string")), 256), 1, 16)


def _clip_id_from_path(path_col):
    """UUID extracted from a parquet filename column."""
    return F.regexp_extract(path_col, UUID_RE, 1)


def _session_id(chunk_col):
    """Stable session id per chunk: 'session_chunk_0042'."""
    return F.concat(F.lit("session_chunk_"), F.lpad(chunk_col.cast("string"), 4, "0"))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CanonicalBronzeBuilder:
    def __init__(self, spark: SparkSession, cfg: NvidiaPipelineConfig,
                 tracker=None, limit_clips: int = 0):
        self.spark = spark
        self.cfg = cfg
        self.tracker = tracker
        self.limit_clips = limit_clips
        self.cat = cfg.spark_catalog_name
        self.ns = cfg.nvidia.namespace_bronze

    def _full(self, name: str) -> str:
        return f"{self.cat}.{self.ns}.{name}"

    def _write(self, df: DataFrame, table: str) -> int:
        """createOrReplace via writeTo, then count."""
        full = self._full(table)
        df.writeTo(full).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()
        n = self.spark.table(full).count()
        return n

    def _begin(self, step: str, table: str = ""):
        if self.tracker:
            self.tracker.begin(step, table)
        return time.perf_counter()

    def _end(self, t0: float, table: str, rows: int):
        wall = time.perf_counter() - t0
        if self.tracker:
            self.tracker.end(rows_out=rows)
        print(f"  [DONE] {self._full(table)}: {rows:,} rows in {wall:.1f}s", flush=True)
        return wall

    # ----- per-table builders --------------------------------------------------

    def build_session(self) -> int:
        """340 rows: one per chunk, with clip_id_list as JSON-encoded array."""
        t0 = self._begin("canonical-session", "Session")
        clip_index = self.spark.table(self._full("clip_index"))
        df = (
            clip_index
            .groupBy("chunk")
            .agg(F.collect_list("clip_id").alias("clip_id_arr"))
            .select(
                _session_id(F.col("chunk")).alias("session_id"),
                F.concat(F.lit("chunk_"), F.lpad(F.col("chunk").cast("string"), 4, "0")).alias("session_name"),
                F.to_json(F.col("clip_id_arr")).alias("clip_id_list"),
            )
        )
        n = self._write(df, "Session")
        self._end(t0, "Session", n)
        return n

    def build_clip(self) -> int:
        """310,895 rows: clip_index ⨝ data_collection."""
        t0 = self._begin("canonical-clip", "Clip")
        clip_index = self.spark.table(self._full("clip_index"))
        data_coll = self.spark.table(self._full("data_collection"))
        joined = clip_index.alias("ci").join(
            data_coll.alias("dc"), on="clip_id", how="left"
        )
        # clip_idx via row_number within chunk; deterministic by clip_id
        from pyspark.sql.window import Window
        win = Window.partitionBy("chunk").orderBy("clip_id")
        df = (
            joined
            .withColumn("clip_idx", F.row_number().over(win))
            .select(
                F.col("clip_id"),
                _session_id(F.col("ci.chunk")).alias("session_id"),
                F.col("clip_idx"),
                F.lit(None).cast(StringType()).alias("frame_id_list"),
                # Best-effort date: month-only sentinel year
                F.when(F.col("dc.month").isNotNull(),
                       F.concat(F.lit("XXXX-"),
                                F.lpad(F.col("dc.month").cast("string"), 2, "0"),
                                F.lit("-01"))
                ).otherwise(F.lit(None).cast(StringType())).alias("date"),
                F.col("dc.country").alias("city"),
                F.lit(None).cast(StringType()).alias("site"),
            )
        )
        if self.limit_clips:
            df = df.limit(self.limit_clips)
        n = self._write(df, "Clip")
        self._end(t0, "Clip", n)
        return n

    def build_calibration(self) -> int:
        """Per (clip_id, sensor_name): extrinsics + camera_intrinsics merged."""
        t0 = self._begin("canonical-calibration", "Calibration")
        cam_int = self.spark.table(self._full("camera_intrinsics"))
        sen_ext = self.spark.table(self._full("sensor_extrinsics"))
        veh_dim = self.spark.table(self._full("vehicle_dimensions"))

        # camera_intrinsics → JSON of all intrinsic params
        cam_int_j = cam_int.select(
            F.col("clip_id"),
            F.col("camera_name").alias("sensor_name"),
            F.to_json(F.struct(
                "width", "height", "cx", "cy",
                "bw_poly_0", "bw_poly_1", "bw_poly_2", "bw_poly_3", "bw_poly_4",
                "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4",
            )).alias("camera_intrinsics"),
        )
        sen_ext_j = sen_ext.select(
            F.col("clip_id"),
            F.col("sensor_name"),
            F.to_json(F.struct("qx", "qy", "qz", "qw", "x", "y", "z")).alias("extrinsics"),
        )
        veh_dim_j = veh_dim.select(
            F.col("clip_id"),
            F.lit("vehicle").alias("sensor_name"),
            F.to_json(F.struct(
                "length", "width", "height",
                "rear_axle_to_bbox_center", "wheelbase", "track_width",
            )).alias("extrinsics"),
            F.lit(None).cast(StringType()).alias("camera_intrinsics"),
        )

        # Merge sensor_extrinsics + camera_intrinsics on (clip_id, sensor_name)
        merged = sen_ext_j.join(cam_int_j, on=["clip_id", "sensor_name"], how="full_outer")
        merged = merged.select(
            "clip_id", "sensor_name", "extrinsics", "camera_intrinsics"
        )

        df = merged.unionByName(veh_dim_j)
        n = self._write(df, "Calibration")
        self._end(t0, "Calibration", n)
        return n

    def build_camera(self) -> int:
        """Union 7 cam_<sensor>_ts tables, derive clip_id+camera_name+frame_id.

        filename is reconstructed from (clip_id, original chunk) — input_file_name()
        on the cam_*_ts tables points into a symlink-staging dir whose chunk
        numbers don't match the real data layout, so we can't use it directly.
        """
        t0 = self._begin("canonical-camera", "Camera")
        clip_index = self.spark.table(self._full("clip_index")).select("clip_id", "chunk")
        src_root = self.cfg.nvidia.source_path
        parts: List[DataFrame] = []
        for sensor in CAMERA_SENSORS:
            tbl = self._full(f"cam_{sensor}_ts")
            try:
                src = self.spark.table(tbl)
            except Exception as e:
                print(f"  [SKIP] {tbl}: {str(e)[:80]}", flush=True)
                continue
            clip_col = _clip_id_from_path(F.input_file_name())
            df = (
                src.select(
                    clip_col.alias("clip_id"),
                    _frame_id(clip_col, F.col("timestamp")).alias("frame_id"),
                    F.col("timestamp").alias("system_timestamp"),
                    F.col("timestamp").alias("sensor_timestamp"),
                    F.lit(sensor).alias("camera_name"),
                ).filter(F.col("clip_id") != "")
            )
            df = (
                df.join(clip_index, on="clip_id", how="left")
                .select(
                    "clip_id", "frame_id", "system_timestamp",
                    "sensor_timestamp", "camera_name",
                    F.concat(
                        F.lit(f"file:{src_root}/camera/{sensor}/{sensor}.chunk_"),
                        F.lpad(F.col("chunk").cast("string"), 4, "0"),
                        F.lit("/"),
                        F.col("clip_id"),
                        F.lit(f".{sensor}.mp4"),
                    ).alias("filename"),
                )
            )
            parts.append(df)
        df = parts[0]
        for p in parts[1:]:
            df = df.unionByName(p)
        n = self._write(df, "Camera")
        self._end(t0, "Camera", n)
        return n

    def build_lidar(self) -> int:
        """6.16M rows from lidar table; clip_id from filename."""
        t0 = self._begin("canonical-lidar", "Lidar")
        src = self.spark.table(self._full("lidar"))
        path = F.input_file_name()
        df = src.select(
            _clip_id_from_path(path).alias("clip_id"),
            _frame_id(_clip_id_from_path(path), F.col("spin_start_timestamp")).alias("frame_id"),
            F.col("spin_end_timestamp").alias("system_timestamp"),
            F.col("spin_start_timestamp").alias("sensor_timestamp"),
            path.alias("filename"),
        ).filter(F.col("clip_id") != "")
        n = self._write(df, "Lidar")
        self._end(t0, "Lidar", n)
        return n

    def build_radar(self) -> int:
        """Union 19 radar tables. Per-row metadata only — no payload columns.

        Note: source radar tables have a UINT_8 column (Open Issue #10) that
        breaks payload reads. This SELECT only references timestamp,
        sensor_timestamp, and input_file_name — all should be projection-pruned
        by Iceberg. If this fails, the canonical Radar table is left empty.
        """
        t0 = self._begin("canonical-radar", "Radar")
        parts: List[DataFrame] = []
        for sensor in RADAR_SENSORS:
            tbl = self._full(f"radar_{sensor}")
            try:
                src = self.spark.table(tbl)
            except Exception as e:
                print(f"  [SKIP] {tbl}: {str(e)[:80]}", flush=True)
                continue
            path = F.input_file_name()
            df = src.select(
                _clip_id_from_path(path).alias("clip_id"),
                _frame_id(_clip_id_from_path(path), F.col("sensor_timestamp")).alias("frame_id"),
                F.col("timestamp").alias("system_timestamp"),
                F.col("sensor_timestamp"),
                F.lit(sensor).alias("radar_name"),
                path.alias("filename"),
            ).filter(F.col("clip_id") != "")
            parts.append(df)
        if not parts:
            return self._build_empty_with_schema("Radar", _radar_schema())
        df = parts[0]
        for p in parts[1:]:
            df = df.unionByName(p)
        n = self._write(df, "Radar")
        self._end(t0, "Radar", n)
        return n

    def build_egomotion(self) -> int:
        """Per-row egomotion with clip_id+session_id+frame_id derived."""
        t0 = self._begin("canonical-egomotion", "EgoMotion")
        src = self.spark.table(self._full("egomotion"))
        clip_index = self.spark.table(self._full("clip_index")).select("clip_id", "chunk")
        path = F.input_file_name()
        annotated = (
            src.select(
                _clip_id_from_path(path).alias("clip_id"),
                F.col("timestamp"),
                F.to_json(F.struct("x", "y", "z")).alias("translation"),
                F.to_json(F.struct("qx", "qy", "qz", "qw")).alias("rotation"),
            )
            .filter(F.col("clip_id") != "")
        )
        df = (
            annotated.alias("e")
            .join(clip_index.alias("c"), on="clip_id", how="left")
            .select(
                _session_id(F.col("c.chunk")).alias("session_id"),
                F.col("e.clip_id").alias("clip_id"),
                _frame_id(F.col("e.clip_id"), F.col("e.timestamp")).alias("frame_id"),
                F.col("translation"),
                F.col("rotation"),
            )
        )
        n = self._write(df, "EgoMotion")
        self._end(t0, "EgoMotion", n)
        return n

    def build_frame(self) -> int:
        """Distinct (clip_id, sensor_timestamp) across egomotion + lidar + cameras + radar.

        Per-source distinct first to minimize shuffle, then UNION + DISTINCT.
        """
        t0 = self._begin("canonical-frame", "Frame")

        # Egomotion
        ego = self.spark.table(self._full("egomotion"))
        ego_path = F.input_file_name()
        ego_pairs = (ego.select(
            _clip_id_from_path(ego_path).alias("clip_id"),
            F.col("timestamp").alias("sensor_timestamp"),
        ).filter(F.col("clip_id") != "").distinct())

        # Lidar
        lid = self.spark.table(self._full("lidar"))
        lid_pairs = (lid.select(
            _clip_id_from_path(F.input_file_name()).alias("clip_id"),
            F.col("spin_start_timestamp").alias("sensor_timestamp"),
        ).filter(F.col("clip_id") != "").distinct())

        # Cameras
        cam_pairs_list: List[DataFrame] = []
        for sensor in CAMERA_SENSORS:
            tbl = self._full(f"cam_{sensor}_ts")
            try:
                src = self.spark.table(tbl)
            except Exception:
                continue
            cam_pairs_list.append(src.select(
                _clip_id_from_path(F.input_file_name()).alias("clip_id"),
                F.col("timestamp").alias("sensor_timestamp"),
            ).filter(F.col("clip_id") != "").distinct())

        # Radar
        rad_pairs_list: List[DataFrame] = []
        for sensor in RADAR_SENSORS:
            tbl = self._full(f"radar_{sensor}")
            try:
                src = self.spark.table(tbl)
            except Exception:
                continue
            rad_pairs_list.append(src.select(
                _clip_id_from_path(F.input_file_name()).alias("clip_id"),
                F.col("sensor_timestamp"),
            ).filter(F.col("clip_id") != "").distinct())

        all_pairs = [ego_pairs, lid_pairs] + cam_pairs_list + rad_pairs_list
        unioned = all_pairs[0]
        for p in all_pairs[1:]:
            unioned = unioned.unionByName(p)
        unioned = unioned.distinct()

        # Per-clip frame_idx via row_number ordered by sensor_timestamp
        from pyspark.sql.window import Window
        win = Window.partitionBy("clip_id").orderBy("sensor_timestamp")
        df = unioned.select(
            _frame_id(F.col("clip_id"), F.col("sensor_timestamp")).alias("frame_id"),
            F.col("clip_id"),
            F.lit(None).cast(StringType()).alias("episode_id"),
            F.row_number().over(win).alias("frame_idx"),
            F.to_json(F.array(F.col("sensor_timestamp"))).alias("sensor_timestamps"),
        )
        n = self._write(df, "Frame")
        self._end(t0, "Frame", n)
        return n

    # ----- empty tables --------------------------------------------------------

    def build_empties(self) -> Dict[str, int]:
        results = {}
        for name, schema in [
            ("Episode", _episode_schema()),
            ("CanBus", _canbus_schema()),
            ("HDMap", _hdmap_schema()),
            ("Session_EgoMotion", _session_egomotion_schema()),
            ("Category", _category_schema()),
            ("DynamicObject", _dynamic_object_schema()),
            ("Occupancy", _occupancy_schema()),
            ("Motion", _motion_schema()),
        ]:
            results[name] = self._build_empty_with_schema(name, schema)
        return results

    def _build_empty_with_schema(self, table: str, schema: StructType) -> int:
        t0 = self._begin(f"canonical-empty-{table.lower()}", table)
        empty_df = self.spark.createDataFrame([], schema)
        n = self._write(empty_df, table)
        self._end(t0, table, n)
        return n

    # ----- orchestration -------------------------------------------------------

    def build_all(self) -> Dict[str, int]:
        results: Dict[str, int] = {}
        results["Session"] = self.build_session()
        results["Clip"] = self.build_clip()
        results["Calibration"] = self.build_calibration()
        results["Camera"] = self.build_camera()
        results["Lidar"] = self.build_lidar()
        results["Radar"] = self.build_radar()
        results["EgoMotion"] = self.build_egomotion()
        results["Frame"] = self.build_frame()
        results.update(self.build_empties())
        return results

    def drop_old_tables(self) -> List[str]:
        """Drop the 41 old tables. Only run after canonical build is verified.

        IMPORTANT: Uses plain DROP TABLE (no PURGE). The old tables were
        registered via add_files() pointing at NFS source parquets; PURGE
        would delete those source files. We only want to remove the Iceberg
        catalog entries.
        """
        old = ["clip_index", "data_collection", "sensor_presence",
               "camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions",
               "egomotion", "lidar"]
        old += [f"radar_{s}" for s in RADAR_SENSORS]
        for s in CAMERA_SENSORS:
            old += [f"cam_{s}_ts", f"cam_{s}_blur"]
        dropped = []
        for t in old:
            full = self._full(t)
            try:
                self.spark.sql(f"DROP TABLE IF EXISTS {full}")  # NO PURGE
                dropped.append(full)
            except Exception as e:
                print(f"  [DROP-FAIL] {full}: {e}")
        return dropped


# ---------------------------------------------------------------------------
# Schemas for empty tables
# ---------------------------------------------------------------------------

def _episode_schema():
    return StructType([
        StructField("episode_id", StringType(), False),
        StructField("from_clip_id", StringType(), True),
        StructField("to_clip_id", StringType(), True),
        StructField("frame_id_list", StringType(), True),
    ])

def _canbus_schema():
    return StructType([
        StructField("clip_id", StringType(), False),
        StructField("frame_id", StringType(), False),
        StructField("system_timestamp", LongType(), True),
        StructField("sensor_timestamp", LongType(), True),
        StructField("filename", StringType(), True),
    ])

def _hdmap_schema():
    return StructType([
        StructField("filename", StringType(), False),
        StructField("city", StringType(), True),
        StructField("site", StringType(), True),
    ])

def _session_egomotion_schema():
    return StructType([
        StructField("session_id", StringType(), False),
        StructField("translation", StringType(), True),
        StructField("rotation", StringType(), True),
        StructField("start", StringType(), True),
        StructField("goal", StringType(), True),
    ])

def _category_schema():
    return StructType([StructField("category", StringType(), False)])

def _dynamic_object_schema():
    return StructType([
        StructField("clip_id", StringType(), False),
        StructField("frame_id", StringType(), False),
        StructField("boxes_3d", StringType(), True),
        StructField("category", StringType(), True),
    ])

def _occupancy_schema():
    return StructType([
        StructField("clip_id", StringType(), False),
        StructField("frame_id", StringType(), False),
        StructField("category", StringType(), True),
    ])

def _motion_schema():
    return StructType([
        StructField("clip_id", StringType(), False),
        StructField("frame_id", StringType(), False),
    ])

def _radar_schema():
    return StructType([
        StructField("clip_id", StringType(), False),
        StructField("frame_id", StringType(), False),
        StructField("system_timestamp", LongType(), True),
        StructField("sensor_timestamp", LongType(), True),
        StructField("radar_name", StringType(), False),
        StructField("filename", StringType(), True),
    ])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_canonical_build(
    config: Optional[NvidiaPipelineConfig] = None,
    tracker=None,
    drop_old: bool = False,
    limit_clips: int = 0,
) -> Dict[str, int]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-canonical-bronze")
    try:
        create_namespaces(spark, config)
        b = CanonicalBronzeBuilder(spark, config, tracker=tracker, limit_clips=limit_clips)
        results = b.build_all()
        if drop_old:
            print("\n[FINALIZE] dropping old Bronze tables ...", flush=True)
            dropped = b.drop_old_tables()
            print(f"  dropped {len(dropped)} old tables", flush=True)
        return results
    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--drop-old", action="store_true",
                   help="After build, drop the 41 old Bronze tables")
    p.add_argument("--limit-clips", type=int, default=0,
                   help="Limit Clip table to N rows (testing)")
    args = p.parse_args()
    res = run_canonical_build(drop_old=args.drop_old, limit_clips=args.limit_clips)
    print("\n=== Canonical Bronze build summary ===")
    for k, v in res.items():
        print(f"  {k}: {v:,} rows")
