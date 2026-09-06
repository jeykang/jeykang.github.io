#!/usr/bin/env python3
"""
AD-Specific Workload Benchmarks for the Autonomous Driving Data Lakehouse.

Demonstrates that the lakehouse is designed specifically for AD data access
patterns through five benchmark experiments:

  1. Three-Workload Gold Table Comparison
     - Object Detection (camera_annotations)
     - SLAM / Localization (lidar_with_ego)
     - Multi-Modal Fusion (sensor_fusion_frame)

  2. Partition Pruning Measurement
     - Sensor-specific pruning (camera_name filter)
     - Combined pruning (camera_name + clip_id)
     - Partitioned vs unpartitioned comparison

  3. Temporal Sequence Replay
     - Sorted Silver vs unsorted Bronze for sequential frame access

  4. Iceberg Time Travel for Training Reproducibility
     - Snapshot-based dataset pinning

  5. Column-Level Metrics Effectiveness
     - Timestamp range predicate pushdown

Usage (inside spark-iceberg container):
    python benchmarks/ad_workload_benchmark.py
    # or via the pipeline runner:
    python -m kaist_ingestion.kaist_runner benchmark
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pyspark.sql import SparkSession

from kaist_ingestion.config import PipelineConfig, build_spark_session


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    experiment: str
    variant: str
    description: str
    elapsed_seconds: float
    row_count: int
    extra: Dict[str, str] = field(default_factory=dict)


class BenchmarkSuite:
    """
    Runs and collects AD workload benchmark results.

    Each experiment targets a specific claim about the AD lakehouse's
    domain-specific optimizations.
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.results: List[BenchmarkResult] = []

        # Namespace shortcuts
        self.bronze = f"{self.cat}.{config.kaist.namespace_bronze}"
        self.silver = f"{self.cat}.{config.kaist.namespace_silver}"
        self.gold = f"{self.cat}.{config.kaist.namespace_gold}"

    def _is_spark_alive(self) -> bool:
        """Check if SparkContext is still active."""
        try:
            sc = self.spark.sparkContext
            return not sc._jsc.sc().isStopped()
        except Exception:
            return False

    def _warmup_jvm(self):
        """Run a trivial query to warm up the JVM before timing begins."""
        print("\n  Warming up JVM...")
        for _ in range(3):
            self.spark.sql(
                f"SELECT count(*) FROM {self.gold}.camera_annotations"
            ).collect()
        print("  JVM warm.")

    def _time_query(
        self, query_fn, warmup: int = 2, runs: int = 5, use_count: bool = False,
    ) -> Tuple[float, int]:
        """
        Time a query function with warmup runs.

        Args:
            query_fn: Callable returning a DataFrame
            warmup: Number of warmup (untimed) runs
            runs: Number of timed runs (takes median)
            use_count: If True, use df.count() instead of df.collect().
                       This measures scan+join time without Python
                       serialization overhead (fairer for large results
                       with complex nested types).

        Returns (median_seconds, row_count).
        """
        # Warmup: execute query but discard timing
        for _ in range(warmup):
            df = query_fn()
            if use_count:
                df.count()
            else:
                df.collect()

        # Timed runs
        times = []
        row_count = 0
        for _ in range(runs):
            start = time.time()
            df = query_fn()
            if use_count:
                row_count = df.count()
            else:
                results = df.collect()
                row_count = len(results)
            elapsed = time.time() - start
            times.append(elapsed)

        times.sort()
        median = times[len(times) // 2]
        return median, row_count

    def _record(
        self,
        experiment: str,
        variant: str,
        description: str,
        elapsed: float,
        row_count: int,
        **extra,
    ):
        result = BenchmarkResult(
            experiment=experiment,
            variant=variant,
            description=description,
            elapsed_seconds=elapsed,
            row_count=row_count,
            extra={k: str(v) for k, v in extra.items()},
        )
        self.results.append(result)
        print(f"    {variant}: {elapsed:.4f}s ({row_count} rows)")

    # =========================================================================
    # Experiment 1: Three-Workload Gold Table Comparison
    # =========================================================================

    def run_exp1_three_workloads(self):
        """
        Compare Gold table single-read vs Silver multi-table join
        for each of the three target ML workloads.

        This validates the paper's claim that workload-specific Gold tables
        eliminate runtime joins for the dominant AD access patterns.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Three-Workload Gold vs Silver Comparison")
        print("=" * 70)

        # Get a real clip_id and camera_name from the data
        sample_row = self.spark.sql(
            f"SELECT clip_id, camera_name "
            f"FROM {self.gold}.camera_annotations LIMIT 1"
        ).collect()[0]
        clip_id = sample_row["clip_id"]
        camera_name = sample_row["camera_name"]

        # --- 1a: Object Detection (camera_annotations) ---
        print(f"\n  [1a] Object Detection (camera_name='{camera_name}')")

        # Gold: single partitioned read — zero joins
        def gold_objdet():
            return self.spark.sql(
                f"SELECT * FROM {self.gold}.camera_annotations "
                f"WHERE camera_name = '{camera_name}'"
            )

        # Silver: runtime 6-table join including annotations
        # (mirrors build_gold.py join logic for fair comparison)
        def silver_objdet():
            return self.spark.sql(f"""
                SELECT
                    cam.frame_id, cam.clip_id, cl.session_id, f.frame_idx,
                    cam.sensor_timestamp, cam.camera_name, cam.filename,
                    cal.extrinsics, cal.camera_intrinsics,
                    annot.annotations,
                    hm.city, hm.site, cl.date
                FROM {self.silver}.camera cam
                LEFT JOIN {self.silver}.frame f ON cam.frame_id = f.frame_id
                LEFT JOIN {self.silver}.clip cl ON cam.clip_id = cl.clip_id
                LEFT JOIN {self.silver}.calibration cal
                    ON cam.clip_id = cal.clip_id
                    AND cam.camera_name = cal.sensor_name
                LEFT JOIN (
                    SELECT frame_id,
                           collect_list(struct(boxes_3d, category)) AS annotations
                    FROM {self.silver}.dynamic_object
                    GROUP BY frame_id
                ) annot ON cam.frame_id = annot.frame_id
                LEFT JOIN {self.silver}.hdmap hm ON cam.clip_id = hm.clip_id
                WHERE cam.camera_name = '{camera_name}'
            """)

        # Use count() to measure scan+join time without serialization overhead
        elapsed_gold, count_gold = self._time_query(gold_objdet, use_count=True)
        self._record(
            "Three-Workload", "Gold: Object Detection",
            f"camera_annotations WHERE camera_name='{camera_name}'",
            elapsed_gold, count_gold,
        )

        elapsed_silver, count_silver = self._time_query(
            silver_objdet, use_count=True
        )
        speedup = (
            f"{elapsed_silver / elapsed_gold:.1f}x"
            if elapsed_gold > 0 else "N/A"
        )
        self._record(
            "Three-Workload", "Silver JOIN: Object Detection",
            f"6-table join WHERE camera_name='{camera_name}'",
            elapsed_silver, count_silver,
            speedup=speedup,
        )

        # --- 1b: SLAM / Localization (lidar_with_ego) ---
        print(f"\n  [1b] SLAM / Localization (clip_id='{clip_id}')")

        def gold_slam():
            return self.spark.sql(
                f"SELECT * FROM {self.gold}.lidar_with_ego "
                f"WHERE clip_id = '{clip_id}'"
            )

        def silver_slam():
            return self.spark.sql(f"""
                SELECT
                    l.frame_id, l.clip_id, l.filename, l.sensor_timestamp,
                    e.translation AS ego_translation,
                    e.rotation AS ego_rotation,
                    cal.extrinsics
                FROM {self.silver}.lidar l
                LEFT JOIN {self.silver}.ego_motion e
                    ON l.frame_id = e.frame_id
                LEFT JOIN {self.silver}.calibration cal
                    ON l.clip_id = cal.clip_id
                    AND cal.sensor_name = 'lidar'
                WHERE l.clip_id = '{clip_id}'
            """)

        elapsed_gold, count_gold = self._time_query(gold_slam, use_count=True)
        self._record(
            "Three-Workload", "Gold: SLAM/Localization",
            f"lidar_with_ego WHERE clip_id='{clip_id}'",
            elapsed_gold, count_gold,
        )

        elapsed_silver, count_silver = self._time_query(
            silver_slam, use_count=True
        )
        speedup = (
            f"{elapsed_silver / elapsed_gold:.1f}x"
            if elapsed_gold > 0 else "N/A"
        )
        self._record(
            "Three-Workload", "Silver JOIN: SLAM/Localization",
            f"3-table join WHERE clip_id='{clip_id}'",
            elapsed_silver, count_silver,
            speedup=speedup,
        )

        # --- 1c: Multi-Modal Fusion (sensor_fusion_frame) ---
        print(f"\n  [1c] Multi-Modal Fusion (clip_id='{clip_id}')")

        def gold_fusion():
            return self.spark.sql(
                f"SELECT * FROM {self.gold}.sensor_fusion_frame "
                f"WHERE clip_id = '{clip_id}'"
            )

        def silver_fusion():
            return self.spark.sql(f"""
                SELECT
                    f.frame_id, f.clip_id, f.frame_idx,
                    cam_agg.cameras,
                    l.filename AS lidar_filename,
                    l.sensor_timestamp AS lidar_timestamp,
                    rad_agg.radars,
                    annot.annotations
                FROM {self.silver}.frame f
                LEFT JOIN (
                    SELECT frame_id,
                           collect_list(
                               struct(camera_name, filename, sensor_timestamp)
                           ) AS cameras
                    FROM {self.silver}.camera
                    GROUP BY frame_id
                ) cam_agg ON f.frame_id = cam_agg.frame_id
                LEFT JOIN {self.silver}.lidar l
                    ON f.frame_id = l.frame_id
                LEFT JOIN (
                    SELECT frame_id,
                           collect_list(
                               struct(radar_name, filename, sensor_timestamp)
                           ) AS radars
                    FROM {self.silver}.radar
                    GROUP BY frame_id
                ) rad_agg ON f.frame_id = rad_agg.frame_id
                LEFT JOIN (
                    SELECT frame_id,
                           collect_list(struct(boxes_3d, category)) AS annotations
                    FROM {self.silver}.dynamic_object
                    GROUP BY frame_id
                ) annot ON f.frame_id = annot.frame_id
                WHERE f.clip_id = '{clip_id}'
            """)

        elapsed_gold, count_gold = self._time_query(
            gold_fusion, use_count=True
        )
        self._record(
            "Three-Workload", "Gold: Multi-Modal Fusion",
            f"sensor_fusion_frame WHERE clip_id='{clip_id}'",
            elapsed_gold, count_gold,
        )

        elapsed_silver, count_silver = self._time_query(
            silver_fusion, use_count=True
        )
        speedup = (
            f"{elapsed_silver / elapsed_gold:.1f}x"
            if elapsed_gold > 0 else "N/A"
        )
        self._record(
            "Three-Workload", "Silver JOIN: Multi-Modal Fusion",
            f"6-table join + 3 aggregations WHERE clip_id='{clip_id}'",
            elapsed_silver, count_silver,
            speedup=speedup,
        )

    # =========================================================================
    # Experiment 2: Partition Pruning Measurement
    # =========================================================================

    def run_exp2_partition_pruning(self):
        """
        Measure actual Iceberg partition pruning effectiveness.

        Compares query performance with and without partition-aligned filters
        on the camera_annotations Gold table (partitioned by camera_name).
        Also creates an unpartitioned copy for direct comparison.
        Validates the paper's partition pruning analysis (Section VI-C).
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Partition Pruning Measurement")
        print("=" * 70)

        gold_table = f"{self.gold}.camera_annotations"

        # --- File-level metadata from the partitioned table ---
        try:
            total_files = self.spark.sql(
                f"SELECT count(*) as cnt FROM {gold_table}.files"
            ).collect()[0]["cnt"]

            files_by_partition = self.spark.sql(f"""
                SELECT partition, count(*) as file_count,
                       sum(record_count) as records
                FROM {gold_table}.files
                GROUP BY partition
            """).collect()

            print(f"\n  Partitioned table: {total_files} total data file(s)")
            for row in files_by_partition:
                print(
                    f"    {row['partition']}: "
                    f"{row['file_count']} file(s), "
                    f"{row['records']} records"
                )
        except Exception as e:
            print(f"  [NOTE] Could not read .files metadata: {e}")
            total_files = None

        # Get distinct camera names and a clip_id
        cameras = [
            r["camera_name"]
            for r in self.spark.sql(
                f"SELECT DISTINCT camera_name FROM {gold_table}"
            ).collect()
        ]
        target_camera = cameras[0] if cameras else "CAM_FRONT"
        num_cameras = len(cameras)

        sample_clip = self.spark.sql(
            f"SELECT DISTINCT clip_id FROM {gold_table} LIMIT 1"
        ).collect()[0]["clip_id"]

        total_rows = self.spark.sql(
            f"SELECT count(*) as cnt FROM {gold_table}"
        ).collect()[0]["cnt"]
        print(f"  Total rows: {total_rows}, Cameras: {num_cameras}")

        # --- 2a: Full scan (count all) ---
        print(f"\n  [2a] Full scan (no filter)")

        def partitioned_full():
            return self.spark.sql(
                f"SELECT count(*) as cnt FROM {gold_table}"
            )

        t_full, _ = self._time_query(partitioned_full)
        self._record(
            "Partition Pruning", "Full table count (all partitions)",
            f"No filter — reads all {num_cameras} partitions",
            t_full, total_rows,
        )

        # --- 2b: Sensor-only filter ---
        pruning_pct = (
            ((num_cameras - 1) / num_cameras * 100) if num_cameras > 1 else 0
        )
        print(
            f"\n  [2b] Sensor-only filter "
            f"(camera_name='{target_camera}', "
            f"theoretical pruning={pruning_pct:.1f}%)"
        )

        def partitioned_sensor():
            return self.spark.sql(
                f"SELECT * FROM {gold_table} "
                f"WHERE camera_name = '{target_camera}'"
            )

        t_sensor, cnt_sensor = self._time_query(partitioned_sensor)
        actual_pct = (
            (1 - cnt_sensor / total_rows) * 100 if total_rows > 0 else 0
        )
        self._record(
            "Partition Pruning", "Sensor filter (1 partition)",
            f"camera_name='{target_camera}'",
            t_sensor, cnt_sensor,
            theoretical_pruning=f"{pruning_pct:.1f}%",
            actual_data_reduction=f"{actual_pct:.1f}%",
            partitions_scanned=f"1/{num_cameras}",
        )

        # --- 2c: Combined filter (sensor + clip) ---
        print(f"\n  [2c] Combined filter (camera_name + clip_id)")

        def partitioned_combined():
            return self.spark.sql(
                f"SELECT * FROM {gold_table} "
                f"WHERE camera_name = '{target_camera}' "
                f"AND clip_id = '{sample_clip}'"
            )

        t_combined, cnt_combined = self._time_query(partitioned_combined)
        combined_pct = (
            (1 - cnt_combined / total_rows) * 100 if total_rows > 0 else 0
        )
        self._record(
            "Partition Pruning", "Combined filter (sensor + clip)",
            f"camera_name + clip_id — maximum pruning",
            t_combined, cnt_combined,
            data_reduction=f"{combined_pct:.1f}%",
        )

        # --- 2d: Create unpartitioned copy for comparison ---
        print(f"\n  [2d] Unpartitioned comparison")
        unpart_table = f"{self.gold}.camera_annotations_unpartitioned"
        try:
            # Use a subset (single camera) to avoid OOM
            self.spark.sql(f"""
                CREATE OR REPLACE TABLE {unpart_table}
                USING iceberg
                TBLPROPERTIES ('format-version' = '2')
                AS SELECT * FROM {gold_table}
                   WHERE camera_name = '{target_camera}'
            """)

            def unpart_query():
                return self.spark.sql(
                    f"SELECT * FROM {unpart_table}"
                )

            def part_equiv_query():
                return self.spark.sql(
                    f"SELECT * FROM {gold_table} "
                    f"WHERE camera_name = '{target_camera}'"
                )

            t_unpart, cnt_unpart = self._time_query(unpart_query)
            t_part_equiv, cnt_part = self._time_query(part_equiv_query)
            speedup = (
                f"{t_unpart / t_part_equiv:.1f}x"
                if t_part_equiv > 0 else "N/A"
            )

            self._record(
                "Partition Pruning", "Unpartitioned (equivalent data)",
                f"{cnt_unpart} rows, no partition pruning",
                t_unpart, cnt_unpart,
            )
            self._record(
                "Partition Pruning", "Partitioned (equivalent query)",
                f"{cnt_part} rows, partition-pruned",
                t_part_equiv, cnt_part,
                speedup=speedup,
            )

            self.spark.sql(f"DROP TABLE IF EXISTS {unpart_table}")
        except Exception as e:
            print(f"  [NOTE] Unpartitioned comparison failed: {e}")

    # =========================================================================
    # Experiment 3: Temporal Sequence Replay
    # =========================================================================

    def run_exp3_temporal_replay(self):
        """
        Benchmark sequential frame access: sorted Silver vs unsorted Bronze.

        The Silver layer's sort-within-partitions ensures temporal locality
        for AD data, which is critical for sequential frame replay during
        training and evaluation.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Temporal Sequence Replay")
        print("=" * 70)

        # Get a clip_id with the most frames
        clip_row = self.spark.sql(f"""
            SELECT clip_id, count(*) as frame_count
            FROM {self.silver}.frame
            GROUP BY clip_id
            ORDER BY frame_count DESC
            LIMIT 1
        """).collect()[0]
        clip_id = clip_row["clip_id"]
        frame_count = clip_row["frame_count"]

        print(f"\n  Target clip_id: '{clip_id}' ({frame_count} frames)")

        # --- 3a: Frame table sequential access ---
        print(f"\n  [3a] Frame sequential access (ordered by frame_idx)")

        def silver_frames():
            return self.spark.sql(
                f"SELECT * FROM {self.silver}.frame "
                f"WHERE clip_id = '{clip_id}' "
                f"ORDER BY frame_idx"
            )

        def bronze_frames():
            return self.spark.sql(
                f"SELECT * FROM {self.bronze}.frame "
                f"WHERE clip_id = '{clip_id}' "
                f"ORDER BY frame_idx"
            )

        t_silver, cnt_silver = self._time_query(silver_frames)
        t_bronze, cnt_bronze = self._time_query(bronze_frames)
        speedup = f"{t_bronze / t_silver:.1f}x" if t_silver > 0 else "N/A"
        self._record(
            "Temporal Replay", "Silver: frame sequential",
            f"Sorted by (clip_id, frame_idx), clip_id='{clip_id}'",
            t_silver, cnt_silver,
        )
        self._record(
            "Temporal Replay", "Bronze: frame sequential",
            f"Unsorted, requires sort at query time",
            t_bronze, cnt_bronze,
            speedup=speedup,
        )

        # --- 3b: Camera sensor temporal replay ---
        print(f"\n  [3b] Camera sensor temporal replay "
              f"(ordered by sensor_timestamp)")

        def silver_cam_replay():
            return self.spark.sql(
                f"SELECT * FROM {self.silver}.camera "
                f"WHERE clip_id = '{clip_id}' "
                f"ORDER BY sensor_timestamp"
            )

        def bronze_cam_replay():
            return self.spark.sql(
                f"SELECT * FROM {self.bronze}.camera "
                f"WHERE clip_id = '{clip_id}' "
                f"ORDER BY sensor_timestamp"
            )

        t_silver, cnt_silver = self._time_query(silver_cam_replay)
        t_bronze, cnt_bronze = self._time_query(bronze_cam_replay)
        speedup = f"{t_bronze / t_silver:.1f}x" if t_silver > 0 else "N/A"
        self._record(
            "Temporal Replay", "Silver: camera temporal",
            f"Sorted by (clip_id, sensor_timestamp)",
            t_silver, cnt_silver,
        )
        self._record(
            "Temporal Replay", "Bronze: camera temporal",
            f"Unsorted, requires sort at query time",
            t_bronze, cnt_bronze,
            speedup=speedup,
        )

        # --- 3c: LiDAR ego-motion temporal replay (Gold table) ---
        print(f"\n  [3c] LiDAR + ego motion temporal replay "
              f"(Gold table, ordered by sensor_timestamp)")

        def gold_lidar_replay():
            return self.spark.sql(
                f"SELECT * FROM {self.gold}.lidar_with_ego "
                f"WHERE clip_id = '{clip_id}' "
                f"ORDER BY sensor_timestamp"
            )

        def silver_lidar_replay():
            return self.spark.sql(f"""
                SELECT l.*, e.translation, e.rotation
                FROM {self.silver}.lidar l
                LEFT JOIN {self.silver}.ego_motion e
                    ON l.frame_id = e.frame_id
                WHERE l.clip_id = '{clip_id}'
                ORDER BY l.sensor_timestamp
            """)

        t_gold, cnt_gold = self._time_query(gold_lidar_replay)
        t_silver, cnt_silver = self._time_query(silver_lidar_replay)
        speedup = f"{t_silver / t_gold:.1f}x" if t_gold > 0 else "N/A"
        self._record(
            "Temporal Replay", "Gold: lidar+ego temporal",
            f"Pre-joined & pre-sorted Gold table",
            t_gold, cnt_gold,
        )
        self._record(
            "Temporal Replay", "Silver JOIN: lidar+ego temporal",
            f"Runtime join + sort",
            t_silver, cnt_silver,
            speedup=speedup,
        )

    # =========================================================================
    # Experiment 4: Time Travel for Training Reproducibility
    # =========================================================================

    def run_exp4_time_travel(self):
        """
        Demonstrate Iceberg time travel for training dataset reproducibility.

        Shows that the lakehouse natively supports pinning training datasets
        to exact snapshots — a critical requirement for both regulatory
        compliance and scientific reproducibility in AD.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: Time Travel for Training Reproducibility")
        print("=" * 70)

        demo_table = f"{self.gold}.time_travel_demo"

        try:
            # Step 1: Create a demo table from a subset of Gold data
            print("\n  Creating demo table from camera_annotations subset...")
            target_camera = self.spark.sql(
                f"SELECT DISTINCT camera_name "
                f"FROM {self.gold}.camera_annotations LIMIT 1"
            ).collect()[0]["camera_name"]

            self.spark.sql(f"""
                CREATE OR REPLACE TABLE {demo_table}
                USING iceberg
                TBLPROPERTIES ('format-version' = '2')
                AS SELECT * FROM {self.gold}.camera_annotations
                   WHERE camera_name = '{target_camera}'
            """)

            # Step 2: Record the first snapshot (training dataset v1)
            snap1 = self.spark.sql(f"""
                SELECT snapshot_id, committed_at
                FROM {demo_table}.snapshots
                ORDER BY committed_at ASC
                LIMIT 1
            """).collect()[0]
            snap1_id = snap1["snapshot_id"]
            snap1_ts = snap1["committed_at"]

            count_v1 = self.spark.sql(
                f"SELECT count(*) as cnt FROM {demo_table}"
            ).collect()[0]["cnt"]
            print(f"  Snapshot v1: {snap1_id} @ {snap1_ts} — {count_v1} rows")

            # Step 3: Simulate new data ingestion (training dataset v2)
            print("  Simulating new data ingestion (appending duplicate rows)...")
            self.spark.sql(f"""
                INSERT INTO {demo_table}
                SELECT * FROM {demo_table}
            """)

            count_v2 = self.spark.sql(
                f"SELECT count(*) as cnt FROM {demo_table}"
            ).collect()[0]["cnt"]
            print(f"  Current (v2): {count_v2} rows (+{count_v2 - count_v1})")

            # Step 4: Time travel — pin training dataset to v1
            print("  Time travel query: VERSION AS OF (training dataset v1)...")

            def time_travel_read():
                return self.spark.sql(
                    f"SELECT * FROM {demo_table} "
                    f"VERSION AS OF {snap1_id}"
                )

            elapsed, travel_count = self._time_query(
                time_travel_read, warmup=1, runs=3
            )

            match = travel_count == count_v1
            print(
                f"  Time travel result: {travel_count} rows "
                f"(matches v1: {match}) in {elapsed:.4f}s"
            )

            self._record(
                "Time Travel", "Snapshot pinned read",
                f"VERSION AS OF {snap1_id}",
                elapsed, travel_count,
                v1_count=str(count_v1),
                v2_count=str(count_v2),
                match=str(match),
            )

            # Step 5: Show snapshot history
            history = self.spark.sql(f"""
                SELECT snapshot_id, committed_at, operation
                FROM {demo_table}.snapshots
                ORDER BY committed_at
            """).collect()
            print(f"\n  Snapshot history ({len(history)} snapshots):")
            for h in history:
                print(
                    f"    {h['snapshot_id']} | "
                    f"{h['committed_at']} | "
                    f"{h['operation']}"
                )

        except Exception as e:
            print(f"  [ERROR] Time travel experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            self.spark.sql(f"DROP TABLE IF EXISTS {demo_table}")

    # =========================================================================
    # Experiment 5: Column-Level Metrics Effectiveness
    # =========================================================================

    def run_exp5_column_metrics(self):
        """
        Demonstrate column-level metrics for timestamp predicate pushdown.

        With full min/max metrics on sensor_timestamp, Iceberg can skip
        entire data files whose timestamp range doesn't overlap the query
        window — without reading any data.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: Column-Level Metrics for Timestamp Queries")
        print("=" * 70)

        # Get timestamp range from Silver camera table
        ts_stats = self.spark.sql(f"""
            SELECT
                min(sensor_timestamp) as min_ts,
                max(sensor_timestamp) as max_ts,
                count(*) as total_rows
            FROM {self.silver}.camera
        """).collect()[0]

        min_ts = ts_stats["min_ts"]
        max_ts = ts_stats["max_ts"]
        total_rows = ts_stats["total_rows"]

        if min_ts is None or max_ts is None:
            print("  [SKIP] No timestamp data found")
            return

        ts_range = max_ts - min_ts
        if ts_range == 0:
            print("  [SKIP] All timestamps identical — no range to test")
            return

        # Narrow window: ~5% of total timestamp range
        mid_ts = min_ts + ts_range // 2
        narrow_window = max(ts_range // 20, 1)

        print(f"\n  Timestamp range: {min_ts} to {max_ts} ({ts_range} span)")
        print(f"  Narrow window: {mid_ts} to {mid_ts + narrow_window} (~5%)")
        print(f"  Total rows: {total_rows}")

        # Narrow timestamp range query (benefits from column metrics)
        def narrow_ts_query():
            return self.spark.sql(
                f"SELECT * FROM {self.silver}.camera "
                f"WHERE sensor_timestamp BETWEEN "
                f"{mid_ts} AND {mid_ts + narrow_window}"
            )

        # Full scan (no filter)
        def full_scan_query():
            return self.spark.sql(
                f"SELECT * FROM {self.silver}.camera"
            )

        t_narrow, cnt_narrow = self._time_query(narrow_ts_query)
        t_full, cnt_full = self._time_query(full_scan_query)
        speedup = f"{t_full / t_narrow:.1f}x" if t_narrow > 0 else "N/A"

        selectivity = (
            f"{cnt_narrow / cnt_full * 100:.1f}%"
            if cnt_full > 0 else "N/A"
        )

        self._record(
            "Column Metrics", "Narrow timestamp range",
            f"5% window — {cnt_narrow}/{cnt_full} rows ({selectivity})",
            t_narrow, cnt_narrow,
            selectivity=selectivity,
        )
        self._record(
            "Column Metrics", "Full scan (no filter)",
            f"All {cnt_full} rows",
            t_full, cnt_full,
            speedup=speedup,
        )

        # Also test Silver frame table with frame_idx range
        print(f"\n  Frame index range query:")
        frame_stats = self.spark.sql(f"""
            SELECT min(frame_idx) as min_idx,
                   max(frame_idx) as max_idx,
                   count(*) as total
            FROM {self.silver}.frame
        """).collect()[0]

        min_idx = frame_stats["min_idx"]
        max_idx = frame_stats["max_idx"]
        idx_range = max_idx - min_idx if max_idx and min_idx else 0

        if idx_range > 0:
            mid_idx = min_idx + idx_range // 2
            narrow_idx = max(idx_range // 20, 1)

            def narrow_idx_query():
                return self.spark.sql(
                    f"SELECT * FROM {self.silver}.frame "
                    f"WHERE frame_idx BETWEEN "
                    f"{mid_idx} AND {mid_idx + narrow_idx}"
                )

            t_narrow_idx, cnt_narrow_idx = self._time_query(narrow_idx_query)

            self._record(
                "Column Metrics", "Narrow frame_idx range",
                f"5% window — {cnt_narrow_idx} rows",
                t_narrow_idx, cnt_narrow_idx,
            )

    # =========================================================================
    # Report Generation
    # =========================================================================

    def run_all(self):
        """Run all experiments and print results."""
        print("\n" + "#" * 70)
        print("# AD-SPECIFIC WORKLOAD BENCHMARKS")
        print("# Autonomous Driving Data Lakehouse")
        print("#" * 70)

        # Warm up the JVM before any timed experiments
        self._warmup_jvm()

        experiments = [
            ("Exp 1: Three-Workload Gold vs Silver", self.run_exp1_three_workloads),
            ("Exp 2: Partition Pruning", self.run_exp2_partition_pruning),
            ("Exp 3: Temporal Replay", self.run_exp3_temporal_replay),
            ("Exp 4: Time Travel", self.run_exp4_time_travel),
            ("Exp 5: Column Metrics", self.run_exp5_column_metrics),
        ]

        for name, fn in experiments:
            if not self._is_spark_alive():
                print(f"\n  [SKIP] {name} — SparkContext is no longer active")
                continue
            try:
                fn()
            except Exception as e:
                print(f"\n  [ERROR] {name} failed: {e}")
                import traceback
                traceback.print_exc()

        self.print_report()

    def print_report(self):
        """Print a formatted report of all benchmark results."""
        print("\n\n" + "=" * 90)
        print("  BENCHMARK RESULTS SUMMARY")
        print("=" * 90)

        # Group by experiment
        experiments = defaultdict(list)
        for r in self.results:
            experiments[r.experiment].append(r)

        for exp_name, results in experiments.items():
            print(f"\n  {'─' * 86}")
            print(f"  {exp_name}")
            print(f"  {'─' * 86}")
            print(
                f"  {'Variant':<42} "
                f"{'Time (s)':>10} "
                f"{'Rows':>10} "
                f"{'Notes':>20}"
            )
            print(f"  {'─' * 86}")

            for r in results:
                notes = ""
                if "speedup" in r.extra:
                    notes = f"↑ {r.extra['speedup']}"
                elif "theoretical_pruning" in r.extra:
                    notes = f"~{r.extra['theoretical_pruning']} pruned"
                elif "match" in r.extra:
                    notes = f"match={r.extra['match']}"
                elif "selectivity" in r.extra:
                    notes = f"sel={r.extra['selectivity']}"

                print(
                    f"  {r.variant:<42} "
                    f"{r.elapsed_seconds:>10.4f} "
                    f"{r.row_count:>10} "
                    f"{notes:>20}"
                )

        print(f"\n{'=' * 90}")

        # Export results as JSON for paper figures
        json_path = "/opt/spark/benchmarks/benchmark_results.json"
        json_results = [
            {
                "experiment": r.experiment,
                "variant": r.variant,
                "description": r.description,
                "elapsed_seconds": round(r.elapsed_seconds, 6),
                "row_count": r.row_count,
                **r.extra,
            }
            for r in self.results
        ]

        try:
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)
            print(f"\n  Results exported to {json_path}")
        except Exception as e:
            print(f"\n  Failed to export results: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    config = PipelineConfig()
    spark = build_spark_session(config, app_name="ad-workload-benchmarks")

    try:
        suite = BenchmarkSuite(spark, config)
        suite.run_all()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
