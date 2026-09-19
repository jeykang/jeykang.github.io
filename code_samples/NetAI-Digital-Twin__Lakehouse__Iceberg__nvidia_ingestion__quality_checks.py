"""Silver-tier data quality checks against canonical Bronze schema.

Operates on the 16 canonical tables (Session, Clip, Camera, Lidar, Radar,
Frame, EgoMotion, Calibration, …) — see kaist_schema_v2.dbml.

Four checks:
  1. missing_sensors  — clips in Clip without rows in Camera/Lidar/Radar/EgoMotion
  2. timestamps       — non-monotonic / out-of-range timestamps in EgoMotion
  3. camera_corrupt   — Camera.filename samples that don't exist on disk or
                        whose mp4 box header is invalid (WARN, not FAIL)
  4. schema           — null-column check via row sampling

Outputs `iceberg.nvidia_silver.quality_report` (one row per (clip_id, check)).
Silver views then filter Bronze to clips without FAIL findings.
"""
from __future__ import annotations

import glob
import logging
import os
import struct
import time
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces

log = logging.getLogger("nvidia.quality")

QUALITY_SCHEMA = StructType([
    StructField("clip_id", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("status", StringType(), False),
    StructField("detail", StringType(), True),
    StructField("sensor", StringType(), True),
    StructField("checked_at", StringType(), False),
])

# Canonical Bronze tables that get Silver views (excluding Frame/Session — too big
# and not gated by per-clip quality). Empty tables are skipped.
SILVER_VIEW_TABLES = [
    "Clip", "Calibration", "Camera", "Lidar", "Radar", "EgoMotion",
]


@dataclass
class QualityResult:
    clip_id: str
    check_name: str
    status: str   # PASS, FAIL, WARN
    detail: str = ""
    sensor: str = ""


class QualityChecker:
    def __init__(self, spark: SparkSession, config: NvidiaPipelineConfig):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.ns_b = config.nvidia.namespace_bronze
        self.source_path = config.nvidia.source_path
        self._ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    def _bronze(self, t: str) -> str:
        return f"{self.cat}.{self.ns_b}.{t}"

    # -- Check 1: missing_sensors ---------------------------------------------

    def check_missing_sensors(self) -> List[QualityResult]:
        """Flag clips that are genuinely missing a sensor NVIDIA declares present.

        Source of truth for *expected* sensors is `aux_sensor_presence`
        (= v26.03 `feature_presence.parquet`): a per-clip boolean matrix with
        `lidar_top_360fov`, the 19 `radar_*` flags, and `radar_config`. The
        prior implementation used a hardcoded `MIN_RADAR_SENSORS=10` floor and
        treated any absent Lidar as FAIL; an audit of the 190 FAILs it produced
        (bad_clip_lists/missing_sensor_classified.csv, 2026-06-15) found ~all of
        them non-genuine, in these categories:

          - BY_DESIGN_radar_config (95): clip declares <10 radars (low config
            expects ~9; some expect 8 or 0). Fixed: compare actual distinct
            radar_name against the *per-clip expected* count from feature_presence.
          - BY_DESIGN_no_lidar (23): feature_presence declares lidar absent.
            Fixed: only check Lidar when `lidar_top_360fov` is true.
          - ORPHAN_not_in_nvidia_index (26): version-skew clips whose camera
            files survived on disk but are absent from the v26.03 index (Open
            Issue #12). Fixed: restrict the universe via inner-join to
            feature_presence (current-index membership).
          - OUT_OF_SCOPE_not_downloaded (~9): clips in chunks we never
            downloaded — they have no Camera rows, so the Camera-presence
            universe already excludes them.
          - FALSE_POSITIVE_bronze_bug (32): lidar present & non-empty on disk
            but absent from the canonical Lidar table (a registration gap, not a
            clip-quality failure). Fixed: a disk cross-check downgrades these
            from FAIL to WARN.

        A clip is only FAILed when an expected sensor is missing from Bronze
        *and* (for lidar) absent from disk.
        """
        print("[QUALITY] Check: missing sensors per clip")
        results: List[QualityResult] = []

        # Expected sensors per clip, from feature_presence (current-index truth).
        sp = self.spark.table(self._bronze("aux_sensor_presence"))
        radar_cols = [c for c in sp.columns
                      if c.startswith("radar_") and c != "radar_config"]
        expected_radars = reduce(
            lambda a, b: a + b,
            [F.col(c).cast("int") for c in radar_cols],
        ) if radar_cols else F.lit(0)
        # The lidar presence flag is `lidar_top_360fov` in v25.10 sensor_presence;
        # detect it defensively in case v26.03 feature_presence renamed it. If no
        # lidar flag is found, assume lidar is expected (no by-design exclusion).
        lidar_col = next(
            (c for c in ("lidar_top_360fov", "lidar") if c in sp.columns),
            next((c for c in sp.columns if c.startswith("lidar")), None),
        )
        if lidar_col is None:
            log.warning("aux_sensor_presence has no lidar flag column; "
                        "treating lidar as expected for all clips")
        expect_lidar = (F.col(lidar_col).cast("boolean") if lidar_col
                        else F.lit(True))
        expected = sp.select(
            "clip_id",
            expect_lidar.alias("expect_lidar"),
            expected_radars.alias("expect_radars"),
        )

        # Universe = clips that are (a) in the canonical Clip table, (b) locally
        # present (have Camera rows), and (c) in the current v26.03 index
        # (inner-join to feature_presence drops version-skew orphans).
        clips = self.spark.table(self._bronze("Clip")).select("clip_id").distinct()
        cam_clips = self.spark.table(self._bronze("Camera")).select("clip_id").distinct()
        universe = (
            clips.join(cam_clips, on="clip_id", how="inner")
            .join(expected, on="clip_id", how="inner")
            .cache()
        )
        n_universe = universe.count()
        print(f"  Universe (Clip ∩ Camera ∩ feature_presence): {n_universe:,}")

        # -- Lidar: only clips that declare lidar present --------------------
        lid_clips = self.spark.table(self._bronze("Lidar")).select("clip_id").distinct()
        cand_lidar = [
            r.clip_id for r in (
                universe.filter(F.col("expect_lidar"))
                .select("clip_id")
                .join(lid_clips, on="clip_id", how="left_anti")
                .collect()
            )
        ]
        n_fail_lidar = n_warn_lidar = 0
        for clip_id in cand_lidar:
            if self._lidar_on_disk(clip_id):
                # Data exists on disk but isn't in canonical Lidar: registration
                # gap, not a clip-quality failure → WARN (doesn't exclude clip).
                results.append(QualityResult(
                    clip_id, "missing_sensor", "WARN",
                    "Lidar on disk but absent from Bronze (registration gap)", "lidar"))
                n_warn_lidar += 1
            else:
                results.append(QualityResult(
                    clip_id, "missing_sensor", "FAIL", "No Lidar rows", "lidar"))
                n_fail_lidar += 1
        print(f"  Lidar: {n_fail_lidar:,} FAIL, {n_warn_lidar:,} WARN "
              f"(of {len(cand_lidar):,} declared-present but absent from Bronze)")

        # -- EgoMotion: online egomotion is expected for every clip ---------
        ego_clips = self.spark.table(self._bronze("EgoMotion")).select("clip_id").distinct()
        missing_ego = universe.select("clip_id").join(ego_clips, on="clip_id", how="left_anti")
        n_missing_ego = missing_ego.count()
        for r in missing_ego.limit(100).collect():
            results.append(QualityResult(r.clip_id, "missing_sensor", "FAIL",
                                         "No EgoMotion rows", "egomotion"))
        if n_missing_ego > 100:
            results.append(QualityResult("*", "missing_sensor_summary", "WARN",
                                         f"{n_missing_ego} clips missing EgoMotion (first 100 only)",
                                         "egomotion"))
        print(f"  EgoMotion: {n_missing_ego:,} clips missing")

        # -- Radar: actual distinct radar_name < per-clip expected ----------
        radar_per_clip = (
            self.spark.table(self._bronze("Radar"))
            .groupBy("clip_id")
            .agg(F.countDistinct("radar_name").alias("n_sensors"))
        )
        low_radar = (
            universe.join(radar_per_clip, on="clip_id", how="left")
            .withColumn("n_sensors", F.coalesce(F.col("n_sensors"), F.lit(0)))
            .filter((F.col("expect_radars") > 0)
                    & (F.col("n_sensors") < F.col("expect_radars")))
        )
        n_low_radar = low_radar.count()
        for r in low_radar.select("clip_id", "n_sensors", "expect_radars").limit(100).collect():
            results.append(QualityResult(
                r.clip_id, "missing_sensor", "FAIL",
                f"Only {r.n_sensors} radar sensors (expected {r.expect_radars})", "radar"))
        if n_low_radar > 100:
            results.append(QualityResult("*", "missing_sensor_summary", "WARN",
                                         f"{n_low_radar} clips below expected radar count (first 100 only)",
                                         "radar"))
        print(f"  Radar: {n_low_radar:,} clips below per-clip expected count")
        universe.unpersist()
        return results

    def _lidar_on_disk(self, clip_id: str) -> bool:
        """True if a non-empty lidar parquet for this clip exists on NFS.

        Mirrors the glob the Gold scorer uses (edge_case_scorer._score_per_clip).
        """
        lidar_dir = os.path.join(self.source_path, "lidar", "lidar_top_360fov")
        if not os.path.isdir(lidar_dir):
            return False
        for p in glob.glob(os.path.join(lidar_dir, "**", f"{clip_id}*.parquet"),
                           recursive=True):
            try:
                if os.path.getsize(p) > 0:
                    return True
            except OSError:
                continue
        return False

    # -- Check 2: timestamps ---------------------------------------------------

    def check_timestamps(self) -> List[QualityResult]:
        """Per-clip timestamp sanity from aux_egomotion (dense per-clip ts).

        Uses aux_egomotion rather than canonical Frame because:
          - canonical Frame is the union of distinct (clip_id, sensor_timestamp)
            across egomotion+lidar+camera+radar; clips with sparse sensor coverage
            (e.g. cameras only) end up with very few rows in Frame, so a
            per-clip count threshold there flags them as "anomalous" when in
            fact their issue is missing sensors (caught separately).
          - aux_egomotion has the original ~3000 timestamps per clip at ~15Hz,
            making count/min/max thresholds genuinely meaningful.

        PhysicalAI egomotion uses per-clip relative microseconds with a
        ~200ms pre-roll, so min_ts may legitimately be negative down to
        -200000us. Only flag min_ts < -1_000_000us as anomalous.

        Clips with no aux_egomotion data are not flagged here — the
        missing_sensors check handles those.
        """
        print("[QUALITY] Check: timestamp quality (aux_egomotion)")
        UUID_RE = (
            r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."
        )
        ego = (
            self.spark.table(self._bronze("aux_egomotion"))
            .withColumn("clip_id", F.regexp_extract(F.input_file_name(), UUID_RE, 1))
            .filter(F.col("clip_id") != "")
        )
        per_clip = (
            ego.groupBy("clip_id")
            .agg(
                F.min("timestamp").alias("min_ts"),
                F.max("timestamp").alias("max_ts"),
                F.count("timestamp").alias("n"),
            )
            .filter(
                # Thresholds tuned 2026-05-04 against actual distribution
                # (probe_ts_range.py): healthy max_ts ranges 139M-141M (clips
                # are NOT 20s as docs imply — actual range is ~140s); n median
                # 2,814 with 1%-tile 2,036; min_ts always ~-200,000 (200ms
                # pre-roll). Old `max_ts > 60M` flagged 100% of healthy clips.
                (F.col("min_ts") < -1_000_000)
                | (F.col("max_ts") > 200_000_000)
                | (F.col("n") < 1000)
            )
        )
        rows = per_clip.collect()
        results = []
        for r in rows:
            results.append(QualityResult(
                r.clip_id, "timestamps", "FAIL",
                f"min={r.min_ts}, max={r.max_ts}, n={r.n}", "egomotion"))
        print(f"  {len(results)} clips with timestamp anomalies")
        return results

    # -- Check 3: camera_corrupt ----------------------------------------------

    def check_camera_integrity(self, sample_per_camera: int = 100) -> List[QualityResult]:
        """Sample N Camera rows per camera; verify mp4 exists and has a valid
        ftyp/moov box header in the first 12 bytes. WARN, not FAIL — gaps in
        camera data don't automatically exclude clips from non-camera Silver views.
        """
        print(f"[QUALITY] Check: camera mp4 integrity (sample {sample_per_camera}/sensor)")
        cam = self.spark.table(self._bronze("Camera"))
        sensors = [r.camera_name for r in cam.select("camera_name").distinct().collect()]
        results: List[QualityResult] = []
        for sensor in sensors:
            sample = (cam.filter(F.col("camera_name") == sensor)
                      .select("clip_id", "filename")
                      .distinct().limit(sample_per_camera).collect())
            for r in sample:
                path = r["filename"].replace("file:", "", 1)
                if not os.path.exists(path):
                    results.append(QualityResult(r["clip_id"], "camera_corrupt", "WARN",
                                                 f"mp4 missing: {os.path.basename(path)}", sensor))
                    continue
                try:
                    with open(path, "rb") as f:
                        head = f.read(12)
                    if len(head) < 12 or not (b"ftyp" in head or b"moov" in head):
                        results.append(QualityResult(r["clip_id"], "camera_corrupt", "WARN",
                                                     "invalid mp4 box header", sensor))
                except OSError as e:
                    results.append(QualityResult(r["clip_id"], "camera_corrupt", "WARN",
                                                 f"open failed: {e}", sensor))
        print(f"  {len(results)} sample mp4s flagged")
        return results

    # -- Check 4: schema ------------------------------------------------------

    def check_schema(self) -> List[QualityResult]:
        """For each canonical table, sample 1000 rows and check for all-null columns."""
        print("[QUALITY] Check: schema null-column scan")
        results = []
        for tbl in SILVER_VIEW_TABLES + ["Frame", "Session"]:
            try:
                df = self.spark.table(self._bronze(tbl)).limit(1000)
                cols = df.columns
                if not cols:
                    continue
                nullness = df.agg(*[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
                                    for c in cols]).collect()[0]
                n = df.count()
                for c in cols:
                    if nullness[c] == n and n > 0:
                        results.append(QualityResult("*", "schema_null_column", "WARN",
                                                     f"{tbl}.{c} all-null in 1000-row sample", tbl))
            except Exception as e:
                results.append(QualityResult("*", "schema_unreadable", "FAIL",
                                             f"{tbl}: {str(e)[:200]}", tbl))
        print(f"  {len(results)} schema findings")
        return results

    # -- Run all --------------------------------------------------------------

    def run_all_checks(self, checks: Optional[List[str]] = None) -> DataFrame:
        all_checks = {
            "missing_sensors": self.check_missing_sensors,
            "timestamps": self.check_timestamps,
            "camera": self.check_camera_integrity,
            "schema": self.check_schema,
        }
        if checks:
            all_checks = {k: v for k, v in all_checks.items() if k in checks}
        results: List[QualityResult] = []
        for name, fn in all_checks.items():
            t0 = time.perf_counter()
            before = len(results)
            try:
                results.extend(fn())
            except Exception as e:
                log.exception("Check %s crashed", name)
                results.append(QualityResult("*", name, "FAIL",
                                             f"check crashed: {e}"))
            produced = results[before:]
            print(f"  → {name}: {len(produced)} findings "
                  f"({sum(1 for r in produced if r.status == 'FAIL')} FAIL, "
                  f"{sum(1 for r in produced if r.status == 'WARN')} WARN) "
                  f"in {time.perf_counter() - t0:.1f}s")

        rows = [(r.clip_id, r.check_name, r.status, r.detail, r.sensor, self._ts)
                for r in results]
        return self.spark.createDataFrame(rows, schema=QUALITY_SCHEMA)


# ---------------------------------------------------------------------------
# Silver view builder
# ---------------------------------------------------------------------------

class SilverQualityBuilder:
    def __init__(self, spark: SparkSession, config: NvidiaPipelineConfig):
        self.spark = spark
        self.config = config
        self.cat = config.spark_catalog_name
        self.ns_b = config.nvidia.namespace_bronze
        self.ns_s = config.nvidia.namespace_silver

    def build_silver_views(self) -> Dict[str, int]:
        """Create a Silver view per canonical Bronze table that excludes
        clips with FAIL quality findings.
        """
        qr = f"{self.cat}.{self.ns_s}.quality_report"
        # Distinct clip_ids with at least one FAIL (non-global, real per-clip)
        failed_view = f"{self.cat}.{self.ns_s}._failed_clips"
        self.spark.sql(f"""
            CREATE OR REPLACE VIEW {failed_view} AS
            SELECT DISTINCT clip_id FROM {qr}
            WHERE status = 'FAIL' AND clip_id != '*'
        """)
        results = {}
        for tbl in SILVER_VIEW_TABLES:
            silver_view = f"{self.cat}.{self.ns_s}.{tbl}"
            bronze_tbl = f"{self.cat}.{self.ns_b}.{tbl}"
            self.spark.sql(f"""
                CREATE OR REPLACE VIEW {silver_view} AS
                SELECT * FROM {bronze_tbl}
                WHERE clip_id NOT IN (SELECT clip_id FROM {failed_view})
            """)
            try:
                results[tbl] = self.spark.table(silver_view).count()
            except Exception as e:
                log.warning("Silver view %s count failed: %s", silver_view, e)
                results[tbl] = -1
            print(f"  [VIEW] {silver_view}: {results[tbl]:,} rows")
        return results


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_quality_pipeline(
    config: Optional[NvidiaPipelineConfig] = None,
    checks: Optional[List[str]] = None,
    build_views: bool = True,
) -> Tuple[DataFrame, Dict[str, int]]:
    if config is None:
        config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="nvidia-quality")
    try:
        create_namespaces(spark, config)
        checker = QualityChecker(spark, config)
        report_df = checker.run_all_checks(checks=checks)

        qr_table = f"{config.spark_catalog_name}.{config.nvidia.namespace_silver}.quality_report"
        report_df.writeTo(qr_table).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()
        qr_count = spark.table(qr_table).count()
        fail_count = report_df.filter(F.col("status") == "FAIL").count()
        warn_count = report_df.filter(F.col("status") == "WARN").count()
        failed_clips = (
            report_df.filter((F.col("status") == "FAIL") & (F.col("clip_id") != "*"))
            .select("clip_id").distinct().count()
        )
        print(f"\n{'='*60}")
        print(f"QUALITY REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"  Total findings:  {qr_count}")
        print(f"  FAIL:            {fail_count}")
        print(f"  WARN:            {warn_count}")
        print(f"  Clips excluded:  {failed_clips}")
        print(f"  Written to:      {qr_table}")

        view_counts: Dict[str, int] = {}
        if build_views:
            print(f"\n[SILVER VIEWS] Building quality-filtered views ...")
            view_counts = SilverQualityBuilder(spark, config).build_silver_views()
        return report_df, view_counts
    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Silver data quality pipeline (canonical schema)")
    p.add_argument("--checks", nargs="*",
                   choices=["missing_sensors", "timestamps", "camera", "schema"])
    p.add_argument("--no-views", action="store_true")
    args = p.parse_args()
    run_quality_pipeline(checks=args.checks, build_views=not args.no_views)
