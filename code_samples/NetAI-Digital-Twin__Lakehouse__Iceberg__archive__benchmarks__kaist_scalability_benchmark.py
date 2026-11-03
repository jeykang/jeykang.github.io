#!/usr/bin/env python3
"""
KAIST Scalability Benchmark — Actual Bronze/Silver/Gold Tiers.

Measures query latency on the REAL KAIST medallion architecture tables at
scale factors 1x–50x.  Unlike the earlier nuScenes-based scalability
benchmark (nuscenes_experiment/scalability_benchmark.py), this one:

  - Uses the actual KAIST simulated data (camera.json, dynamic_object.json …)
  - Runs the Python baseline against KAIST JSON with the same access pattern
    as the real pipeline
  - Queries the actual kaist_silver namespace tables with the exact 6-table
    join used by build_gold.build_camera_annotations()
  - Queries the actual kaist_gold table structure (camera_annotations
    partitioned by camera_name)
  - Scales only the fact tables (camera, dynamic_object) while keeping
    dimension tables (frame, clip, calibration, hdmap) at 1x — the same
    realistic growth model

Target workload:
  "Get all camera images + annotations + calibration for a single camera"
  — the Object Detection training data assembly query.

Usage (inside spark-iceberg container):
    python benchmarks/kaist_scalability_benchmark.py

Outputs:
    benchmarks/kaist_scalability_results.json
"""

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAIST_DATA_PATH = os.environ.get("KAIST_SOURCE_PATH", "/user_data/kaist-simulated")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "kaist_scalability_results.json")

# 1, then every 100 up to 1000 → 11 scale factors
SCALE_FACTORS = [1] + list(range(100, 1001, 100))

BASELINE_RUNS = 5
SPARK_WARMUP = 1
SPARK_TIMED = 3

# Spark / Polaris config (from environment, same as kaist_ingestion.config)
CATALOG = "iceberg"
NAMESPACE_SCALE = "kaist_scalability"  # Scratch namespace for scaled copies


# ===================================================================
# Helper: load KAIST JSON files
# ===================================================================
_RAW = {}

def _load(name: str) -> List[Dict]:
    if name not in _RAW:
        path = os.path.join(KAIST_DATA_PATH, name)
        with open(path) as f:
            _RAW[name] = json.load(f)
    return _RAW[name]


# ===================================================================
# 1.  Python Baseline
# ===================================================================
def run_baseline(sf: int) -> Dict:
    """
    Pure-Python object detection query at scale factor *sf*.

    Replicates the exact access pattern of the camera_annotations Gold table:
      camera × frame × clip × calibration × dynamic_object × hdmap
    filtered to a single camera_name.

    Only the fact tables (camera, dynamic_object) are scaled.
    """
    cameras = _load("camera.json") * sf
    dynamic_objects = _load("dynamic_object.json") * sf
    frames = _load("frame.json")          # dimension — 1x
    clips = _load("clip.json")            # dimension — 1x
    calibrations = _load("calibration.json")  # dimension — 1x
    hdmaps = _load("hdmap.json")          # dimension — 1x

    # Index structures (not timed — generous to baseline)
    frame_idx_map = {f["frame_id"]: f.get("frame_idx", 0) for f in frames}
    clip_map = {c["clip_id"]: c for c in clips}
    cal_map = {}
    for cal in calibrations:
        cal_map[(cal["clip_id"], cal["sensor_name"])] = cal
    hdmap_map = {h["clip_id"]: h for h in hdmaps}
    annot_map: Dict[str, list] = {}
    for d in dynamic_objects:
        annot_map.setdefault(d["frame_id"], []).append(d)

    # Pick the first camera_name as filter target
    target_camera = cameras[0]["camera_name"] if cameras else "CAM_FRONT"

    timings = []
    row_count = 0
    for _ in range(BASELINE_RUNS):
        t0 = time.perf_counter()
        results = []
        for cam in cameras:
            if cam["camera_name"] != target_camera:
                continue
            fid = cam["frame_id"]
            cid = cam["clip_id"]

            frame_idx = frame_idx_map.get(fid, 0)
            clip = clip_map.get(cid, {})
            cal = cal_map.get((cid, target_camera), {})
            hm = hdmap_map.get(cid, {})
            annots = annot_map.get(fid, [])

            results.append({
                "frame_id": fid,
                "clip_id": cid,
                "session_id": clip.get("session_id"),
                "frame_idx": frame_idx,
                "sensor_timestamp": cam.get("sensor_timestamp"),
                "camera_name": cam["camera_name"],
                "filename": cam["filename"],
                "extrinsics": cal.get("extrinsics"),
                "camera_intrinsics": cal.get("camera_intrinsics"),
                "annotations": annots,
                "city": hm.get("city"),
                "site": hm.get("site"),
                "date": clip.get("date"),
            })
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        row_count = len(results)

    return {
        "elapsed_seconds": round(statistics.median(timings), 6),
        "row_count": row_count,
    }


# ===================================================================
# 2.  Spark session
# ===================================================================
_spark = None


def _get_spark():
    global _spark
    if _spark is not None:
        return _spark

    from kaist_ingestion.config import PipelineConfig, build_spark_session
    config = PipelineConfig()
    _spark = build_spark_session(config, app_name="KAISTScalabilityBenchmark")
    return _spark


def _scale_df(spark, df, n):
    """Replicate a DataFrame n times via crossJoin."""
    if n <= 1:
        return df
    return df.crossJoin(spark.range(n)).drop("id")


def _timed_query(spark, sql: str, warmup: int = SPARK_WARMUP,
                 runs: int = SPARK_TIMED) -> Dict:
    """Run SQL with warmup + timed runs; return median."""
    timings = []
    row_count = 0
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        row_count = spark.sql(sql).count()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            timings.append(elapsed)
    return {
        "elapsed_seconds": round(statistics.median(timings), 6),
        "row_count": row_count,
    }


# ===================================================================
# 3.  Silver JOIN  (actual KAIST Silver schema, 6-table join)
# ===================================================================

def _setup_silver_dimension_tables(spark) -> str:
    """
    Write KAIST dimension tables (frame, clip, calibration, hdmap) at 1x
    into the scalability namespace.  These mirror the actual kaist_silver
    schema exactly.
    """
    ns = f"{CATALOG}.{NAMESPACE_SCALE}"
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {ns}")

    # Read actual Silver dimension tables
    silver = f"{CATALOG}.kaist_silver"
    for table in ("frame", "clip", "calibration", "hdmap"):
        spark.table(f"{silver}.{table}") \
            .write.format("iceberg").mode("overwrite") \
            .saveAsTable(f"{ns}.{table}")
    return ns


def _ingest_silver_facts(spark, ns: str, sf: int):
    """
    Write scaled KAIST fact tables (camera, dynamic_object) to the
    scalability namespace.  These are copied from the actual kaist_silver
    tables and replicated sf times.
    """
    silver = f"{CATALOG}.kaist_silver"

    # Camera: scale and partition by camera_name (matches real Silver)
    df_cam = spark.table(f"{silver}.camera")
    _scale_df(spark, df_cam, sf) \
        .write.format("iceberg").partitionBy("camera_name") \
        .mode("overwrite").saveAsTable(f"{ns}.camera")

    # Dynamic object: scale (matches real Silver)
    df_do = spark.table(f"{silver}.dynamic_object")
    _scale_df(spark, df_do, sf) \
        .write.format("iceberg").mode("overwrite") \
        .saveAsTable(f"{ns}.dynamic_object")


def _query_silver(spark, ns: str, target_camera: str) -> Dict:
    """
    Run the exact same 6-table join used by build_gold.build_camera_annotations,
    filtered to a single camera_name.
    """
    sql = f"""
        SELECT
            cam.frame_id, cam.clip_id, cl.session_id, f.frame_idx,
            cam.sensor_timestamp, cam.camera_name, cam.filename,
            cal.extrinsics, cal.camera_intrinsics,
            annot.annotations,
            hm.city, hm.site, cl.date
        FROM {ns}.camera cam
        LEFT JOIN {ns}.frame f ON cam.frame_id = f.frame_id
        LEFT JOIN {ns}.clip cl ON cam.clip_id = cl.clip_id
        LEFT JOIN {ns}.calibration cal
            ON cam.clip_id = cal.clip_id
            AND cam.camera_name = cal.sensor_name
        LEFT JOIN (
            SELECT frame_id,
                   collect_list(struct(boxes_3d, category)) AS annotations
            FROM {ns}.dynamic_object
            GROUP BY frame_id
        ) annot ON cam.frame_id = annot.frame_id
        LEFT JOIN {ns}.hdmap hm ON cam.clip_id = hm.clip_id
        WHERE cam.camera_name = '{target_camera}'
    """
    return _timed_query(spark, sql)


# ===================================================================
# 4.  Gold (actual KAIST Gold schema — camera_annotations)
# ===================================================================

def _ingest_gold(spark, ns: str, sf: int, target_camera: str):
    """
    Build a scaled copy of the camera_annotations Gold table.
    Reads from the actual kaist_gold.camera_annotations, replicates sf times,
    writes partitioned by camera_name — identical structure to the real table.
    """
    gold = f"{CATALOG}.kaist_gold"
    df = spark.table(f"{gold}.camera_annotations")
    _scale_df(spark, df, sf) \
        .write.format("iceberg").partitionBy("camera_name") \
        .mode("overwrite").saveAsTable(f"{ns}.gold_camera_annotations")


def _query_gold(spark, ns: str, target_camera: str) -> Dict:
    """
    Single-table partitioned read — mirrors the actual Gold table access.
    """
    sql = f"""
        SELECT *
        FROM {ns}.gold_camera_annotations
        WHERE camera_name = '{target_camera}'
    """
    return _timed_query(spark, sql)


# ===================================================================
# Main sweep
# ===================================================================
def main():
    t_start = time.perf_counter()
    results = []

    # Determine target camera from actual data
    cameras = _load("camera.json")
    target_camera = cameras[0]["camera_name"] if cameras else "CAM_FRONT"
    print(f"Target camera for all queries: {target_camera}")
    print(f"Scale factors: {SCALE_FACTORS}")

    # --- Phase A: Python Baseline -------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE A  |  Python Baseline (KAIST JSON)  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Baseline] SF={sf:>2d} ... ", end="", flush=True)
        r = run_baseline(sf)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Python Baseline", "scale_factor": sf, **r})

    # --- Spark init & dimension tables --------------------------------------
    spark = _get_spark()
    ns = _setup_silver_dimension_tables(spark)
    print(f"\nDimension tables written to {ns}")

    # --- Phase B: Silver JOIN ------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE B  |  Silver JOIN (actual KAIST schema)  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Silver] SF={sf:>2d}  ingest ... ", end="", flush=True)
        _ingest_silver_facts(spark, ns, sf)
        print("query ... ", end="", flush=True)
        r = _query_silver(spark, ns, target_camera)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Silver JOIN", "scale_factor": sf, **r})

    # --- Phase C: Gold -------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE C  |  Gold (actual camera_annotations)  |  SF 1 - 50")
    print("=" * 65)
    for sf in SCALE_FACTORS:
        print(f"  [Gold]   SF={sf:>2d}  ingest ... ", end="", flush=True)
        _ingest_gold(spark, ns, sf, target_camera)
        print("query ... ", end="", flush=True)
        r = _query_gold(spark, ns, target_camera)
        print(f"{r['elapsed_seconds']:.4f} s  ({r['row_count']:,} rows)")
        results.append({"strategy": "Gold", "scale_factor": sf, **r})

    spark.stop()

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal benchmark time: {elapsed_total:.1f} s")

    # --- Write JSON ----------------------------------------------------------
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results  -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
