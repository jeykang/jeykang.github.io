"""
Gold-tier edge-case scoring for Nvidia PhysicalAI dataset.

Under the redefined medallion architecture:
  Bronze = full dataset ingested as-is
  Silver = Bronze minus broken/unusable data
  Gold   = curated edge-case subset (hardest clips for an AV model)

Composite difficulty score has two components:

  1. Scene Complexity Score (CPU, no GPU needed)
     Derived from metadata: time of day, season, country/geography,
     sensor coverage, ego dynamics (acceleration variance, maneuver
     complexity), and obstacle density (when v26.03 labels available).

  2. Perception Difficulty Score (GPU, BEVFusion or similar)
     Run AV model inference per clip, measure:
     - Temporal consistency of detections (jittery = harder)
     - Detection count/density (more objects = more complex)
     - Ego-perception correlation (hard braking + low confidence = difficult)

The two scores are combined: final = w1 * scene + w2 * perception.
Scene-only scoring works without GPU and is the default starting point.

Model backends are pluggable via the ModelBackend interface:
  - DummyBackend: deterministic hash-based scores (testing)
  - MetadataBackend: scene complexity from metadata only (no GPU)
  - BEVFusionBackend: placeholder for multi-modal fusion (~12GB VRAM)

GPU allocation:
  - RTX 6000 (24GB, Turing): one evaluation pipeline
  - A10 (23GB, Ampere): one evaluation pipeline

Usage:
    # Metadata-only scoring (no GPU, works now)
    python -m nvidia_ingestion.edge_case_scorer --backend metadata --top-pct 10

    # With perception model (when available)
    python -m nvidia_ingestion.edge_case_scorer --backend bevfusion --gpu 0 --top-pct 10

    # Pipeline testing
    python -m nvidia_ingestion.edge_case_scorer --backend dummy --top-pct 10
"""

import abc
import csv
import hashlib
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from .config import NvidiaPipelineConfig, build_spark_session, create_namespaces

log = logging.getLogger("nvidia.gold_scorer")


# ---------------------------------------------------------------------------
# Score schema
# ---------------------------------------------------------------------------

SCORE_SCHEMA = StructType([
    StructField("clip_id", StringType(), False),
    StructField("difficulty_score", DoubleType(), False),   # primary = difficulty_camera (this consumer)
    StructField("difficulty_camera", DoubleType(), True),   # camera-only perceptual axis
    StructField("difficulty_lidar", DoubleType(), True),    # lidar-fused perceptual axis (general)
    StructField("scene_score", DoubleType(), True),         # scene complexity component
    StructField("perception_score", DoubleType(), True),    # perception difficulty component
    StructField("backend", StringType(), False),
    StructField("detail", StringType(), True),              # JSON blob with sub-scores
    StructField("scored_at", StringType(), False),
    StructField("sensor_covered", BooleanType(), True),    # has a sensor-based
    # signal (conflict/perception) — i.e. is in the on-disk 10TB sample. The
    # difficulty tier is scoped to these; catalog-only clips score on weak
    # metadata alone (battery 2026-06-23) and are excluded from Gold.
])


# ---------------------------------------------------------------------------
# Model backend interface
# ---------------------------------------------------------------------------

class ModelBackend(abc.ABC):
    """Interface for AV model evaluation backends."""

    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this backend."""

    @abc.abstractmethod
    def setup(self, gpu_id: int = 0) -> None:
        """Load model weights and initialize on the given GPU."""

    @abc.abstractmethod
    def score_clip(
        self,
        clip_id: str,
        lidar_paths: List[str],
        ego_trajectory: List[dict],
        camera_paths: Optional[Dict[str, List[str]]] = None,
        radar_data: Optional[List[dict]] = None,
    ) -> Tuple[float, str]:
        """Score a single clip's perception difficulty.

        Returns:
            (perception_score, detail_json)
            score is 0.0 (trivial) to 1.0 (extremely difficult)
        """

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release GPU resources."""


# ---------------------------------------------------------------------------
# Scene complexity scoring (CPU-only, metadata-based)
# ---------------------------------------------------------------------------

# DEPRECATED (2026-06-23): the weighted-average composite is superseded by the
# noisy-OR axis union in `compute_scene_score` (behavioral=conflict ∪ perceptual=
# darkness/low-conf), after the validity battery refuted the metadata dims and the
# edge-case-mining goal called for a union, not an average
# (VALIDITY_BATTERY_FINDINGS.md). Kept only for reference; not read anywhere.
_SCENE_WEIGHTS = {
    "time_of_day": 0.15,       # weak environmental nudge (battery AUC ~0.51)
    "season_geography": 0.0,   # DROPPED: near-constant (92% one value), no signal
    "sensor_coverage": 0.0,    # DROPPED: miscalibrated (47% maxed), anti-aligned
    "ego_dynamics": 0.0,       # DROPPED: 89% default, AUC 0.500 (chance)
    "obstacle_density": 0.20,  # more actors = harder (when available)
    "perception": 0.25,        # BEVFusion damper — validated secondary (modest)
    "planning": 0.15,          # detached driving-difficulty module (open-loop);
                               # refuted by the battery, kept only as a hook
    "conflict": 0.60,          # agent-interaction difficulty (obstacle.offline);
                               # the one validated facet (OOD AUC ~0.65) — primary
}

# Time-of-day difficulty (hour_of_day from data_collection)
_HOUR_DIFFICULTY = {
    # Night (0-5): hardest
    **{h: 1.0 for h in range(0, 6)},
    # Dawn/dusk (6-7, 18-20): hard
    **{h: 0.7 for h in [6, 7, 18, 19, 20]},
    # Morning rush (8-9): medium-high
    **{h: 0.5 for h in [8, 9]},
    # Evening rush (16-17): medium
    **{h: 0.4 for h in [16, 17]},
    # Midday (10-15): easiest
    **{h: 0.2 for h in range(10, 16)},
    # Late night (21-23): hard
    **{h: 0.8 for h in [21, 22, 23]},
}

# Season difficulty multiplier per latitude band
# Northern European countries in winter are hardest
_NORTHERN_COUNTRIES = {
    "Finland", "Sweden", "Denmark", "Estonia", "Latvia", "Lithuania",
    "Poland", "Germany", "Netherlands", "Belgium", "Luxembourg",
}
_SEASON_DIFFICULTY = {
    "winter": 0.9,
    "fall": 0.5,
    "spring": 0.4,
    "summer": 0.2,
}


def _score_time_of_day(hour: int) -> float:
    """Score 0-1 based on hour_of_day."""
    return _HOUR_DIFFICULTY.get(hour, 0.3)


def _score_season_geography(season: str, country: str) -> float:
    """Score 0-1 based on season and geography."""
    base = _SEASON_DIFFICULTY.get(season, 0.3)
    # Northern countries in winter get a boost
    if country in _NORTHERN_COUNTRIES and season == "winter":
        return min(1.0, base + 0.1)
    return base


def _score_sensor_coverage(sensor_flags: dict) -> float:
    """Score 0-1 based on sensor availability (fewer = harder).

    sensor_flags: dict of sensor_name -> bool from sensor_presence
    """
    radar_sensors = [k for k, v in sensor_flags.items()
                     if k.startswith("radar_") and k != "radar_config"]
    n_radars_present = sum(1 for k in radar_sensors if sensor_flags.get(k, False))
    total_radars = len(radar_sensors)
    if total_radars == 0:
        return 0.5

    # Full radar suite = easy (low score), partial = harder
    coverage_ratio = n_radars_present / total_radars
    # Invert: low coverage = high difficulty
    return 1.0 - coverage_ratio


def _score_ego_dynamics(ego_data: List[dict]) -> float:
    """Score 0-1 based on ego motion dynamics.

    High acceleration variance, speed changes, and angular velocity
    indicate complex maneuvers (turns, lane changes, emergency stops).
    """
    if not ego_data or len(ego_data) < 3:
        return 0.5  # neutral if no data

    # Extract positions over time
    positions = []
    for e in ego_data:
        x = e.get("x", 0.0)
        y = e.get("y", 0.0)
        t = e.get("timestamp", 0.0)
        if t is not None and x is not None and y is not None:
            positions.append((float(t), float(x), float(y)))

    if len(positions) < 3:
        return 0.5

    positions.sort(key=lambda p: p[0])

    # Compute velocities
    velocities = []
    for i in range(1, len(positions)):
        dt = positions[i][0] - positions[i-1][0]
        if dt <= 0:
            continue
        dx = positions[i][1] - positions[i-1][1]
        dy = positions[i][2] - positions[i-1][2]
        speed = math.sqrt(dx*dx + dy*dy) / dt
        velocities.append(speed)

    if len(velocities) < 2:
        return 0.5

    # Compute accelerations
    accels = []
    for i in range(1, len(velocities)):
        dt = positions[i+1][0] - positions[i][0]
        if dt <= 0:
            continue
        accels.append(abs(velocities[i] - velocities[i-1]) / dt)

    if not accels:
        return 0.5

    # Score based on acceleration variance (high variance = complex driving)
    mean_accel = sum(accels) / len(accels)
    var_accel = sum((a - mean_accel) ** 2 for a in accels) / len(accels)
    std_accel = math.sqrt(var_accel)

    # Also compute heading changes (lateral complexity)
    heading_changes = []
    for i in range(1, len(positions)):
        dx = positions[i][1] - positions[i-1][1]
        dy = positions[i][2] - positions[i-1][2]
        heading_changes.append(math.atan2(dy, dx))
    angular_diffs = []
    for i in range(1, len(heading_changes)):
        diff = abs(heading_changes[i] - heading_changes[i-1])
        if diff > math.pi:
            diff = 2 * math.pi - diff
        angular_diffs.append(diff)

    mean_angular = sum(angular_diffs) / len(angular_diffs) if angular_diffs else 0

    # Normalize: typical straight highway has std_accel ~0.5, angular ~0.01
    # Complex urban: std_accel ~3.0, angular ~0.1
    accel_score = min(1.0, std_accel / 3.0)
    angular_score = min(1.0, mean_angular / 0.1)

    return 0.6 * accel_score + 0.4 * angular_score


def compute_scene_score(
    hour: int,
    season: str,
    country: str,
    sensor_flags: dict,
    ego_data: Optional[List[dict]] = None,
    obstacle_count: Optional[int] = None,
    ego_score: Optional[float] = None,
    perception_score: Optional[float] = None,
    planning_score: Optional[float] = None,
    conflict_score: Optional[float] = None,
    low_conf_score: Optional[float] = None,
    camera_low_conf_score: Optional[float] = None,
) -> Tuple[float, dict]:
    """Compute composite difficulty for edge-case mining.

    Returns (score, sub_scores_dict).

    The composite is a **noisy-OR union of two validated hardness axes** (not a
    weighted average), because the goal is to KEEP a clip if it is hard on EITHER
    axis and strip only the trivially-easy (VALIDITY_BATTERY_FINDINGS.md):
      behavioral = agent-interaction difficulty  -> `conflict_score`
      perceptual = perception-degradation difficulty -> darkness (time_of_day)
                   reinforced by low detection confidence (`low_conf_score`).
      difficulty = 1 - (1 - behavioral) * (1 - perceptual)

    `ego_score`/`perception_score`/`planning_score` are kept in `sub` for
    diagnostics but no longer drive the composite (battery-refuted / superseded).
    `low_conf_score` ∈ [0,1] is 1 − mean_max_conf from the BEVFusion stats.
    """
    sub = {}
    sub["time_of_day"] = _score_time_of_day(hour)
    sub["season_geography"] = _score_season_geography(season, country)
    sub["sensor_coverage"] = _score_sensor_coverage(sensor_flags)
    if ego_score is not None:
        sub["ego_dynamics"] = ego_score
    else:
        sub["ego_dynamics"] = _score_ego_dynamics(ego_data or [])

    # Optional sub-scores — computed into `sub` for diagnostics. season_geography
    # / sensor_coverage / ego_dynamics / obstacle_density / perception / planning
    # are kept for the record but do NOT drive the composite (battery-refuted or
    # superseded by the axis union below).
    def _clamp(v):
        return float(max(0.0, min(1.0, v))) if v is not None else None
    sub["obstacle_density"] = (min(1.0, obstacle_count / 20.0)
                               if obstacle_count is not None else None)
    sub["perception"] = _clamp(perception_score)
    sub["planning"] = _clamp(planning_score)
    sub["conflict"] = _clamp(conflict_score)
    sub["low_conf"] = _clamp(low_conf_score)              # lidar-fused
    sub["camera_low_conf"] = _clamp(camera_low_conf_score)  # camera-only

    # --- Edge-case mining: difficulty = noisy-OR union of validated axes ---
    # Keep a clip if it is hard on EITHER axis (a weighted average would dilute a
    # clip hard on only one axis and wrongly strip it). See VALIDITY_BATTERY_FINDINGS.md.
    #   behavioral  = agent-interaction difficulty  -> conflict (OOD AUC ~0.65)
    #   perceptual  = perception-degradation diff.   -> darkness (time_of_day;
    #                 face-valid + empirically -10% conf / -24% detections in the
    #                 dark), reinforced by low detection confidence where available.
    behavioral = sub["conflict"] if sub["conflict"] is not None else 0.0
    darkness = sub["time_of_day"]
    # Two perceptual axes (each = darkness reinforced by its modality's low-conf):
    #   lidar  -> general-purpose, lidar-fused stack (BEVFusion)
    #   camera -> the consumer's camera-only final product
    perc_lidar = max(darkness, sub["low_conf"]) if sub["low_conf"] is not None else darkness
    perc_camera = max(darkness, sub["camera_low_conf"]) if sub["camera_low_conf"] is not None else darkness
    sub["behavioral_axis"] = behavioral
    sub["perceptual_lidar_raw"] = perc_lidar
    sub["perceptual_camera_raw"] = perc_camera
    # Returned score = camera-based (the active consumer); the main loop computes
    # both difficulties from the rank-normalized axes.
    sub["perceptual_axis"] = perc_camera
    score = 1.0 - (1.0 - behavioral) * (1.0 - perc_camera)
    return score, sub


# ---------------------------------------------------------------------------
# Dummy backend (pipeline testing)
# ---------------------------------------------------------------------------

class DummyBackend(ModelBackend):
    """Deterministic hash-based scores for testing the pipeline end-to-end."""

    def name(self) -> str:
        return "dummy"

    def setup(self, gpu_id: int = 0) -> None:
        log.info("DummyBackend: no GPU needed")

    def score_clip(self, clip_id, lidar_paths, ego_trajectory,
                   camera_paths=None, radar_data=None):
        h = int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16)
        score = (h % 1000) / 1000.0
        detail = json.dumps({
            "method": "hash-based-dummy",
            "n_lidar_files": len(lidar_paths),
            "n_ego_samples": len(ego_trajectory),
        })
        return score, detail

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Metadata-only backend (scene complexity, no GPU)
# ---------------------------------------------------------------------------

class MetadataBackend(ModelBackend):
    """Scene-complexity-only scoring using dataset metadata.

    No GPU needed. Uses data_collection (country, hour, month),
    sensor_presence (radar coverage), selected_chunks (season),
    and egomotion (dynamics) to score clip difficulty.

    This is the default starting point. Perception scoring can be
    layered on top later with a GPU-backed backend.
    """

    def __init__(self, chunk_metadata: Optional[Dict[int, dict]] = None):
        self._chunk_meta = chunk_metadata or {}

    def name(self) -> str:
        return "metadata"

    def setup(self, gpu_id: int = 0) -> None:
        log.info("MetadataBackend: CPU-only, no GPU needed")

    def score_clip(self, clip_id, lidar_paths, ego_trajectory,
                   camera_paths=None, radar_data=None):
        # This method is not used directly — MetadataBackend uses
        # score_clip_with_metadata() instead, called by the orchestrator.
        return 0.5, json.dumps({"method": "metadata-fallback"})

    def score_clip_with_metadata(
        self,
        clip_id: str,
        hour: int,
        season: str,
        country: str,
        sensor_flags: dict,
        ego_data: Optional[List[dict]] = None,
        obstacle_count: Optional[int] = None,
    ) -> Tuple[float, float, str]:
        """Score using metadata. Returns (final_score, scene_score, detail_json)."""
        scene_score, sub_scores = compute_scene_score(
            hour, season, country, sensor_flags, ego_data, obstacle_count
        )
        detail = json.dumps({
            "method": "metadata-scene-complexity",
            "sub_scores": {k: (round(v, 4) if v is not None else None)
                           for k, v in sub_scores.items()},
            "hour": hour,
            "season": season,
            "country": country,
            "n_ego_samples": len(ego_data) if ego_data else 0,
        })
        return scene_score, scene_score, detail

    def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BEVFusion backend (placeholder)
# ---------------------------------------------------------------------------

class BEVFusionBackend(ModelBackend):
    """BEVFusion multi-modal perception for difficulty scoring.

    Scoring approach (open-loop, no ground truth needed):
      1. Run BEVFusion on each lidar+camera frame
      2. Measure temporal consistency of detections across frames
      3. Score based on: detection jitter, count variance, confidence drops
      4. Combine with scene complexity for final score

    Requires:
      - mmdetection3d with BEVFusion checkpoint
      - ~12GB VRAM (fits RTX 6000 24GB or A10 23GB)
    """

    def __init__(self, config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 scene_weight: float = 0.4,
                 perception_weight: float = 0.6):
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._scene_w = scene_weight
        self._percep_w = perception_weight

    def name(self) -> str:
        return "bevfusion"

    def setup(self, gpu_id: int = 0) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            device = f"cuda:{gpu_id}"
            log.info("BEVFusionBackend: loading model on %s", device)

            if self._config_path and self._checkpoint_path:
                from mmdet3d.apis import init_model
                self._model = init_model(
                    self._config_path,
                    self._checkpoint_path,
                    device=device,
                )
                log.info("BEVFusionBackend: model loaded (~12GB VRAM)")
            else:
                raise FileNotFoundError(
                    "BEVFusion config/checkpoint paths not provided. "
                    "Download from mmdetection3d model zoo or "
                    "NVIDIA's model registry."
                )
        except ImportError as e:
            raise ImportError(
                f"BEVFusion requires mmdetection3d: {e}. "
                "Install with: pip install mmdet3d"
            ) from e

    def score_clip(self, clip_id, lidar_paths, ego_trajectory,
                   camera_paths=None, radar_data=None):
        if self._model is None:
            raise RuntimeError("Model not loaded — call setup() first")

        # Placeholder for actual BEVFusion inference:
        # 1. Load lidar point clouds from parquet (Draco-decode)
        # 2. Load camera frames from mp4
        # 3. Run BEVFusion forward pass per frame
        # 4. Measure detection temporal consistency:
        #    - Track detections across frames
        #    - High ID switches / position jitter = hard scene
        # 5. Measure confidence drops:
        #    - Frames where avg confidence < threshold = hard
        # 6. Detection density variance:
        #    - Sudden changes in object count = dynamic scene

        # For now, proxy scoring
        n_frames = len(lidar_paths)
        perception_score = 0.5  # placeholder

        detail = json.dumps({
            "method": "bevfusion-placeholder",
            "n_frames": n_frames,
            "perception_score": perception_score,
            "note": "Actual inference not yet implemented",
        })
        return perception_score, detail

    def teardown(self) -> None:
        if self._model is not None:
            del self._model
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            self._model = None


# ---------------------------------------------------------------------------
# Scoring orchestrator
# ---------------------------------------------------------------------------

BACKENDS = {
    "dummy": DummyBackend,
    "metadata": MetadataBackend,
    "bevfusion": BEVFusionBackend,
}


def _load_chunk_seasons(source_path: str) -> Dict[int, str]:
    """Load season per chunk from selected_chunks.csv."""
    csv_path = os.path.join(source_path, "selected_chunks.csv")
    chunk_season = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                chunk_season[int(row["chunk"])] = row["season"]
    return chunk_season


def _load_perception_scores(spark, source_path: str) -> Dict[str, float]:
    """Load per-clip perception scores written by the perception scorer
    (BEVFusion runner, or the legacy camera_perception_scorer).

    Each shard writes `<source>/.perception/*.parquet` with columns
    (clip_id, perception_score, ...). Reads them via Spark and merges into a
    `{clip_id: perception_score}` dict. Returns empty dict if none exist
    (perception dimension is then dropped by compute_scene_score).

    Uses Spark rather than pyarrow on purpose: the spark-submit driver Python
    does not bundle pyarrow, so a pyarrow import here silently failed and the
    perception dimension was never applied.
    """
    perc_dir = os.path.join(source_path, ".perception")
    out: Dict[str, float] = {}
    if not os.path.isdir(perc_dir):
        return out
    paths = [f"file://{os.path.join(perc_dir, f)}"
             for f in sorted(os.listdir(perc_dir)) if f.endswith(".parquet")]
    if not paths:
        return out
    try:
        df = spark.read.parquet(*paths).select("clip_id", "perception_score")
        for r in df.collect():
            if r["perception_score"] is not None:
                out[r["clip_id"]] = float(r["perception_score"])
    except Exception as e:
        log.warning("Failed to load perception scores: %s", e)
    return out


def _load_low_conf_scores(spark, source_path: str) -> Dict[str, float]:
    """LIDAR-FUSED perception-degradation signal (1 − mean_max_conf from the
    BEVFusion stats, `<source>/.perception/*.parquet`). High = the fused detector
    is uncertain = perceptually hard for a lidar-included stack. Feeds the
    `difficulty_lidar` axis (the general-purpose score). Empty dict if none.
    Pairs with `_load_camera_low_conf_scores` (the camera-only axis) — the scorer
    emits BOTH difficulties so Gold serves both lidar-fused and camera-only
    consumers (see cosmos_augmentation/FINDINGS.md).
    """
    out: Dict[str, float] = {}
    perc_dir = os.path.join(source_path, ".perception")
    if not os.path.isdir(perc_dir):
        return out
    paths = [f"file://{os.path.join(perc_dir, f)}"
             for f in sorted(os.listdir(perc_dir)) if f.endswith(".parquet")]
    if not paths:
        return out
    try:
        df = spark.read.parquet(*paths).select("clip_id", "mean_max_conf")
        for r in df.collect():
            if r["mean_max_conf"] is not None:
                out[r["clip_id"]] = float(max(0.0, min(1.0, 1.0 - r["mean_max_conf"])))
    except Exception as e:
        log.warning("Failed to load low_conf scores: %s", e)
    return out


def _load_camera_low_conf_scores(spark, source_path: str) -> Dict[str, float]:
    """CAMERA-ONLY agent-gated perception-degradation signal (`<source>/
    .camera_perception/camera_gated.parquet`, 1 − front-cam YOLO max-conf, gated
    to scenes with agents present). Feeds the `difficulty_camera` axis — matches
    the consumer's camera-only final product, which the lidar-fused signal is
    blind to (lidar masks camera degradation). Empty dict if absent."""
    out: Dict[str, float] = {}
    cam_gated = os.path.join(source_path, ".camera_perception", "camera_gated.parquet")
    if not os.path.isfile(cam_gated):
        return out
    try:
        df = spark.read.parquet(f"file://{cam_gated}").select("clip_id", "low_conf")
        for r in df.collect():
            if r["low_conf"] is not None:
                out[r["clip_id"]] = float(max(0.0, min(1.0, r["low_conf"])))
    except Exception as e:
        log.warning("Failed to load camera_gated scores: %s", e)
    return out


def _load_planning_scores(spark, source_path: str) -> Dict[str, float]:
    """Load per-clip driving-difficulty scores from the planning/ module.

    Reads `<source>/.planning/*.parquet` (clip_id, planning_score) via Spark
    and returns `{clip_id: planning_score}`. Empty dict if the module hasn't
    run — the `planning` dimension is then dropped by compute_scene_score, so
    the module is fully detachable.
    """
    plan_dir = os.path.join(source_path, ".planning")
    out: Dict[str, float] = {}
    if not os.path.isdir(plan_dir):
        return out
    paths = [f"file://{os.path.join(plan_dir, f)}"
             for f in sorted(os.listdir(plan_dir)) if f.endswith(".parquet")]
    if not paths:
        return out
    try:
        df = spark.read.parquet(*paths).select("clip_id", "planning_score")
        for r in df.collect():
            if r["planning_score"] is not None:
                out[r["clip_id"]] = float(r["planning_score"])
    except Exception as e:
        log.warning("Failed to load planning scores: %s", e)
    return out


def _load_conflict_scores(spark, source_path: str) -> Dict[str, float]:
    """Load per-clip agent-interaction (conflict) scores (.conflict/ file-drop).

    Reads `<source>/.conflict/*.parquet` (clip_id, conflict_score) via Spark.
    Empty dict if the conflict runner hasn't run — the `conflict` dimension is
    then dropped by compute_scene_score, so the sub-score is fully detachable.
    """
    cdir = os.path.join(source_path, ".conflict")
    out: Dict[str, float] = {}
    if not os.path.isdir(cdir):
        return out
    paths = [f"file://{os.path.join(cdir, f)}"
             for f in sorted(os.listdir(cdir)) if f.endswith(".parquet")]
    if not paths:
        return out
    try:
        df = spark.read.parquet(*paths).select("clip_id", "conflict_score")
        for r in df.collect():
            if r["conflict_score"] is not None:
                out[r["clip_id"]] = float(r["conflict_score"])
    except Exception as e:
        log.warning("Failed to load conflict scores: %s", e)
    return out


class EdgeCaseScorer:
    """Orchestrates clip scoring and Gold subset selection."""

    def __init__(self, spark: SparkSession, config: NvidiaPipelineConfig,
                 backend: ModelBackend):
        self.spark = spark
        self.config = config
        self.backend = backend
        self.cat = config.spark_catalog_name
        self.ns_b = config.nvidia.namespace_bronze
        self.ns_s = config.nvidia.namespace_silver
        self.ns_g = config.nvidia.namespace_gold

    def _bronze(self, t: str) -> str:
        return f"{self.cat}.{self.ns_b}.{t}"

    def _silver(self, t: str) -> str:
        return f"{self.cat}.{self.ns_s}.{t}"

    def _gold(self, t: str) -> str:
        return f"{self.cat}.{self.ns_g}.{t}"

    def score_all_clips(self, limit: int = 0) -> DataFrame:
        """Score all clips using the composite approach.

        For MetadataBackend: uses Spark to join metadata tables and score
        in bulk (much faster than per-clip iteration).

        For GPU backends: iterates clips and runs model inference.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        if isinstance(self.backend, MetadataBackend):
            return self._score_metadata_bulk(limit, ts)
        else:
            return self._score_per_clip(limit, ts)

    def _score_metadata_bulk(self, limit: int, ts: str) -> DataFrame:
        """Efficient bulk scoring using Spark SQL on metadata tables."""
        print(f"[GOLD SCORER] Bulk metadata scoring via Spark")

        # Load chunk -> season mapping
        chunk_seasons = _load_chunk_seasons(self.config.nvidia.source_path)

        # Canonical schema: Clip is the canonical clip metadata; aux_*
        # tables hold Nvidia-specific fields not in the canonical schema
        # (hour_of_day, month, platform_class, radar_config, sensor flags).
        # `chunk` is no longer in canonical Clip — read it from aux_data_collection
        # join through `selected_chunks.csv` (same chunk for all clips in a chunk
        # dir → use clip_id↔chunk via a lookup we'll do downstream).
        try:
            clips = self.spark.table(self._silver("Clip")).alias("ci")
        except Exception:
            clips = self.spark.table(self._bronze("Clip")).alias("ci")
        data_coll = self.spark.table(self._bronze("aux_data_collection")).alias("dc")
        sensors = self.spark.table(self._bronze("aux_sensor_presence")).alias("sp")
        # Inject chunk by parsing it back from session_id ('session_chunk_NNNN')
        clips = clips.withColumn(
            "chunk",
            F.regexp_extract(F.col("session_id"), r"session_chunk_(\d+)", 1).cast("int"),
        )

        # Join on clip_id
        joined = (
            clips
            .join(data_coll, "clip_id", "left")
            .join(sensors, "clip_id", "left")
        )

        if limit > 0:
            joined = joined.limit(limit)

        # Collect and score (metadata scoring is fast, ~1ms per clip)
        rows = joined.collect()
        total = len(rows)
        print(f"  Scoring {total:,} clips...")

        # Get radar sensor column names for coverage scoring
        radar_cols = [c for c in sensors.columns
                      if c.startswith("radar_") and c != "radar_config"]

        # Bulk-aggregate per-clip ego dynamics from the aux_egomotion table
        # (single pass over ~101M rows instead of per-clip reads). aux_egomotion
        # is a zero-copy re-registration of the original egomotion source — has
        # ax/ay/az/curvature columns that canonical EgoMotion dropped. clip_id
        # is missing from the source (filename-encoded), so derive it via
        # input_file_name() before grouping.
        ego_scores: dict = {}
        try:
            ego_tbl = self.spark.table(self._bronze("aux_egomotion"))
            UUID_RE = (
                r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."
            )
            if "clip_id" not in ego_tbl.columns:
                ego_tbl = ego_tbl.withColumn(
                    "clip_id", F.regexp_extract(F.input_file_name(), UUID_RE, 1)
                ).filter(F.col("clip_id") != "")
            ego_cols = set(ego_tbl.columns)
            if {"ax", "ay", "az", "curvature", "clip_id"} <= ego_cols:
                print(f"  Computing bulk ego_dynamics aggregates...")
                t_ego = time.time()
                accel_mag = F.sqrt(
                    F.col("ax") * F.col("ax")
                    + F.col("ay") * F.col("ay")
                    + F.col("az") * F.col("az")
                )
                ego_agg = (
                    ego_tbl
                    .groupBy("clip_id")
                    .agg(
                        F.stddev(accel_mag).alias("accel_std"),
                        F.stddev(F.col("curvature")).alias("curv_std"),
                    )
                    .collect()
                )
                for r in ego_agg:
                    a = r["accel_std"] or 0.0
                    c = r["curv_std"] or 0.0
                    accel_s = min(1.0, float(a) / 3.0)
                    angular_s = min(1.0, float(c) / 0.1)
                    ego_scores[r["clip_id"]] = 0.6 * accel_s + 0.4 * angular_s
                print(f"  Ego aggregates for {len(ego_scores):,} clips "
                      f"in {time.time() - t_ego:.1f}s")
        except Exception as e:
            log.warning("Bulk ego aggregation skipped: %s", e)

        # Load externally-computed perception scores if the YOLO-based camera
        # scorer has already run. Each shard writes a parquet with per-clip
        # `perception_score` under <source>/.perception/.
        perception_scores: dict = _load_perception_scores(
            self.spark, self.config.nvidia.source_path
        )
        if perception_scores:
            print(f"  Loaded perception scores for {len(perception_scores):,} clips")
        else:
            print("  No perception scores found; scoring will skip that dimension")

        planning_scores: dict = _load_planning_scores(
            self.spark, self.config.nvidia.source_path
        )
        if planning_scores:
            print(f"  Loaded planning scores for {len(planning_scores):,} clips")
        else:
            print("  No planning scores found; scoring will skip that dimension")

        conflict_scores: dict = _load_conflict_scores(
            self.spark, self.config.nvidia.source_path
        )
        if conflict_scores:
            print(f"  Loaded conflict scores for {len(conflict_scores):,} clips")
        else:
            print("  No conflict scores found; scoring will skip that dimension")

        low_conf_scores: dict = _load_low_conf_scores(
            self.spark, self.config.nvidia.source_path
        )
        if low_conf_scores:
            print(f"  Loaded LIDAR-fused low-conf scores for "
                  f"{len(low_conf_scores):,} clips")
        camera_low_conf_scores: dict = _load_camera_low_conf_scores(
            self.spark, self.config.nvidia.source_path
        )
        if camera_low_conf_scores:
            print(f"  Loaded CAMERA-only low-conf scores for "
                  f"{len(camera_low_conf_scores):,} clips")

        # First pass: compute per-clip axes (behavioral=conflict, perceptual=raw).
        recs = []
        t0 = time.time()
        for i, row in enumerate(rows):
            d = row.asDict()
            clip_id = d["clip_id"]
            chunk = d.get("chunk") or 0
            hour = d.get("hour_of_day") or 12
            country = d.get("country") or "Unknown"
            season = chunk_seasons.get(chunk, "summer")
            sensor_flags = {rc: bool(d.get(rc)) for rc in radar_cols}

            _, sub = compute_scene_score(
                hour, season, country, sensor_flags,
                ego_data=None, obstacle_count=None,
                ego_score=ego_scores.get(clip_id),
                perception_score=perception_scores.get(clip_id),
                planning_score=planning_scores.get(clip_id),
                conflict_score=conflict_scores.get(clip_id),
                low_conf_score=low_conf_scores.get(clip_id),
                camera_low_conf_score=camera_low_conf_scores.get(clip_id),
            )
            recs.append({
                "clip_id": clip_id, "hour": hour, "season": season,
                "country": country, "chunk": chunk, "sub": sub,
                "behavioral": sub["behavioral_axis"],
                "perceptual_lidar_raw": sub["perceptual_lidar_raw"],
                "perceptual_camera_raw": sub["perceptual_camera_raw"],
                "covered": (conflict_scores.get(clip_id) is not None
                            or perception_scores.get(clip_id) is not None),
            })
            if (i + 1) % 50000 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{total}] {el:.0f}s, {(i+1)/el:.0f} clips/s")

        # Rank-normalize the perceptual axis over the covered population so it is
        # on the same uniform [0,1] scale as conflict (already rank-normalized by
        # the conflict runner). Otherwise darkness=1.0 for night saturates the
        # noisy-OR into a veto and crowds out validated daytime behavioral cases.
        # Rank-normalize BOTH perceptual axes (lidar-fused + camera-only) over the
        # covered population, independently, so each is on the same [0,1] scale as
        # the (already rank-normed) conflict axis.
        cov = [r for r in recs if r["covered"]]
        denom = max(1, len(cov) - 1)
        for axis in ("perceptual_lidar", "perceptual_camera"):
            order = sorted(range(len(cov)), key=lambda k: cov[k][axis + "_raw"])
            for rnk, k in enumerate(order):
                cov[k][axis] = rnk / denom
        for r in recs:                                   # non-covered (excluded from Gold)
            r.setdefault("perceptual_lidar", r["perceptual_lidar_raw"])
            r.setdefault("perceptual_camera", r["perceptual_camera_raw"])

        # Second pass: emit BOTH difficulties = noisy-OR union (behavioral ∪ perceptual).
        scores = []
        for r in recs:
            diff_lidar = 1.0 - (1.0 - r["behavioral"]) * (1.0 - r["perceptual_lidar"])
            diff_camera = 1.0 - (1.0 - r["behavioral"]) * (1.0 - r["perceptual_camera"])
            r["sub"]["perceptual_lidar_axis"] = round(r["perceptual_lidar"], 4)
            r["sub"]["perceptual_camera_axis"] = round(r["perceptual_camera"], 4)
            detail = json.dumps({
                "method": "noisy-or-union(behavioral=conflict, perceptual=darkness/low_conf); dual lidar+camera",
                "sub_scores": {k: (round(v, 4) if v is not None else None)
                               for k, v in r["sub"].items()},
                "hour": r["hour"], "season": r["season"],
                "country": r["country"], "chunk": r["chunk"],
            })
            scores.append(Row(
                clip_id=r["clip_id"],
                difficulty_score=float(diff_camera),   # primary = camera (this consumer)
                difficulty_camera=float(diff_camera),
                difficulty_lidar=float(diff_lidar),
                scene_score=float(diff_camera),
                perception_score=perception_scores.get(r["clip_id"]),
                backend="metadata",
                detail=detail,
                scored_at=ts,
                sensor_covered=r["covered"],
            ))

        elapsed = time.time() - t0
        print(f"  Scored {total:,} clips in {elapsed:.1f}s "
              f"({total/elapsed:.0f} clips/s)")

        return self.spark.createDataFrame(scores, SCORE_SCHEMA)

    def _score_per_clip(self, limit: int, ts: str) -> DataFrame:
        """Per-clip scoring for GPU backends."""
        print(f"[GOLD SCORER] Per-clip scoring with {self.backend.name()}")

        # Load chunk -> season mapping for scene score
        chunk_seasons = _load_chunk_seasons(self.config.nvidia.source_path)
        source = self.config.nvidia.source_path

        # Get clip metadata (canonical Clip + aux tables for Nvidia-specific fields)
        try:
            clip_base = self.spark.table(self._silver("Clip"))
        except Exception:
            clip_base = self.spark.table(self._bronze("Clip"))
        clips_df = (
            clip_base
            .withColumn(
                "chunk",
                F.regexp_extract(F.col("session_id"), r"session_chunk_(\d+)", 1).cast("int"),
            )
            .join(self.spark.table(self._bronze("aux_data_collection")), "clip_id", "left")
            .join(self.spark.table(self._bronze("aux_sensor_presence")), "clip_id", "left")
        )

        if limit > 0:
            clips_df = clips_df.limit(limit)
        clip_rows = clips_df.collect()
        print(f"  Scoring {len(clip_rows):,} clips")

        radar_cols = [c for c in clips_df.columns
                      if c.startswith("radar_") and c != "radar_config"]

        scores = []
        t0 = time.time()
        for i, row in enumerate(clip_rows):
            d = row.asDict()
            clip_id = d["clip_id"]
            chunk = d.get("chunk") or 0
            hour = d.get("hour_of_day") or 12
            country = d.get("country") or "Unknown"
            season = chunk_seasons.get(chunk, "summer")
            sensor_flags = {rc: bool(d.get(rc)) for rc in radar_cols}

            # Find lidar files
            lidar_dir = os.path.join(source, "lidar", "lidar_top_360fov")
            lidar_paths = []
            if os.path.isdir(lidar_dir):
                import glob as globmod
                lidar_paths = globmod.glob(
                    os.path.join(lidar_dir, "**", f"{clip_id}*parquet"),
                    recursive=True)

            # Get ego trajectory from aux_egomotion (clip_id derived from filename)
            ego_traj = []
            try:
                UUID_RE = (
                    r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."
                )
                ego_rows = (
                    self.spark.table(self._bronze("aux_egomotion"))
                    .withColumn("clip_id", F.regexp_extract(F.input_file_name(), UUID_RE, 1))
                    .filter(F.col("clip_id") == clip_id)
                    .select("timestamp", "x", "y", "z", "qw", "qx", "qy", "qz")
                    .collect()
                )
                ego_traj = [r.asDict() for r in ego_rows]
            except Exception:
                pass

            # Scene complexity score
            scene_score, scene_sub = compute_scene_score(
                hour, season, country, sensor_flags, ego_traj)

            # Perception score from model
            try:
                camera_paths = {}
                cam_base = os.path.join(source, "camera")
                if os.path.isdir(cam_base):
                    import glob as globmod
                    for cam_sensor in os.listdir(cam_base):
                        mp4s = globmod.glob(os.path.join(
                            cam_base, cam_sensor, "**", f"{clip_id}*.mp4"),
                            recursive=True)
                        if mp4s:
                            camera_paths[cam_sensor] = mp4s

                percep_score, percep_detail = self.backend.score_clip(
                    clip_id, lidar_paths, ego_traj, camera_paths)
            except Exception as e:
                log.warning("Perception scoring failed for %s: %s", clip_id, e)
                percep_score = 0.5

            # Combine: scene + perception
            scene_w = 0.4
            percep_w = 0.6
            final = scene_w * scene_score + percep_w * percep_score

            detail = json.dumps({
                "method": f"composite-{self.backend.name()}",
                "scene_sub_scores": {k: (round(v, 4) if v is not None else None)
                                     for k, v in scene_sub.items()},
                "scene_score": round(scene_score, 4),
                "perception_score": round(percep_score, 4),
                "weights": {"scene": scene_w, "perception": percep_w},
            })

            scores.append(Row(
                clip_id=clip_id,
                difficulty_score=float(final),
                scene_score=float(scene_score),
                perception_score=float(percep_score),
                backend=self.backend.name(),
                detail=detail,
                scored_at=ts,
                sensor_covered=True,   # GPU backends run on on-disk sensor data
            ))

            if (i + 1) % 100 == 0 or (i + 1) == len(clip_rows):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(clip_rows)}] "
                      f"{elapsed:.0f}s, {rate:.1f} clips/s")

        return self.spark.createDataFrame(scores, SCORE_SCHEMA)

    def build_gold_subset(self, scores_df: DataFrame, top_pct: float = 10.0,
                          gold_axis: str = "camera") -> Dict[str, int]:
        """Select the hardest clips as the Gold subset.

        clip_scores stores BOTH `difficulty_camera` (camera-only, this consumer)
        and `difficulty_lidar` (lidar-fused, general purpose). `gold_axis` picks
        which one materializes the Gold VIEWS (default "camera"); the other tier
        is reported + remains derivable from clip_scores by thresholding its column.

        Args:
            scores_df: DataFrame with difficulty scores
            top_pct: percentage of hardest clips to include (default 10%)
            gold_axis: "camera" or "lidar" — which difficulty drives the Gold views

        Returns:
            Dict of Gold table name -> row count
        """
        diff_col = "difficulty_camera" if gold_axis == "camera" else "difficulty_lidar"
        print(f"[GOLD] Building Gold subset (top {top_pct}%, axis={gold_axis} -> {diff_col})")

        # Write scores table (both difficulty columns persisted)
        scores_table = self._gold("clip_scores")
        scores_df.writeTo(scores_table).using("iceberg").tableProperty(
            "format-version", "2"
        ).createOrReplace()
        total_scored = self.spark.table(scores_table).count()
        print(f"  Wrote {total_scored:,} scores to {scores_table}")

        # Determine threshold — over the SENSOR-COVERED clips only (the on-disk
        # 10TB sample). Catalog-only clips lack a validated signal and are
        # excluded from the difficulty tier (battery, 2026-06-23).
        covered = scores_df.filter(F.col("sensor_covered"))
        n_covered = covered.count()
        print(f"  Sensor-covered (on-disk) clips: {n_covered:,} of {total_scored:,}")

        # Report BOTH tiers (camera + lidar) so each Gold subset is quantified.
        q = 1.0 - top_pct / 100.0
        for ax, col in (("camera", "difficulty_camera"), ("lidar", "difficulty_lidar")):
            tr = covered.filter(F.col(col) >= 0).selectExpr(
                f"percentile_approx({col}, {q}) as t").collect()
            if tr and tr[0].t is not None:
                ng = covered.filter(F.col(col) >= tr[0].t).filter(F.col(col) >= 0).count()
                tag = "  <- materialized" if ax == gold_axis else ""
                print(f"  [{ax:6s}] threshold={tr[0].t:.4f}  Gold={ng:,}{tag}")

        threshold_row = (
            covered
            .filter(F.col(diff_col) >= 0)
            .selectExpr(f"percentile_approx({diff_col}, {q}) as threshold")
            .collect()
        )
        if not threshold_row:
            print("  [SKIP] No valid scores — cannot build Gold subset")
            return {}
        threshold = threshold_row[0].threshold

        # Gold clip_ids (sensor-covered + above threshold on the chosen axis)
        n_gold = (
            covered
            .filter(F.col(diff_col) >= threshold)
            .filter(F.col(diff_col) >= 0)
            .count()
        )
        print(f"  Gold clips ({gold_axis}): {n_gold:,} (of {n_covered:,} sensor-covered)")

        # Create Gold views for each Silver/Bronze table+view.
        # Silver sensor objects are Iceberg VIEWs, so SHOW TABLES misses them;
        # union SHOW TABLES with SHOW VIEWS.
        def _list_all(ns: str) -> List[str]:
            names: set[str] = set()
            try:
                names.update(
                    r[1] for r in
                    self.spark.sql(f"SHOW TABLES IN {self.cat}.{ns}").collect()
                )
            except Exception:
                pass
            try:
                names.update(
                    r[1] for r in
                    self.spark.sql(f"SHOW VIEWS IN {self.cat}.{ns}").collect()
                )
            except Exception:
                pass
            return sorted(names)

        results = {}
        source_tables = _list_all(self.ns_s)
        source_ns = self.ns_s
        if not source_tables:
            source_tables = _list_all(self.ns_b)
            source_ns = self.ns_b

        for tbl in source_tables:
            # Skip report tables and Silver helper views (e.g. `_failed_clips`),
            # which are internal and shouldn't get a Gold mirror.
            if tbl in ("quality_report", "clip_scores") or tbl.startswith("_"):
                continue

            try:
                cols = [f.name for f in
                        self.spark.table(f"{self.cat}.{source_ns}.{tbl}").schema]
                if "clip_id" in cols:
                    sql = (
                        f"CREATE OR REPLACE VIEW {self._gold(tbl)} "
                        f"AS SELECT s.* "
                        f"FROM {self.cat}.{source_ns}.{tbl} s "
                        f"WHERE s.clip_id IN ("
                        f"  SELECT clip_id FROM {scores_table} "
                        f"  WHERE {diff_col} >= {threshold} "
                        f"  AND {diff_col} >= 0 "
                        f"  AND sensor_covered"
                        f")"
                    )
                else:
                    sql = (
                        f"CREATE OR REPLACE VIEW {self._gold(tbl)} "
                        f"AS SELECT * FROM {self.cat}.{source_ns}.{tbl}"
                    )
                self.spark.sql(sql)
                count = self.spark.table(self._gold(tbl)).count()
                results[tbl] = count
                print(f"  [VIEW] {self._gold(tbl)}: {count:,} rows")
            except Exception as e:
                print(f"  [ERROR] {tbl}: {e}")
                results[tbl] = -1

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gold_scoring(
    config: Optional[NvidiaPipelineConfig] = None,
    backend_name: str = "metadata",
    gpu_id: int = 0,
    top_pct: float = 10.0,
    limit: int = 0,
    backend_kwargs: Optional[dict] = None,
    gold_axis: str = "camera",
) -> Tuple[DataFrame, Dict[str, int]]:
    """Run edge-case scoring and build Gold subset.

    Args:
        config: pipeline config
        backend_name: "dummy", "metadata", or "bevfusion"
        gpu_id: GPU device ID (0 = RTX 6000, 1 = A10 typically)
        top_pct: percentage of hardest clips for Gold
        limit: max clips to score (0 = all)
        backend_kwargs: extra kwargs for backend constructor

    Returns:
        (scores_df, gold_view_counts)
    """
    if config is None:
        config = NvidiaPipelineConfig()

    if backend_name not in BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available: {list(BACKENDS.keys())}")
    backend_cls = BACKENDS[backend_name]
    backend = backend_cls(**(backend_kwargs or {}))

    spark = build_spark_session(config, app_name=f"nvidia-gold-{backend_name}")
    try:
        create_namespaces(spark, config)
        backend.setup(gpu_id=gpu_id)

        scorer = EdgeCaseScorer(spark, config, backend)
        scores_df = scorer.score_all_clips(limit=limit)
        gold_results = scorer.build_gold_subset(scores_df, top_pct=top_pct, gold_axis=gold_axis)

        # Score distribution summary (over the sensor-covered difficulty tier)
        stats = (
            scores_df
            .filter(F.col("difficulty_score") >= 0)
            .filter(F.col("sensor_covered"))
            .agg(
                F.min("difficulty_score").alias("min"),
                F.max("difficulty_score").alias("max"),
                F.avg("difficulty_score").alias("mean"),
                F.stddev("difficulty_score").alias("std"),
            )
            .collect()[0]
        )

        print(f"\n{'='*60}")
        print(f"GOLD SCORING COMPLETE")
        print(f"{'='*60}")
        print(f"  Backend:       {backend_name}")
        print(f"  Clips scored:  {scores_df.count():,}")
        print(f"  Score range:   [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Score mean:    {stats['mean']:.4f} (std: {stats['std']:.4f})")
        print(f"  Top {top_pct}%:     {sum(1 for v in gold_results.values() if v > 0)} Gold views")

        return scores_df, gold_results

    finally:
        backend.teardown()
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gold edge-case scoring pipeline")
    parser.add_argument("--backend", default="metadata",
                        choices=list(BACKENDS.keys()),
                        help="Scoring backend (default: metadata)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (default: 0)")
    parser.add_argument("--top-pct", type=float, default=10.0,
                        help="Percentage of hardest clips for Gold (default: 10)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max clips to score (0 = all)")
    parser.add_argument("--bevfusion-config", default=None)
    parser.add_argument("--bevfusion-checkpoint", default=None)
    args = parser.parse_args()

    backend_kwargs = {}
    if args.backend == "bevfusion":
        backend_kwargs = {
            "config_path": args.bevfusion_config,
            "checkpoint_path": args.bevfusion_checkpoint,
        }

    scores_df, gold_results = run_gold_scoring(
        backend_name=args.backend,
        gpu_id=args.gpu,
        top_pct=args.top_pct,
        limit=args.limit,
        backend_kwargs=backend_kwargs,
    )

    for tbl, count in sorted(gold_results.items()):
        status = "+" if count >= 0 else "-"
        print(f"  {status} {tbl}: {count:,}")
