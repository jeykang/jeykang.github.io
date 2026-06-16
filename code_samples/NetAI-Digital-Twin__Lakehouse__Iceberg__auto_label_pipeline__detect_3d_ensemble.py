"""Ensemble 3D detector: CenterPoint (TensorRT) + PointPillars (PyTorch).

Combines CenterPoint's higher recall on major classes (car, truck, bus,
bicycle, pedestrian) with PointPillars' broader 10-class vocabulary
(adds motorcycle, trailer, barrier, construction_vehicle, traffic_cone).

Merge strategy per class:
  - Shared classes (5): run class-aware distance NMS across both detector
    outputs, preferring the higher-confidence detection.
  - PointPillars-only classes (5): pass through directly.

The two detectors can run sequentially (sharing GPU) or the voxelization
steps can overlap with inference on a CUDA stream — but for simplicity
we run them sequentially here since voxelization is the bottleneck, not
the neural networks.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from .detect_3d import Detection3D, Detector3D
from .detect_3d_centerpoint import DetectorCenterPoint, CENTERPOINT_CLASSES

logger = logging.getLogger(__name__)

# Classes present in both detectors
SHARED_CLASSES = set(CENTERPOINT_CLASSES)  # car, truck, bus, bicycle, pedestrian

# Classes only in PointPillars (nuScenes 10-class)
PP_ONLY_CLASSES = {"motorcycle", "trailer", "barrier", "construction_vehicle", "traffic_cone"}


def _merge_detections(
    dets_a: List[Detection3D],
    dets_b: List[Detection3D],
    distance_threshold: float = 1.5,
) -> List[Detection3D]:
    """Merge two detection lists with class-aware distance NMS.

    For shared classes: keeps the higher-confidence detection when two
    boxes of the same class are within distance_threshold (or within
    half the larger box's max dimension, whichever is greater).

    For non-shared classes: passes through all detections.

    Args:
        dets_a: First detection list (typically CenterPoint — higher recall).
        dets_b: Second detection list (typically PointPillars — more classes).
        distance_threshold: Base distance for suppression (meters).

    Returns:
        Merged, deduplicated detection list sorted by confidence.
    """
    # Split by shared vs exclusive classes
    shared_a = [d for d in dets_a if d.class_name in SHARED_CLASSES]
    shared_b = [d for d in dets_b if d.class_name in SHARED_CLASSES]
    exclusive_b = [d for d in dets_b if d.class_name in PP_ONLY_CLASSES]

    # Merge shared-class detections via distance NMS
    combined = shared_a + shared_b
    combined.sort(key=lambda d: d.confidence, reverse=True)

    keep = []
    for det in combined:
        pos = np.array([det.x, det.y, det.z])
        suppressed = False
        for kept in keep:
            if det.class_name != kept.class_name:
                continue
            kept_pos = np.array([kept.x, kept.y, kept.z])
            dist = np.linalg.norm(pos - kept_pos)
            # Adaptive threshold: max of base threshold or half the larger box
            size_thresh = max(det.length, det.width, kept.length, kept.width) * 0.5
            if dist < max(size_thresh, distance_threshold):
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    # Add exclusive classes (no NMS needed — only one source)
    keep.extend(exclusive_b)

    # Sort final output by confidence
    keep.sort(key=lambda d: d.confidence, reverse=True)
    return keep


class DetectorEnsemble:
    """Ensemble detector combining CenterPoint and PointPillars."""

    def __init__(
        self,
        score_threshold: float = 0.3,
        distance_threshold: float = 1.5,
        centerpoint_kwargs: Optional[dict] = None,
        pointpillars_kwargs: Optional[dict] = None,
    ):
        cp_kw = centerpoint_kwargs or {}
        pp_kw = pointpillars_kwargs or {}

        self.score_threshold = score_threshold
        self.distance_threshold = distance_threshold

        t0 = time.time()
        self.centerpoint = DetectorCenterPoint(
            score_threshold=score_threshold,
            use_tensorrt=cp_kw.get("use_tensorrt", True),
            **{k: v for k, v in cp_kw.items() if k != "use_tensorrt"},
        )
        t_cp = time.time() - t0

        t0 = time.time()
        self.pointpillars = Detector3D(
            score_threshold=score_threshold,
            **pp_kw,
        )
        t_pp = time.time() - t0

        logger.info(
            f"Ensemble loaded: CenterPoint ({self.centerpoint._backend}, {t_cp:.1f}s) "
            f"+ PointPillars (PyTorch GPU, {t_pp:.1f}s)"
        )

    def detect(
        self,
        points: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> List[Detection3D]:
        """Run both detectors on a single point cloud and merge results."""
        thresh = score_threshold or self.score_threshold

        cp_dets = self.centerpoint.detect(points, score_threshold=thresh)
        pp_dets = self.pointpillars.detect(points, score_threshold=thresh)

        return _merge_detections(cp_dets, pp_dets, self.distance_threshold)

    def detect_clip_spins(
        self,
        lidar_spins: list,
        score_threshold: Optional[float] = None,
    ) -> Dict[int, List[Detection3D]]:
        """Run ensemble detection on all LiDAR spins of a clip."""
        thresh = score_threshold or self.score_threshold

        cp_results = self.centerpoint.detect_clip_spins(lidar_spins, score_threshold=thresh)
        pp_results = self.pointpillars.detect_clip_spins(lidar_spins, score_threshold=thresh)

        merged = {}
        for spin in lidar_spins:
            idx = spin.spin_index
            cp_dets = cp_results.get(idx, [])
            pp_dets = pp_results.get(idx, [])
            merged[idx] = _merge_detections(cp_dets, pp_dets, self.distance_threshold)

        return merged
