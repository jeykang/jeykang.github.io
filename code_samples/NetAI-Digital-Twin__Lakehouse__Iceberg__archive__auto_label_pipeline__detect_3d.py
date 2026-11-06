"""Stage 4: 3D object detection using OpenPCDet PointPillars (nuScenes pretrained).

Replaces the geometric frustum-LiDAR approach with a proper learned 3D detector.
Detects 10 classes: car, truck, construction_vehicle, bus, trailer, barrier,
motorcycle, bicycle, pedestrian, traffic_cone.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .detect_2d import Detection2D

logger = logging.getLogger(__name__)

# nuScenes class names from the PointPillars config
NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Default model config
OPENPCDET_ROOT = "/tmp/OpenPCDet"
CFG_FILE = "cfgs/nuscenes_models/cbgs_pp_multihead.yaml"
CKPT_FILE = "/tmp/pp_multihead_nds5823_updated.pth"


@dataclass
class Detection3D:
    """A 3D oriented bounding box in the LiDAR frame."""

    class_name: str
    confidence: float
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    detection_2d: Optional[Detection2D] = None
    num_lidar_points: int = 0
    dimension_score: float = 1.0


def _voxelize_points(
    points: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    max_points_per_voxel: int = 20,
    max_num_voxels: int = 30000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy voxelization for PointPillars.

    Args:
        points: (N, C) point cloud (at least xyz, optionally more features).
        voxel_size: (3,) voxel dimensions [dx, dy, dz].
        point_cloud_range: (6,) [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_voxel: Max points retained per voxel.
        max_num_voxels: Max total voxels.

    Returns:
        voxels: (M, max_points_per_voxel, C) padded point features per voxel.
        coords: (M, 3) voxel coordinates [z_idx, y_idx, x_idx] (OpenPCDet format).
        num_points: (M,) actual point count per voxel.
    """
    # Filter to range
    mask = (
        (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] < point_cloud_range[3])
        & (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] < point_cloud_range[4])
        & (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] < point_cloud_range[5])
    )
    points = points[mask]

    # Compute voxel indices
    grid_idx = np.floor((points[:, :3] - point_cloud_range[:3]) / voxel_size).astype(np.int32)
    grid_size = np.ceil((point_cloud_range[3:6] - point_cloud_range[:3]) / voxel_size).astype(np.int32)

    # Unique voxel keys
    # Encode as single int: z * Ny * Nx + y * Nx + x
    keys = grid_idx[:, 2] * grid_size[1] * grid_size[0] + grid_idx[:, 1] * grid_size[0] + grid_idx[:, 0]

    # Find unique voxels
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    n_voxels = min(len(unique_keys), max_num_voxels)
    n_features = points.shape[1]

    # Sort by count descending to keep the most populated voxels
    if len(unique_keys) > max_num_voxels:
        top_idx = np.argsort(-counts)[:max_num_voxels]
        keep_keys = set(unique_keys[top_idx])
        keep_mask = np.array([k in keep_keys for k in keys])
        points = points[keep_mask]
        grid_idx = grid_idx[keep_mask]
        keys = keys[keep_mask]
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_voxels = len(unique_keys)

    voxels = np.zeros((n_voxels, max_points_per_voxel, n_features), dtype=np.float32)
    coords = np.zeros((n_voxels, 3), dtype=np.int32)
    num_points = np.zeros(n_voxels, dtype=np.int32)

    # Fill voxels
    voxel_point_count = np.zeros(n_voxels, dtype=np.int32)
    for i in range(len(points)):
        voxel_idx = inverse[i]
        if voxel_point_count[voxel_idx] < max_points_per_voxel:
            voxels[voxel_idx, voxel_point_count[voxel_idx]] = points[i]
            voxel_point_count[voxel_idx] += 1

    num_points = voxel_point_count

    # Set voxel coordinates (z, y, x format for OpenPCDet)
    for i, key in enumerate(unique_keys):
        # Find first point in this voxel to get grid index
        first_pt_idx = np.where(keys == key)[0][0]
        gi = grid_idx[first_pt_idx]
        coords[i] = [gi[2], gi[1], gi[0]]  # z, y, x

    return voxels, coords, num_points


class Detector3D:
    """OpenPCDet PointPillars 3D object detector."""

    def __init__(
        self,
        openpcdet_root: str = OPENPCDET_ROOT,
        cfg_file: str = CFG_FILE,
        ckpt_file: str = CKPT_FILE,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
    ):
        if openpcdet_root not in sys.path:
            sys.path.insert(0, openpcdet_root)

        # Import and patch spconv
        from pcdet.utils.spconv_utils import spconv as _  # noqa: trigger mock
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network

        # Load config from the tools/ directory
        orig_dir = os.getcwd()
        os.chdir(os.path.join(openpcdet_root, "tools"))
        cfg_from_yaml_file(cfg_file, cfg)
        os.chdir(orig_dir)

        cfg.TAG = "cbgs_pp_multihead"
        self.cfg = cfg
        self.device = device
        self.score_threshold = score_threshold

        # Point cloud config
        self.point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        voxel_cfg = cfg.DATA_CONFIG.DATA_PROCESSOR[2]
        self.voxel_size = np.array(voxel_cfg.VOXEL_SIZE)
        self.max_points_per_voxel = voxel_cfg.MAX_POINTS_PER_VOXEL
        self.max_num_voxels = voxel_cfg.MAX_NUMBER_OF_VOXELS.get("test", 30000)
        self.class_names = list(cfg.CLASS_NAMES)

        # Build mock dataset for model init
        class MockEncoder:
            num_point_features = 5  # x, y, z, intensity, timestamp

        class MockDataset:
            class_names = cfg.CLASS_NAMES
            point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
            voxel_size = np.array(voxel_cfg.VOXEL_SIZE)
            point_feature_encoder = MockEncoder()
            depth_downsample_factor = None
            def __init__(self):
                self.grid_size = (
                    (self.point_cloud_range[3:6] - self.point_cloud_range[0:3])
                    / self.voxel_size
                ).astype(int)

        dataset = MockDataset()
        self.model = build_network(
            model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
        )

        log = logging.getLogger("openpcdet")
        log.setLevel(logging.INFO)
        if not log.handlers:
            log.addHandler(logging.StreamHandler())
        self.model.load_params_from_file(ckpt_file, logger=log)
        self.model.to(device)
        self.model.eval()
        logger.info(
            f"PointPillars loaded: {sum(p.numel() for p in self.model.parameters()):,} params, "
            f"{len(self.class_names)} classes"
        )

    @torch.no_grad()
    def detect(
        self,
        points: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> List[Detection3D]:
        """Run 3D detection on a single LiDAR point cloud.

        Args:
            points: (N, 3+) point cloud. Must have at least xyz. If fewer than
                    5 columns, pads with zeros (intensity, timestamp).
            score_threshold: Override default confidence threshold.

        Returns:
            List of Detection3D objects.
        """
        thresh = score_threshold or self.score_threshold

        # Voxelize (GPU-accelerated if torch.cuda is available)
        try:
            from .voxelize_gpu import voxelize_pointpillars
            voxels, coords, num_points = voxelize_pointpillars(
                points,
                self.voxel_size,
                self.point_cloud_range,
                self.max_points_per_voxel,
                self.max_num_voxels,
            )
        except Exception:
            # Fallback to CPU voxelization
            if points.shape[1] < 5:
                pad = np.zeros((points.shape[0], 5 - points.shape[1]), dtype=np.float32)
                points = np.hstack([points.astype(np.float32), pad])
            else:
                points = points[:, :5].astype(np.float32)
            voxels, coords, num_points = _voxelize_points(
                points,
                self.voxel_size,
                self.point_cloud_range,
                self.max_points_per_voxel,
                self.max_num_voxels,
            )

        if len(voxels) == 0:
            return []

        # Add batch index to coords: (M, 3) -> (M, 4) with batch=0
        batch_coords = np.hstack([np.zeros((len(coords), 1), dtype=np.int32), coords])

        # Build batch_dict
        batch_dict = {
            "voxels": torch.from_numpy(voxels).to(self.device),
            "voxel_coords": torch.from_numpy(batch_coords).to(self.device),
            "voxel_num_points": torch.from_numpy(num_points).to(self.device),
            "batch_size": 1,
        }

        # Forward pass
        pred_dicts, _ = self.model(batch_dict)

        # Parse predictions
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].cpu().numpy()  # (N, 9): x,y,z,dx,dy,dz,heading,vx,vy
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()  # 1-indexed

        detections = []
        for i in range(len(scores)):
            if scores[i] < thresh:
                continue

            class_idx = int(labels[i]) - 1  # convert to 0-indexed
            if class_idx < 0 or class_idx >= len(self.class_names):
                continue

            box = boxes[i]
            detections.append(Detection3D(
                class_name=self.class_names[class_idx],
                confidence=float(scores[i]),
                x=float(box[0]),
                y=float(box[1]),
                z=float(box[2]),
                length=float(box[3]),  # dx
                width=float(box[4]),   # dy
                height=float(box[5]),  # dz
                yaw=float(box[6]),
                num_lidar_points=0,    # could count points in box if needed
                dimension_score=1.0,
            ))

        return detections

    def detect_clip_spins(
        self,
        lidar_spins: list,
        score_threshold: Optional[float] = None,
    ) -> Dict[int, List[Detection3D]]:
        """Run detection on all LiDAR spins of a clip.

        Args:
            lidar_spins: List of LidarSpin objects from decode.py.
            score_threshold: Override default confidence threshold.

        Returns:
            Dict[spin_index -> List[Detection3D]].
        """
        results = {}
        for spin in lidar_spins:
            dets = self.detect(spin.points, score_threshold=score_threshold)
            results[spin.spin_index] = dets
        return results


def suppress_duplicate_3d(
    detections: List[Detection3D],
    distance_threshold: float = 1.5,
) -> List[Detection3D]:
    """Distance-based NMS for 3D detections (e.g., from multiple sources)."""
    if len(detections) <= 1:
        return detections

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    for det in dets:
        pos = np.array([det.x, det.y, det.z])
        suppressed = False
        for kept in keep:
            kept_pos = np.array([kept.x, kept.y, kept.z])
            dist = np.linalg.norm(pos - kept_pos)
            size_thresh = max(det.length, det.width, kept.length, kept.width) * 0.5
            if dist < max(size_thresh, distance_threshold) and det.class_name == kept.class_name:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep
