"""Stage 4b: 3D object detection using CenterPoint (Autoware ONNX, TensorRT/ORT).

Uses the Autoware Foundation's pretrained CenterPoint PointPillar model
(nuScenes + TIER IV data, 5 classes: car, truck, bus, bicycle, pedestrian).

Two-stage ONNX inference:
  1. Pillar Feature Encoder → per-pillar 32-dim features
  2. BEV Backbone + CenterHead → heatmap + regression outputs
  With a scatter step in between (NumPy).
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Autoware CenterPoint config (matches the ONNX models)
POINT_CLOUD_RANGE = np.array([-76.8, -76.8, -4.0, 76.8, 76.8, 6.0])
VOXEL_SIZE = np.array([0.32, 0.32, 10.0])
GRID_SIZE = np.array([480, 480, 1])  # X, Y, Z grid dimensions
MAX_POINTS_PER_VOXEL = 20
MAX_NUM_VOXELS = 40000
NUM_POINT_FEATURES = 9  # x, y, z, intensity, time_lag, x_off, y_off, z_off, dist

# Class names (Autoware CenterPoint head order)
CENTERPOINT_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian"]

# ONNX model paths
ENCODER_ONNX = "/tmp/centerpoint_onnx/pts_voxel_encoder.onnx"
BACKBONE_ONNX = "/tmp/centerpoint_onnx/pts_backbone_neck_head.onnx"

# TensorRT engine paths
ENCODER_TRT = "/tmp/centerpoint_onnx/pts_voxel_encoder.engine"
BACKBONE_TRT = "/tmp/centerpoint_onnx/pts_backbone_neck_head.engine"


def _voxelize_pillars(
    points: np.ndarray,
    point_cloud_range: np.ndarray = POINT_CLOUD_RANGE,
    voxel_size: np.ndarray = VOXEL_SIZE,
    max_points_per_voxel: int = MAX_POINTS_PER_VOXEL,
    max_num_voxels: int = MAX_NUM_VOXELS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize points into pillars and compute 9-feature representation.

    The 9 features per point (matching Autoware PillarVFE):
        [x, y, z, intensity, time_lag, x-x_c, y-y_c, z-z_c, dist]
    where (x_c, y_c, z_c) is the pillar center and dist = sqrt(x²+y²).

    Returns:
        voxel_features: (M, max_points_per_voxel, 9)
        coords: (M, 2) pillar grid coordinates [x_idx, y_idx]
        num_points: (M,) actual point count per pillar
    """
    # Filter to range
    mask = (
        (points[:, 0] >= point_cloud_range[0]) & (points[:, 0] < point_cloud_range[3])
        & (points[:, 1] >= point_cloud_range[1]) & (points[:, 1] < point_cloud_range[4])
        & (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] < point_cloud_range[5])
    )
    points = points[mask]

    if len(points) == 0:
        return (
            np.zeros((0, max_points_per_voxel, 9), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )

    # Ensure at least 4 features (x, y, z, intensity); pad if needed
    if points.shape[1] < 4:
        pad = np.zeros((len(points), 4 - points.shape[1]), dtype=np.float32)
        points = np.hstack([points.astype(np.float32), pad])
    else:
        points = points[:, :4].astype(np.float32)

    # Compute pillar grid indices (x, y only — z is a single pillar)
    grid_idx = np.floor(
        (points[:, :2] - point_cloud_range[:2]) / voxel_size[:2]
    ).astype(np.int32)

    # Clamp to grid bounds
    grid_idx[:, 0] = np.clip(grid_idx[:, 0], 0, GRID_SIZE[0] - 1)
    grid_idx[:, 1] = np.clip(grid_idx[:, 1], 0, GRID_SIZE[1] - 1)

    # Unique pillar keys
    keys = grid_idx[:, 1] * GRID_SIZE[0] + grid_idx[:, 0]  # y * W + x
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    n_pillars = min(len(unique_keys), max_num_voxels)

    # Keep most populated pillars if over limit
    if len(unique_keys) > max_num_voxels:
        top_idx = np.argsort(-counts)[:max_num_voxels]
        keep_set = set(unique_keys[top_idx])
        keep_mask = np.array([k in keep_set for k in keys])
        points = points[keep_mask]
        grid_idx = grid_idx[keep_mask]
        keys = keys[keep_mask]
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_pillars = len(unique_keys)

    # Build raw voxels (x, y, z, intensity)
    raw_voxels = np.zeros((n_pillars, max_points_per_voxel, 4), dtype=np.float32)
    coords = np.zeros((n_pillars, 2), dtype=np.int32)  # x_idx, y_idx
    num_points = np.zeros(n_pillars, dtype=np.int32)

    voxel_point_count = np.zeros(n_pillars, dtype=np.int32)
    for i in range(len(points)):
        vidx = inverse[i]
        if voxel_point_count[vidx] < max_points_per_voxel:
            raw_voxels[vidx, voxel_point_count[vidx]] = points[i, :4]
            voxel_point_count[vidx] += 1
    num_points = voxel_point_count

    # Set coordinates
    for i, key in enumerate(unique_keys):
        x_idx = int(key % GRID_SIZE[0])
        y_idx = int(key // GRID_SIZE[0])
        coords[i] = [x_idx, y_idx]

    # Compute pillar centers (geometric center of the pillar cell)
    pillar_center_x = coords[:, 0].astype(np.float32) * voxel_size[0] + point_cloud_range[0] + voxel_size[0] / 2
    pillar_center_y = coords[:, 1].astype(np.float32) * voxel_size[1] + point_cloud_range[1] + voxel_size[1] / 2
    pillar_center_z = (point_cloud_range[2] + point_cloud_range[5]) / 2  # mid of z range

    # Build 9-feature representation per point
    # Standard PillarVFE features (OpenPCDet/Autoware):
    #   [x, y, z, intensity, x-mean_x, y-mean_y, z-mean_z, x-x_c, y-y_c]
    # where mean_x/y/z is the arithmetic mean of all points in the pillar
    # and x_c/y_c is the geometric center of the pillar cell
    features = np.zeros((n_pillars, max_points_per_voxel, 9), dtype=np.float32)
    for p in range(n_pillars):
        npts = num_points[p]
        if npts == 0:
            continue
        pts = raw_voxels[p, :npts]  # (npts, 4)

        # Cluster center (mean of points in pillar)
        mean_xyz = pts[:, :3].mean(axis=0)

        # Features 0-3: x, y, z, intensity
        features[p, :npts, 0] = pts[:, 0]
        features[p, :npts, 1] = pts[:, 1]
        features[p, :npts, 2] = pts[:, 2]
        features[p, :npts, 3] = pts[:, 3]

        # Features 4-6: offset from cluster center (mean of points)
        features[p, :npts, 4] = pts[:, 0] - mean_xyz[0]
        features[p, :npts, 5] = pts[:, 1] - mean_xyz[1]
        features[p, :npts, 6] = pts[:, 2] - mean_xyz[2]

        # Features 7-8: offset from pillar geometric center (x, y only)
        features[p, :npts, 7] = pts[:, 0] - pillar_center_x[p]
        features[p, :npts, 8] = pts[:, 1] - pillar_center_y[p]

    return features, coords, num_points


def _scatter_to_bev(
    pillar_features: np.ndarray,
    coords: np.ndarray,
    grid_x: int = 480,
    grid_y: int = 480,
    channels: int = 32,
) -> np.ndarray:
    """Scatter pillar features onto the BEV grid.

    Args:
        pillar_features: (M, 1, C) encoder output features.
        coords: (M, 2) pillar grid coordinates [x_idx, y_idx].
        grid_x, grid_y: BEV grid dimensions.
        channels: Feature channels (32).

    Returns:
        bev: (1, C, grid_y, grid_x) BEV pseudo-image.
    """
    bev = np.zeros((1, channels, grid_y, grid_x), dtype=np.float32)
    features = pillar_features[:, 0, :]  # (M, C)

    for i in range(len(coords)):
        xi, yi = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= xi < grid_x and 0 <= yi < grid_y:
            bev[0, :, yi, xi] = features[i]

    return bev


def _decode_heatmap(
    heatmap: np.ndarray,
    reg: np.ndarray,
    height: np.ndarray,
    dim: np.ndarray,
    rot: np.ndarray,
    vel: np.ndarray,
    score_threshold: float = 0.3,
    point_cloud_range: np.ndarray = POINT_CLOUD_RANGE,
    voxel_size: np.ndarray = VOXEL_SIZE,
) -> list:
    """Decode CenterPoint heatmap outputs into 3D detections.

    Args:
        heatmap: (1, num_classes, H, W) — class heatmaps (logits).
        reg: (1, 2, H, W) — center offset regression (x, y).
        height: (1, 1, H, W) — z regression.
        dim: (1, 3, H, W) — dimensions (dx, dy, dz).
        rot: (1, 2, H, W) — rotation (sin, cos).
        vel: (1, 2, H, W) — velocity (vx, vy).
        score_threshold: Minimum confidence.

    Returns:
        List of detection dicts.
    """
    from .detect_3d import Detection3D

    # Sigmoid on heatmap logits
    hm = 1.0 / (1.0 + np.exp(-np.clip(heatmap[0], -10, 10)))  # (num_classes, H, W)
    num_classes, H, W = hm.shape

    # Feature map stride (output is same size as BEV for this model)
    # The output grid matches the pillar grid, so stride = voxel_size
    stride_x = voxel_size[0]
    stride_y = voxel_size[1]

    detections = []

    for cls_idx in range(num_classes):
        cls_hm = hm[cls_idx]  # (H, W)

        # Find local maxima above threshold (simple 3x3 max pooling NMS)
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(cls_hm, size=3)
        peaks = (cls_hm == local_max) & (cls_hm >= score_threshold)

        ys, xs = np.where(peaks)

        for y, x in zip(ys, xs):
            score = float(cls_hm[y, x])

            # Decode center position
            cx = (x + float(reg[0, 0, y, x])) * stride_x + point_cloud_range[0]
            cy = (y + float(reg[0, 1, y, x])) * stride_y + point_cloud_range[1]
            cz = float(height[0, 0, y, x])

            # Decode dimensions (exp for positive values)
            dx = float(np.exp(dim[0, 0, y, x]))
            dy = float(np.exp(dim[0, 1, y, x]))
            dz = float(np.exp(dim[0, 2, y, x]))

            # Decode rotation
            sin_r = float(rot[0, 0, y, x])
            cos_r = float(rot[0, 1, y, x])
            yaw = float(np.arctan2(sin_r, cos_r))

            detections.append(Detection3D(
                class_name=CENTERPOINT_CLASSES[cls_idx],
                confidence=score,
                x=cx,
                y=cy,
                z=cz,
                length=dx,
                width=dy,
                height=dz,
                yaw=yaw,
                num_lidar_points=0,
                dimension_score=1.0,
            ))

    # Sort by confidence descending
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


class _TRTRunner:
    """Thin wrapper for TensorRT engine inference via the Python API."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        trt_logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Inspect I/O tensors
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def __call__(self, inputs: dict) -> dict:
        """Run inference. inputs: {name: np.ndarray}. Returns {name: np.ndarray}."""
        torch = self._torch
        trt = self._trt
        stream = torch.cuda.current_stream().cuda_stream

        # Set input shapes and copy data
        device_buffers = {}
        for name in self.input_names:
            arr = inputs[name]
            self.context.set_input_shape(name, arr.shape)
            d_buf = torch.from_numpy(arr).cuda().contiguous()
            device_buffers[name] = d_buf
            self.context.set_tensor_address(name, d_buf.data_ptr())

        # Allocate outputs
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            np_dtype = np.float32 if dtype == trt.float32 else np.float16
            d_buf = torch.empty(tuple(shape), dtype=torch.float32 if np_dtype == np.float32 else torch.float16, device="cuda")
            device_buffers[name] = d_buf
            self.context.set_tensor_address(name, d_buf.data_ptr())

        # Execute
        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()

        # Copy outputs back
        for name in self.output_names:
            outputs[name] = device_buffers[name].cpu().numpy().astype(np.float32)

        return outputs


class DetectorCenterPoint:
    """CenterPoint 3D detector with TensorRT (GPU) or ONNX Runtime (CPU) backends."""

    def __init__(
        self,
        encoder_path: str = ENCODER_ONNX,
        backbone_path: str = BACKBONE_ONNX,
        encoder_trt: str = ENCODER_TRT,
        backbone_trt: str = BACKBONE_TRT,
        score_threshold: float = 0.3,
        use_tensorrt: bool = True,
    ):
        self.score_threshold = score_threshold
        self.use_tensorrt = use_tensorrt and os.path.exists(encoder_trt) and os.path.exists(backbone_trt)

        if self.use_tensorrt:
            try:
                self._encoder = _TRTRunner(encoder_trt)
                self._backbone = _TRTRunner(backbone_trt)
                self._backend = "TensorRT"
                logger.info(
                    f"CenterPoint loaded (TensorRT FP16): {len(CENTERPOINT_CLASSES)} classes, "
                    f"grid={GRID_SIZE[0]}x{GRID_SIZE[1]}"
                )
            except Exception as e:
                logger.warning(f"TensorRT init failed ({e}), falling back to ONNX Runtime")
                self.use_tensorrt = False

        if not self.use_tensorrt:
            import onnxruntime as ort
            self._encoder = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
            self._backbone = ort.InferenceSession(backbone_path, providers=["CPUExecutionProvider"])
            self._backend = "ONNX Runtime (CPU)"
            logger.info(
                f"CenterPoint loaded (ONNX Runtime CPU): {len(CENTERPOINT_CLASSES)} classes, "
                f"grid={GRID_SIZE[0]}x{GRID_SIZE[1]}"
            )

    def _run_encoder(self, voxel_features: np.ndarray) -> np.ndarray:
        if self.use_tensorrt:
            out = self._encoder({"input_features": voxel_features})
            return out["pillar_features"]
        else:
            return self._encoder.run(None, {"input_features": voxel_features})[0]

    def _run_backbone(self, bev: np.ndarray) -> dict:
        if self.use_tensorrt:
            return self._backbone({"spatial_features": bev})
        else:
            outputs = self._backbone.run(None, {"spatial_features": bev})
            names = [o.name for o in self._backbone.get_outputs()]
            return dict(zip(names, outputs))

    def detect(
        self,
        points: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> list:
        """Run CenterPoint 3D detection on a single LiDAR point cloud.

        Args:
            points: (N, 3+) point cloud (xyz + optional intensity).
            score_threshold: Override default confidence threshold.

        Returns:
            List of Detection3D objects.
        """
        thresh = score_threshold or self.score_threshold

        # Step 1: Voxelize into pillars with 9-feature encoding
        try:
            from .voxelize_gpu import voxelize_centerpoint
            voxel_features, coords, num_points = voxelize_centerpoint(
                points, VOXEL_SIZE, POINT_CLOUD_RANGE, GRID_SIZE,
                MAX_POINTS_PER_VOXEL, MAX_NUM_VOXELS,
            )
        except Exception:
            voxel_features, coords, num_points = _voxelize_pillars(points)

        if len(voxel_features) == 0:
            return []

        # Step 2: Encode pillars → (M, 1, 32)
        enc_out = self._run_encoder(voxel_features)

        # Step 3: Scatter to BEV grid → (1, 32, 480, 480)
        # Use GPU scatter when possible
        try:
            import torch
            enc_feat = torch.from_numpy(enc_out[:, 0, :]).cuda()  # (M, 32)
            bev_t = torch.zeros((1, 32, GRID_SIZE[1], GRID_SIZE[0]),
                                dtype=torch.float32, device="cuda")
            cx = torch.from_numpy(coords[:, 0].astype(np.int64)).cuda()
            cy = torch.from_numpy(coords[:, 1].astype(np.int64)).cuda()
            # Clamp to valid BEV grid bounds
            valid = (cx >= 0) & (cx < GRID_SIZE[0]) & (cy >= 0) & (cy < GRID_SIZE[1])
            bev_t[0, :, cy[valid], cx[valid]] = enc_feat[valid].t()
            bev = bev_t.cpu().numpy()
        except Exception:
            bev = _scatter_to_bev(enc_out, coords)

        # Step 4: Backbone + CenterHead
        out_dict = self._run_backbone(bev)

        # Step 5: Decode detections
        detections = _decode_heatmap(
            heatmap=out_dict["heatmap"],
            reg=out_dict["reg"],
            height=out_dict["height"],
            dim=out_dict["dim"],
            rot=out_dict["rot"],
            vel=out_dict["vel"],
            score_threshold=thresh,
        )

        return detections

    def detect_clip_spins(
        self,
        lidar_spins: list,
        score_threshold: Optional[float] = None,
    ) -> Dict[int, list]:
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
