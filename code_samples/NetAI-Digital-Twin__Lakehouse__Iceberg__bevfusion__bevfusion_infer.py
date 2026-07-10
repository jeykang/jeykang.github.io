"""Shared BEVFusion inference helpers: model-input assembly + per-frame run.

Both the smoke test (`test_one_clip.py`) and the batch scorer (`runner.py`)
import from here, so the exact data-dict contract that was validated end-to-end
is the one that runs in production.

The contract was hard-won (see MEDALLION_PROGRESS.md §12). Three things matter:
  1. Multi-cam images must be passed as `inputs['img'] = [tensor(N,C,H,W)]`
     — a *list* of per-sample 4D tensors. Det3DDataPreprocessor.collate_data
     then routes through multiview_img_stack_batch to produce the 5D
     (B,N,C,H,W) tensor BEVFusion.extract_img_feat unpacks. A single 4D tensor
     stays 4D and crashes the 5-way unpack.
  2. Images must be resized to the config's view_transform image_size
     (256×704, H×W); otherwise the LSS depth map and image features disagree
     on spatial size in get_cam_feats.
  3. The data sample must carry `box_type_3d` / `box_mode_3d` metainfo — the
     detection head's predict_by_feat needs them to wrap decoded boxes.

Calibration is a placeholder (nuScenes-typical intrinsics + identity
extrinsics). We accept the domain shift: the Gold subset only needs a relative
difficulty signal, not absolute detection accuracy.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

# NOTE: torch and mmdet3d are imported lazily inside build_data/run_frame so
# that importing this module for its constants (CAM_ORDER, IMAGE_HW, …) stays
# cheap — runner.py relies on that to keep `--help` fast.

# nuScenes camera order BEVFusion expects, mapped to PhysicalAI sensors.
# nuScenes has a rear-center cam PhysicalAI lacks, so rear_left substitutes.
CAM_ORDER = [
    "camera_front_wide_120fov",   # CAM_FRONT
    "camera_cross_right_120fov",  # CAM_FRONT_RIGHT
    "camera_cross_left_120fov",   # CAM_FRONT_LEFT
    "camera_rear_left_70fov",     # CAM_BACK (substitute; nuScenes has rear-center)
    "camera_rear_left_70fov",     # CAM_BACK_LEFT
    "camera_rear_right_70fov",    # CAM_BACK_RIGHT
]

# BEVFusion config view_transform image_size, as (H, W). cv2.resize wants (W, H).
IMAGE_HW = (256, 704)

# nuScenes 10-class detection order (label index -> class name).
NUSC_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def build_default_calibration(n_cams: int, image_hw=IMAGE_HW) -> List[np.ndarray]:
    """nuScenes-typical 4×4 intrinsics with principal point at image center."""
    h, w = image_hw
    intrinsics = []
    for _ in range(n_cams):
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = K[1, 1] = 600.0   # plausible placeholder focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        intrinsics.append(K)
    return intrinsics


def ensure_5col(points: np.ndarray) -> np.ndarray:
    """BEVFusion expects (x, y, z, intensity, time). PhysicalAI Draco gives xyz;
    synthesize intensity=1.0, time=0.0 for the missing columns."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[1] >= 5:
        return points[:, :5]
    out = np.zeros((points.shape[0], 5), dtype=np.float32)
    out[:, :3] = points[:, :3]
    out[:, 3] = 1.0  # intensity
    return out


def build_data(points: np.ndarray, cam_imgs: List[np.ndarray], device,
               image_hw=IMAGE_HW) -> dict:
    """Assemble the dict for `model.test_step` (batch size 1).

    cam_imgs: list of N RGB uint8 (H, W, 3) arrays, ALREADY resized to
    image_hw. Pixels stay in 0-255 range — the model's Det3DDataPreprocessor
    (bgr_to_rgb=False, ImageNet mean/std) does normalization itself.
    """
    import torch
    from mmdet3d.structures import (Box3DMode, Det3DDataSample,
                                    LiDARInstance3DBoxes)

    n_cams = len(cam_imgs)
    imgs = np.stack(cam_imgs, axis=0)                       # (N, H, W, 3)
    points_t = torch.from_numpy(ensure_5col(points)).float().to(device)
    imgs_t = (torch.from_numpy(imgs).float()
              .permute(0, 3, 1, 2).contiguous().to(device))  # (N, 3, H, W)

    intrinsics = build_default_calibration(n_cams, image_hw)
    eye4 = np.eye(4, dtype=np.float32)
    lidar2cam = [eye4 for _ in range(n_cams)]
    cam2lidar = [eye4 for _ in range(n_cams)]
    lidar2img = [intrinsics[i] @ lidar2cam[i] for i in range(n_cams)]
    img_aug = [eye4 for _ in range(n_cams)]

    ds = Det3DDataSample()
    ds.set_metainfo({
        "img_shape": [image_hw] * n_cams,
        "ori_shape": [image_hw] * n_cams,
        "cam2img": intrinsics,
        "lidar2cam": lidar2cam,
        "cam2lidar": cam2lidar,
        "lidar2img": lidar2img,
        "img_aug_matrix": img_aug,
        "lidar_aug_matrix": eye4,
        "num_views": n_cams,
        "box_type_3d": LiDARInstance3DBoxes,
        "box_mode_3d": Box3DMode.LIDAR,
    })
    return {"inputs": {"points": [points_t], "img": [imgs_t]},
            "data_samples": [ds]}


def run_frame(model, points: np.ndarray, cam_imgs: List[np.ndarray],
              score_thr: float = 0.1) -> Dict:
    """Run BEVFusion on one (lidar, N-cam) frame.

    Returns {n_detections, max_conf, class_counts} where n_detections /
    class_counts count boxes with score >= score_thr and max_conf is the raw
    top score (kept threshold-independent so a clip is never fully "empty").
    """
    import torch

    device = next(model.parameters()).device
    data = build_data(points, cam_imgs, device)
    with torch.no_grad():
        result = model.test_step(data)
    pi = result[0].pred_instances_3d
    scores = pi.scores_3d.detach().cpu().numpy()
    labels = pi.labels_3d.detach().cpu().numpy()
    keep = scores >= score_thr
    class_counts: Dict[str, int] = {}
    for lb in labels[keep]:
        lb = int(lb)
        name = NUSC_CLASSES[lb] if 0 <= lb < len(NUSC_CLASSES) else str(lb)
        class_counts[name] = class_counts.get(name, 0) + 1
    return {
        "n_detections": int(keep.sum()),
        "max_conf": float(scores.max()) if scores.size else 0.0,
        "class_counts": class_counts,
    }
