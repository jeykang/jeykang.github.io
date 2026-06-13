"""Smoke test: run BEVFusion inference on one PhysicalAI clip frame.

Bypasses inference_multi_modality_detector (which requires a nuScenes-format
ann file). Constructs the data dict directly and calls model.test_step().

Goal: prove we can get a non-empty Det3DDataSample out for one (lidar, 6×cam)
input. Once this works, we scale to multi-frame sampling per clip in runner.py.

Calibration strategy: nuScenes default intrinsics/extrinsics — we accept the
domain shift since we only need the relative difficulty signal, not absolute
detection accuracy.
"""
import argparse
import glob as globmod
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import projects.BEVFusion.bevfusion  # registers BEVFusion
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmengine.structures import InstanceData

CFG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = "/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth"

# Camera ordering BEVFusion config expects (nuScenes layout)
CAM_ORDER = [
    "camera_front_wide_120fov",   # CAM_FRONT
    "camera_cross_right_120fov",  # CAM_FRONT_RIGHT
    "camera_cross_left_120fov",   # CAM_FRONT_LEFT
    "camera_rear_left_70fov",     # CAM_BACK (substituting; nuScenes has rear-center)
    "camera_rear_left_70fov",     # CAM_BACK_LEFT
    "camera_rear_right_70fov",    # CAM_BACK_RIGHT
]


def load_lidar_points(parquet_path: str, frame_idx: int) -> np.ndarray:
    """Decode one Draco-encoded spin from a lidar parquet → (N, 5) array.

    BEVFusion expects (x, y, z, intensity, time). PhysicalAI lidar gives only
    xyz from DracoPy; we synthesize intensity=1.0, time=0.0.
    """
    import DracoPy
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path, columns=["draco_encoded_pointcloud"])
    blobs = t.column("draco_encoded_pointcloud").to_pylist()
    if not blobs:
        return np.zeros((0, 5), dtype=np.float32)
    blob = blobs[min(frame_idx, len(blobs) - 1)]
    mesh = DracoPy.decode(blob)
    pts = np.asarray(mesh.points, dtype=np.float32)  # (N, 3)
    if pts.shape[1] < 5:
        intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
        time = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([pts[:, :3], intensity, time], axis=1)
    return pts


def load_camera_frame(mp4_path: str, target_size=(800, 448)) -> np.ndarray:
    """Open mp4, grab middle frame, resize to BEVFusion-expected dims.

    Returns RGB uint8 (H, W, 3).
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame from {mp4_path}")
    frame = cv2.resize(frame, target_size)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_files(nfs_root: str, clip_id: str):
    cams = {}
    for sensor in CAM_ORDER:
        m = globmod.glob(f"{nfs_root}/camera/{sensor}/*/{clip_id}.{sensor}.mp4")
        cams[sensor] = m[0] if m else None
    lidar = globmod.glob(f"{nfs_root}/lidar/lidar_top_360fov/*/{clip_id}.lidar_top_360fov.parquet")
    return cams, (lidar[0] if lidar else None)


def build_default_calibration(n_cams=6):
    """Hardcoded nuScenes-typical intrinsics and lidar2img matrices.

    Used because we're trading accuracy for getting a signal flowing — we
    only need relative difficulty, not absolute mAP.
    """
    # Default nuScenes intrinsics for 1600×900 → scaled to 800×448
    intrinsics = []
    for _ in range(n_cams):
        K = np.array([[600.0, 0.0, 400.0],
                      [0.0, 600.0, 224.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        # Make 4×4 for lidar2img projection
        Kx = np.eye(4, dtype=np.float32)
        Kx[:3, :3] = K
        intrinsics.append(Kx)
    return intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--frame-idx", type=int, default=100)
    ap.add_argument("--nfs-root",
                    default="/mnt/netai-e2e/nvidia-physicalai-av-subset")
    args = ap.parse_args()

    print(f">>> Loading model ...", flush=True)
    model = init_model(CFG, CKPT, device="cuda:0")
    model.eval()
    print(f"  Model loaded: {type(model).__name__}", flush=True)

    print(f">>> Locating files for {args.clip_id} ...", flush=True)
    cams, lidar_path = find_files(args.nfs_root, args.clip_id)
    if not lidar_path:
        print("ERROR: no lidar parquet found")
        sys.exit(1)
    print(f"  lidar: {os.path.basename(lidar_path)}")
    for s, p in cams.items():
        print(f"  {s}: {os.path.basename(p) if p else 'MISSING'}")

    print(f">>> Decoding lidar frame {args.frame_idx} ...", flush=True)
    points = load_lidar_points(lidar_path, args.frame_idx)
    print(f"  points shape: {points.shape}", flush=True)

    print(f">>> Loading camera frames ...", flush=True)
    cam_imgs = []
    for sensor in CAM_ORDER:
        if not cams[sensor]:
            print(f"  WARN: missing {sensor}, skipping smoke test")
            sys.exit(2)
        img = load_camera_frame(cams[sensor])
        cam_imgs.append(img)
    imgs = np.stack(cam_imgs, axis=0)  # (6, H, W, 3)
    print(f"  imgs shape: {imgs.shape}", flush=True)

    print(f">>> Constructing data dict ...", flush=True)
    points_t = torch.from_numpy(points).float().cuda()
    # BEVFusion expects channels-first (N_cams, 3, H, W) normalized
    imgs_t = torch.from_numpy(imgs).float().permute(0, 3, 1, 2).cuda() / 255.0
    imgs_t = imgs_t.unsqueeze(0)  # batch dim

    intrinsics = build_default_calibration(n_cams=len(CAM_ORDER))
    # Identity extrinsics (placeholder — accept domain shift)
    eye4 = np.eye(4, dtype=np.float32)
    lidar2cam = [eye4 for _ in range(len(CAM_ORDER))]
    cam2lidar = [eye4 for _ in range(len(CAM_ORDER))]  # inverse of identity = identity
    lidar2img = [intrinsics[i] @ lidar2cam[i] for i in range(len(CAM_ORDER))]
    img_aug = [eye4 for _ in range(len(CAM_ORDER))]

    data_sample = Det3DDataSample()
    data_sample.set_metainfo({
        "img_shape": [(imgs.shape[1], imgs.shape[2])] * len(CAM_ORDER),
        "ori_shape": [(imgs.shape[1], imgs.shape[2])] * len(CAM_ORDER),
        "cam2img": intrinsics,
        "lidar2cam": lidar2cam,
        "cam2lidar": cam2lidar,
        "lidar2img": lidar2img,
        "img_aug_matrix": img_aug,
        "lidar_aug_matrix": eye4,
        "lidar_path": lidar_path,
        "num_views": len(CAM_ORDER),
    })

    # NOTE: mmdet3d 1.4.0's data_preprocessor.simple_process has an inconsistency:
    # it checks `if 'img' in data['inputs']` to compute batch_pad_shape, but
    # then accesses `inputs['imgs']` (plural) for the actual tensor. We pass
    # both keys with the same tensor to satisfy both code paths.
    # _get_pad_shape needs a 4D tensor for `img` (NCHW). Multi-cam BEVFusion
    # uses `imgs` (5D) for the actual fusion path. Pass `imgs_t.squeeze(0)`
    # which is (N_cams=6, 3, H, W) — treats cams as batch for pad_shape calc.
    data = {
        "inputs": {
            "points": [points_t],
            "imgs": imgs_t,
            "img": imgs_t.squeeze(0),
        },
        "data_samples": [data_sample],
    }

    print(f">>> Running inference ...", flush=True)
    with torch.no_grad():
        try:
            result = model.test_step(data)
            print(f"  result type: {type(result)}")
            if isinstance(result, list) and result:
                r = result[0]
                if hasattr(r, "pred_instances_3d"):
                    pi = r.pred_instances_3d
                    print(f"  detections: {len(pi.bboxes_3d)}")
                    if len(pi.bboxes_3d) > 0:
                        print(f"  scores range: [{pi.scores_3d.min():.3f}, "
                              f"{pi.scores_3d.max():.3f}]")
                        print(f"  classes: {pi.labels_3d.unique().tolist()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"INFERENCE FAILED: {e}")
            sys.exit(3)


if __name__ == "__main__":
    main()
