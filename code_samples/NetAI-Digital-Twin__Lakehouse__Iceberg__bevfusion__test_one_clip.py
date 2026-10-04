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

import projects.BEVFusion.bevfusion  # registers BEVFusion
from mmdet3d.apis import init_model

# Production inference path — exercised here so the smoke test validates the
# exact contract runner.py uses.
from bevfusion_infer import CAM_ORDER, run_frame

CFG = "/workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT = "/workspace/checkpoints/bevfusion_lidar-cam_spconv2.pth"


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


def load_camera_frame(mp4_path: str, target_size=(704, 256)) -> np.ndarray:
    """Open mp4, grab middle frame, resize to BEVFusion-expected dims.

    target_size is (W, H) for cv2.resize. It MUST match the config's
    view_transform `image_size=[256, 704]` (H, W) → feature_size [32, 88]:
    the LSS depth map and image features are concatenated at feature
    resolution, so a wrong input size makes their spatial dims disagree
    (e.g. 448-tall input → feature height 56 ≠ the expected 32).

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
        cam_imgs.append(load_camera_frame(cams[sensor]))
    print(f"  imgs: {len(cam_imgs)} × {cam_imgs[0].shape}", flush=True)

    print(f">>> Running inference (shared bevfusion_infer.run_frame) ...", flush=True)
    try:
        # score_thr=0.0 so the smoke test reports the full raw query set; the
        # batch scorer applies a real threshold via --score-thr.
        res = run_frame(model, points, cam_imgs, score_thr=0.0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"INFERENCE FAILED: {e}")
        sys.exit(3)
    print(f"  detections (score>=0): {res['n_detections']}")
    print(f"  max_conf:              {res['max_conf']:.3f}")
    print(f"  class_counts:          {res['class_counts']}")
    if res["n_detections"] == 0:
        print("WARN: zero detections — check model/inputs")
        sys.exit(4)
    print(">>> SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
