#!/usr/bin/env python3
"""load_ncore.py — one PhysicalAI-AV clip's NCore v4 store -> HUGSIM reconstruction input.

    cd evaluation/ncore && PYTHONPATH=repo .venv/bin/python ../hugsim/pai/load_ncore.py \
        --ncore out/pai_<clip>/pai_<clip>.json --out ../hugsim/data/pai/<short> [--stride 3] [--width 960]

This is the dataset-specific `load.py` step of HUGSIM's preprocessing (see HUGSIM's
data/nusc/load.py, which it mirrors); the rest of the pipeline (InverseForm semantics,
dynamic masks, UniDepth depth, merged point clouds) is HUGSIM's own. The clip is presented
as data_type "nuscenes", so the output has nuScenes camera folders, six images per sample
in nuScenes order (HUGSIM's dataset steps back six frames to find the previous image of the
same camera):

    CAM_FRONT       <- camera_front_wide_120fov     CAM_BACK        <- camera_front_tele_30fov
    CAM_FRONT_LEFT  <- camera_cross_left_120fov     CAM_BACK_LEFT   <- camera_rear_left_70fov
    CAM_FRONT_RIGHT <- camera_cross_right_120fov    CAM_BACK_RIGHT  <- camera_rear_right_70fov

The names are labels only (poses carry the geometry); CAM_BACK is really the front tele.

Writes, as nusc/load.py does:
  images/CAM_*/NNNNN.jpg     resampled to an ideal pinhole (see below)
  meta_data.json             OPENCV model; camtoworld relative to the first front-camera pose;
                             4x4 intrinsics; per-sample `dynamics` {track: box pose};
                             per-track box `verts`; timestamps in s from the first sample
  front_info.json            front-camera height over a RANSAC ground plane + pitch rect_mat
  cam_rigid_config.json      camera-to-front-camera extrinsics (rigid bundle adjustment)
  ground_lidar.ply           the near-ego ground points of the first LiDAR frame (rig frame)
  rectification.json         per-camera pinhole parameters and valid-pixel fraction (ours)
  semantics/CAM_*/NNNNN.npy  with --aux-sseg: the NRE aux store's Mask2Former labels, which
                             already use HUGSIM's Cityscapes ids (0 road ... 18 bicycle, 19
                             ego car), resampled with the same maps (nearest); this replaces
                             HUGSIM's InverseForm step. Rows at the bottom (hood) and columns at
                             an edge (body side, on the rear cameras) where the ego car covers
                             more than --ego-frac are cropped from every image of that camera.

Optics: every PAI camera is f-theta with a rolling shutter. Each is resampled to a pinhole
with fx = fy, the horizontal FOV capped at --max-hfov, and the principal point placed so the
image stays inside the valid f-theta field (PAI's principal point sits low, y ~ 747/1080),
using NCore's own camera model. The rolling shutter is not modelled: every image gets the
end-of-exposure pose.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from upath import UPath

from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader
from ncore.impl.sensors.camera import FThetaCameraModel

CAM_MAP = {  # nuScenes order (HUGSIM's AVAILABLE_CAMERAS)
    "CAM_FRONT": "camera_front_wide_120fov",
    "CAM_FRONT_LEFT": "camera_cross_left_120fov",
    "CAM_FRONT_RIGHT": "camera_cross_right_120fov",
    "CAM_BACK": "camera_front_tele_30fov",
    "CAM_BACK_LEFT": "camera_rear_left_70fov",
    "CAM_BACK_RIGHT": "camera_rear_right_70fov",
}
# HUGSIM box frame (nusc/utils.py): x = right (width), y = forward (length), z = up
WLH_TO_LWH = np.array([[0, 1.0, 0, 0], [-1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def get_vertices(dim):
    """HUGSIM's get_vertices: dim = (w, l, h) along the box frame's x, y, z; centred."""
    v = np.zeros((8, 3))
    v[:4, 0] += dim[0] / 2
    v[4:, 0] -= dim[0] / 2
    v[[0, 1, 4, 5], 1] += dim[1] / 2
    v[[2, 3, 6, 7], 1] -= dim[1] / 2
    v[[0, 2, 5, 7], 2] += dim[2] / 2
    v[[1, 3, 4, 6], 2] -= dim[2] / 2
    return v


def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
               "property float x", "property float y", "property float z"]
        if rgb is not None:
            hdr += ["property uchar red", "property uchar green", "property uchar blue"]
        f.write(("\n".join(hdr + ["end_header"]) + "\n").encode())
        if rgb is None:
            f.write(xyz.tobytes())
        else:
            rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            rgb = np.asarray(rgb, dtype=np.uint8)
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())


def ransac_plane(pts, thresh=0.02, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ p[0]
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, d, inl)
    n, d, inl = best
    # least-squares refit on the inliers
    q = pts[inl]
    c = q.mean(0)
    n = np.linalg.svd(q - c)[2][-1]
    if n[2] < 0:
        n = -n
    return np.array([n[0], n[1], n[2], -n @ c]), inl


class Rectifier:
    """f-theta -> ideal pinhole resampling for one camera (static maps)."""

    def __init__(self, params, width, max_hfov_deg, margin=0.97, max_aspect=0.75):
        self.src_w, self.src_h = (int(x) for x in params.resolution)
        self.model = FThetaCameraModel(params, device="cpu", dtype=torch.float32)
        ppx, ppy = (float(x) for x in params.principal_point)
        probe = np.array([[0.0, ppy], [self.src_w - 1.0, ppy], [ppx, 0.0], [ppx, self.src_h - 1.0]])
        rays = self.model.image_points_to_camera_rays(probe).cpu().numpy()
        ang_h = [abs(math.atan2(r[0], r[2])) for r in rays[:2]]
        up, down = abs(math.atan2(rays[2][1], rays[2][2])), abs(math.atan2(rays[3][1], rays[3][2]))
        h_half = min(min(ang_h) * margin, math.radians(max_hfov_deg) / 2)
        up, down = up * margin, down * margin
        self.W = int(width)
        for _ in range(40):  # shrink until (almost) every pinhole pixel lands in the f-theta image
            fx = (self.W / 2) / math.tan(h_half)
            cy = fx * math.tan(min(up, math.radians(80)))
            H = cy + fx * math.tan(min(down, math.radians(80)))
            Hmax = int(round(self.W * max_aspect))
            if H > Hmax:  # trim the sky first
                cy -= H - Hmax
                H = Hmax
            H = int(round(H)) // 2 * 2
            self.fx = self.fy = fx
            self.cx, self.cy, self.H = self.W / 2.0, cy, H
            self._build_maps()
            if self.valid_frac >= 0.995:
                break
            h_half *= 0.97
            up *= 0.97
            down *= 0.97
        self.hfov_deg = math.degrees(2 * math.atan(self.W / 2 / self.fx))

    def _build_maps(self):
        u, v = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        rays = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], -1).reshape(-1, 3)
        res = self.model.camera_rays_to_image_points(rays.astype(np.float32))
        pts = res.image_points.cpu().numpy().reshape(self.H, self.W, 2)
        ok = res.valid_flag.cpu().numpy().reshape(self.H, self.W)
        ok &= (pts[..., 0] >= 0) & (pts[..., 0] <= self.src_w - 1) & (pts[..., 1] >= 0) & (pts[..., 1] <= self.src_h - 1)
        self.map_x = np.where(ok, pts[..., 0] - 0.5, -1).astype(np.float32)
        self.map_y = np.where(ok, pts[..., 1] - 0.5, -1).astype(np.float32)
        self.valid_frac = float(ok.mean())

    def K4(self):
        K = np.eye(4)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = self.fx, self.fy, self.cx, self.cy
        return K

    def __call__(self, img):
        return cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def labels(self, lab):
        return cv2.remap(lab, self.map_x, self.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=19)

    def crop(self, bottom=0, left=0, right=0):
        """Drop ego-car rows/columns (hood at the bottom, body side at an edge); keeps sizes even."""
        bottom = int(bottom) + (int(self.H - bottom) % 2)
        right = int(right) + (int(self.W - left - right) % 2)
        self.H -= bottom
        self.map_x, self.map_y = self.map_x[: self.H, left: self.W - right], self.map_y[: self.H, left: self.W - right]
        self.W -= left + right
        self.cx -= left


class AuxSseg:
    """Per-frame Mask2Former labels from the NRE aux store (PNG, class id = R / 3), keyed by
    (camera, end-of-exposure timestamp)."""

    def __init__(self, path):
        self.tar = tarfile.open(path)
        self.members = {}
        for n in self.tar.getnames():
            p = n.split("/")
            if len(p) == 5 and p[0] == "aux" and p[1] == "semantic_segmentation" and p[4] == "0":
                self.members[(p[2], int(p[3]))] = n

    def get(self, cam, ts_end):
        raw = self.tar.extractfile(self.members[(cam, int(ts_end))]).read()
        return (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)[..., 2] // 3).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncore", required=True, help="NCore v4 sequence manifest (pai_<clip>.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3, help="front-camera frame stride (30 Hz / 3 = 10 Hz samples)")
    ap.add_argument("--width", type=int, default=960, help="rectified image width")
    ap.add_argument("--max-hfov", type=float, default=100.0)
    ap.add_argument("--start", type=int, default=0, help="first sample")
    ap.add_argument("--end", type=int, default=-1, help="one past the last sample (-1 = all)")
    ap.add_argument("--dynamic-min-move", type=float, default=2.0, help="m; tracks moving less are static background")
    ap.add_argument("--aux-sseg", default=None, help="NRE aux store with semantic segmentation (*.aux.sseg.zarr.itar)")
    ap.add_argument("--ego-frac", type=float, default=0.1, help="crop edge rows/columns whose ego-car fraction exceeds this")
    a = ap.parse_args()

    L = SequenceLoaderV4(SequenceComponentGroupsReader([UPath(a.ncore)]))
    sensors = {nus: L.get_camera_sensor(pai) for nus, pai in CAM_MAP.items()}
    rect = {}
    for nus, s in sensors.items():
        rect[nus] = Rectifier(s.model_parameters, a.width, a.max_hfov if "tele" not in CAM_MAP[nus] else 40.0)
        r = rect[nus]
        print(f"{nus:15s} <- {CAM_MAP[nus]:26s} {r.W}x{r.H} fx={r.fx:.1f} cy={r.cy:.1f} hfov={r.hfov_deg:.1f} valid={r.valid_frac:.4f}")
    sseg = AuxSseg(a.aux_sseg) if a.aux_sseg else None
    if sseg is not None:
        for nus, s in sensors.items():
            ts = np.asarray(s.frames_timestamps_us)[:, 1]
            lab = rect[nus].labels(sseg.get(CAM_MAP[nus], ts[len(ts) // 2]))
            ego = lab == 19
            run = lambda flags: int(np.argmax(~flags)) if (~flags).any() else len(flags)  # leading run of True
            bottom = run(ego.mean(1)[::-1] > a.ego_frac)
            left, right = run(ego.mean(0) > a.ego_frac), run(ego.mean(0)[::-1] > a.ego_frac)
            rect[nus].crop(bottom, left, right)
            print(f"{nus:15s} ego crop: bottom {bottom}, left {left}, right {right} -> {rect[nus].W}x{rect[nus].H}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({nus: {"source": CAM_MAP[nus], "width": r.W, "height": r.H, "fx": r.fx, "fy": r.fy, "cx": r.cx, "cy": r.cy,
                     "hfov_deg": r.hfov_deg, "valid_fraction": r.valid_frac} for nus, r in rect.items()},
              open(os.path.join(a.out, "rectification.json"), "w"), indent=1)

    # --- samples: every --stride-th front-camera frame; other cameras take their closest frame
    front = sensors["CAM_FRONT"]
    f_ts = np.asarray(front.frames_timestamps_us)[:, 1]  # end of exposure
    sample_idx = list(range(0, len(f_ts), a.stride))
    sample_idx = sample_idx[a.start:(None if a.end < 0 else a.end)]
    sample_ts = f_ts[sample_idx]
    cam_frames = {}
    for nus, s in sensors.items():
        ts = np.asarray(s.frames_timestamps_us)[:, 1]
        cam_frames[nus] = [int(np.argmin(np.abs(ts - t))) for t in sample_ts]
    T_front0 = np.asarray(front.get_frames_T_sensor_target("world", np.array([sample_idx[0]])))[0]
    inv_pose = np.linalg.inv(T_front0)

    # --- front_info.json + ground_lidar.ply (first LiDAR frame, rig frame)
    T_cam_rig = {nus: np.asarray(s.T_sensor_rig, dtype=np.float64) for nus, s in sensors.items()}
    lid = L.get_lidar_sensor(L.lidar_ids[0])
    li0 = int(np.argmin(np.abs(np.asarray(lid.frames_timestamps_us)[:, 1] - sample_ts[0])))
    pc = lid.get_frame_point_cloud(li0, motion_compensation=True, with_start_points=False)
    T_lid_rig = np.asarray(lid.T_sensor_rig, dtype=np.float64)
    pts = np.asarray(pc.xyz_m_end, dtype=np.float64) @ T_lid_rig[:3, :3].T + T_lid_rig[:3, 3]
    near = (pts[:, 0] > -8) & (pts[:, 0] < 12) & (np.abs(pts[:, 1]) < 6) & (pts[:, 2] < 0.5)
    ego = (pts[:, 0] > -1.5) & (pts[:, 0] < 5.0) & (np.abs(pts[:, 1]) < 1.5)
    gpts = pts[near & ~ego]
    plane, inl = ransac_plane(gpts)
    a_, b_, c_, d_ = plane
    write_ply(os.path.join(a.out, "ground_lidar.ply"), gpts[inl])
    fc = T_cam_rig["CAM_FRONT"]
    ground_z = -(a_ * fc[0, 3] + b_ * fc[1, 3] + d_) / c_
    cam_x = fc[:3, 0]  # nusc/load.py uses the camera x axis (column 0) here
    pitch_angle = np.arccos(np.dot(plane[:3], cam_x) / (np.linalg.norm(plane[:3]) * np.linalg.norm(cam_x)))
    rect_mat = R.from_euler("x", np.pi / 2 - pitch_angle).as_matrix()
    json.dump({"height": float(fc[2, 3] - ground_z), "rect_mat": rect_mat.tolist()},
              open(os.path.join(a.out, "front_info.json"), "w"))
    print(f"ground plane n=({a_:.3f},{b_:.3f},{c_:.3f}) inliers {int(inl.sum())}/{len(gpts)}; front camera height {fc[2, 3] - ground_z:.3f} m")

    # --- cam_rigid_config.json
    cams = []
    for i, nus in enumerate(CAM_MAP):
        rel = np.linalg.inv(T_cam_rig[nus]) @ T_cam_rig["CAM_FRONT"]
        q = R.from_matrix(rel[:3, :3]).as_quat()
        cams.append({"camera_id": i + 1, "image_prefix": f"{nus}/", "cam_from_rig_rotation": [q[3], q[0], q[1], q[2]],
                     "cam_from_rig_translation": rel[:3, 3].tolist()})
    json.dump([{"ref_camera_id": 1, "cameras": cams}], open(os.path.join(a.out, "cam_rigid_config.json"), "w"), indent=4)

    # --- dynamic objects: cuboid tracks -> HUGSIM world, dynamic ones only
    obs = defaultdict(list)
    for o in L.get_cuboid_track_observations():
        T_ref_world = np.asarray(L.pose_graph.evaluate_poses(o.reference_frame_id, "world", np.array([int(o.reference_frame_timestamp_us)])))[0]
        b = np.eye(4)
        b[:3, :3] = R.from_euler("xyz", o.bbox3.rot).as_matrix()
        b[:3, 3] = o.bbox3.centroid
        pose = inv_pose @ T_ref_world @ b @ WLH_TO_LWH
        obs[str(o.track_id)].append((int(o.timestamp_us), pose, o.bbox3.dim, o.class_id))
    dynamic, verts = {}, {}
    for tid, lst in obs.items():
        lst.sort(key=lambda x: x[0])
        move = np.abs(lst[0][1][:3, 3] - lst[-1][1][:3, 3]).max()
        if move > a.dynamic_min_move:
            dynamic[tid] = lst
            l_, w_, h_ = max(x[2][0] for x in lst), max(x[2][1] for x in lst), max(x[2][2] for x in lst)
            verts[tid] = get_vertices((w_, l_, h_)).tolist()
    print(f"cuboid tracks: {len(obs)}, dynamic (> {a.dynamic_min_move} m): {len(dynamic)}")

    # --- frames
    meta = {"camera_model": "OPENCV", "verts": verts, "frames": [], "inv_pose": inv_pose.tolist(),
            "source": {"ncore": os.path.abspath(a.ncore), "sequence_id": L.sequence_id, "cameras": CAM_MAP,
                       "stride": a.stride, "samples": len(sample_idx)}}
    for nus in CAM_MAP:
        os.makedirs(os.path.join(a.out, "images", nus), exist_ok=True)
        if sseg is not None:
            os.makedirs(os.path.join(a.out, "semantics", nus), exist_ok=True)
    t0 = sample_ts[0]
    for k, t in enumerate(tqdm(sample_ts, desc="samples")):
        dyn = {}
        for tid, lst in dynamic.items():
            j = int(np.argmin([abs(x[0] - t) for x in lst]))
            if abs(lst[j][0] - t) <= 100_000:
                dyn[tid] = lst[j][1].tolist()
        for nus, s in sensors.items():
            fi = cam_frames[nus][k]
            img = np.asarray(s.get_frame_image_array(fi))
            out = rect[nus](img)
            name = f"{k:05d}.jpg"
            cv2.imwrite(os.path.join(a.out, "images", nus, name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if sseg is not None:
                ts_end = int(np.asarray(s.frames_timestamps_us)[fi, 1])
                np.save(os.path.join(a.out, "semantics", nus, f"{k:05d}.npy"), rect[nus].labels(sseg.get(CAM_MAP[nus], ts_end)))
            c2w = inv_pose @ np.asarray(s.get_frames_T_sensor_target("world", np.array([fi])))[0]
            meta["frames"].append({"rgb_path": os.path.join("./images", nus, name), "camtoworld": c2w.tolist(),
                                   "intrinsics": rect[nus].K4().tolist(), "width": rect[nus].W, "height": rect[nus].H,
                                   "timestamp": float((t - t0) / 1e6), "dynamics": dyn})
    json.dump(meta, open(os.path.join(a.out, "meta_data.json"), "w"), indent=1)
    print(f"wrote {len(meta['frames'])} frames ({len(sample_idx)} samples x {len(CAM_MAP)} cameras) to {a.out}")


if __name__ == "__main__":
    main()
