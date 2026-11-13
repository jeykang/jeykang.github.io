#!/usr/bin/env python3
"""lidar_points.py — LiDAR-seeded HUGSIM initial point clouds for a converted PAI clip.

    cd evaluation/ncore && PYTHONPATH=repo .venv/bin/python ../hugsim/pai/lidar_points.py \
        --ncore out/pai_<clip>/pai_<clip>.json --src ../hugsim/data/pai/<short> --out ../hugsim/data/pai/<short>_lidar

HUGSIM's preprocessing seeds the background Gaussians (points3d.ply) and the ground model
(ground_points3d.ply) by unprojecting monocular UniDepth depth (data/utils/merge_depth_*.py).
PAI clips carry a 360-degree LiDAR, so this rebuilds both clouds from accumulated LiDAR returns,
keeping HUGSIM's filters:

  - each point takes its label and colour from the rectified camera images of the nearest
    sample it projects into (semantics + RGB at the pixel), with a loose occlusion test
    against that image's depth map; points no camera sees are dropped (no supervision);
  - static non-ground points (label > 1, not sky, not ego car) -> points3d.ply
    (merge_depth_wo_ground.py keeps semantics > 1 outside the dynamic masks);
  - road/sidewalk points (label <= 1) -> ground_points3d.ply, flattened to the front-camera
    height in the nearest front-camera frame, exactly as merge_depth_ground.py does;
  - returns from the ego car and from inside moving-vehicle boxes (meta_data 'dynamics') are
    removed: HUGSIM seeds moving vehicles from a template, not from scene points.

Everything else (images, semantics, masks, depth, meta_data.json, front_info.json,
cam_rigid_config.json, ground_param.pkl, ...) is hard-linked from --src, so the new source
directory trains with pai/train.sh unchanged.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from upath import UPath

from ncore.impl.data.v4.compat import SequenceLoaderV4
from ncore.impl.data.v4.components import SequenceComponentGroupsReader

SKY, EGO = 10, 19
EGO_BOX_RIG = ((-1.3, 5.0), (-1.25, 1.25), (-0.5, 2.3))  # x, y, z ranges in the rig frame (m), with margin


def write_ply(path, xyz, rgb):
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}",
                            "property float x", "property float y", "property float z",
                            "property uchar red", "property uchar green", "property uchar blue", "end_header"]) + "\n").encode())
        f.write(rec.tobytes())


def voxel_unique(xyz, voxel):
    """Indices of one point per occupied voxel."""
    q = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return idx


def link_tree(src, dst):
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if f in ("points3d.ply", "ground_points3d.ply"):
                continue
            s, d = os.path.join(root, f), os.path.join(dst, rel, f)
            if not os.path.exists(d):
                os.link(s, d)


class Sample:
    """The six rectified cameras of one sample: projection, labels, colours, depth."""

    def __init__(self, src, frames):
        self.cams = []
        for fr in frames:
            K = np.array(fr["intrinsics"])
            img = cv2.cvtColor(cv2.imread(os.path.join(src, fr["rgb_path"])), cv2.COLOR_BGR2RGB)
            stem = fr["rgb_path"].replace("./", "")
            smt = np.load(os.path.join(src, stem.replace("images", "semantics").replace(".jpg", ".npy")))
            dep_p = os.path.join(src, stem.replace("images", "depth").replace(".jpg", ".pt"))
            dep = torch.load(dep_p).numpy() if os.path.exists(dep_p) else None
            self.cams.append((np.linalg.inv(np.array(fr["camtoworld"])), K[0, 0], K[1, 1], K[0, 2], K[1, 2], img, smt, dep))

    def look_up(self, p):
        """Label and colour for world points p (N,3) from the first camera that sees them; -1 = unseen."""
        n = len(p)
        label = np.full(n, -1, np.int16)
        rgb = np.zeros((n, 3), np.uint8)
        todo = np.ones(n, bool)
        for w2c, fx, fy, cx, cy, img, smt, dep in self.cams:
            if not todo.any():
                break
            idx = np.nonzero(todo)[0]
            q = p[idx] @ w2c[:3, :3].T + w2c[:3, 3]
            z = q[:, 2]
            ok = z > 0.5
            u = np.full(len(idx), -1, np.int64)
            v = np.full(len(idx), -1, np.int64)
            u[ok] = np.floor(fx * q[ok, 0] / z[ok] + cx).astype(np.int64)
            v[ok] = np.floor(fy * q[ok, 1] / z[ok] + cy).astype(np.int64)
            H, W = smt.shape
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if dep is not None:  # loose occlusion test: not far behind the surface the camera saw
                d = np.zeros(len(idx))
                d[ok] = dep[v[ok], u[ok]]
                ok &= z <= d * 1.3 + 1.0
            hit = idx[ok]
            label[hit] = smt[v[ok], u[ok]]
            rgb[hit] = img[v[ok], u[ok]]
            todo[hit] = False
        return label, rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncore", required=True)
    ap.add_argument("--src", required=True, help="source dir written by load_ncore.py (+ prepare.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--n-scene", type=int, default=600_000)
    ap.add_argument("--n-ground", type=int, default=200_000)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.src, "meta_data.json")))
    frames, inv_pose, verts = meta["frames"], np.array(meta["inv_pose"]), meta["verts"]
    ncam = len(meta["source"]["cameras"])
    samples = [frames[i:i + ncam] for i in range(0, len(frames), ncam)]
    half = {tid: np.abs(np.array(v)).max(0) + 0.3 for tid, v in verts.items()}  # box half-extents + margin
    front_info = json.load(open(os.path.join(a.src, "front_info.json")))

    L = SequenceLoaderV4(SequenceComponentGroupsReader([UPath(a.ncore)]))
    front = L.get_camera_sensor(meta["source"]["cameras"]["CAM_FRONT"])
    stride = int(meta["source"]["stride"])
    f_ts = np.asarray(front.frames_timestamps_us)[:, 1]
    sample_us = f_ts[np.arange(0, len(f_ts), stride)][: len(samples)]
    lid = L.get_lidar_sensor(L.lidar_ids[0])
    T_lid_rig = np.asarray(lid.T_sensor_rig, dtype=np.float64)
    l_ts = np.asarray(lid.frames_timestamps_us)[:, 1]

    scene_xyz, scene_rgb, ground_xyz, ground_rgb = [], [], [], []
    stats = dict(total=0, ego=0, far=0, dynamic=0, unseen=0, sky_or_ego=0, ground=0, scene=0)
    cache_k, smp = None, None
    for li in tqdm(range(lid.frames_count), desc="lidar frames"):
        k = int(np.argmin(np.abs(sample_us - l_ts[li])))
        if abs(sample_us[k] - l_ts[li]) > 100_000:
            continue
        pts = np.asarray(lid.get_frame_point_cloud(li, motion_compensation=True, with_start_points=False).xyz_m_end, dtype=np.float64)
        stats["total"] += len(pts)
        rig = pts @ T_lid_rig[:3, :3].T + T_lid_rig[:3, 3]
        (x0, x1), (y0, y1), (z0, z1) = EGO_BOX_RIG
        ego = (rig[:, 0] > x0) & (rig[:, 0] < x1) & (rig[:, 1] > y0) & (rig[:, 1] < y1) & (rig[:, 2] > z0) & (rig[:, 2] < z1)
        far = np.linalg.norm(pts, axis=1) > a.max_range
        stats["ego"] += int(ego.sum()); stats["far"] += int((far & ~ego).sum())
        pts = pts[~ego & ~far]
        T = inv_pose @ np.asarray(lid.get_frames_T_sensor_target("world", np.array([li])))[0]
        p = pts @ T[:3, :3].T + T[:3, 3]
        keep = np.ones(len(p), bool)
        for tid, pose in samples[k][0]["dynamics"].items():  # moving vehicles at this sample
            inv = np.linalg.inv(np.array(pose))
            q = p @ inv[:3, :3].T + inv[:3, 3]
            keep &= ~np.all(np.abs(q) < half[tid], axis=1)
        stats["dynamic"] += int((~keep).sum())
        p = p[keep]
        if cache_k != k:
            cache_k, smp = k, Sample(a.src, samples[k])
        label, rgb = smp.look_up(p)
        seen = label >= 0
        stats["unseen"] += int((~seen).sum())
        g = seen & (label <= 1)
        s = seen & (label > 1) & (label != SKY) & (label != EGO)
        stats["sky_or_ego"] += int((seen & ((label == SKY) | (label == EGO))).sum())
        stats["ground"] += int(g.sum()); stats["scene"] += int(s.sum())
        ground_xyz.append(p[g].astype(np.float32)); ground_rgb.append(rgb[g])
        scene_xyz.append(p[s].astype(np.float32)); scene_rgb.append(rgb[s])
        if li % 20 == 19:  # keep memory bounded: thin to one point per voxel as we go
            for xyz, col in ((scene_xyz, scene_rgb), (ground_xyz, ground_rgb)):
                X, C = np.concatenate(xyz), np.concatenate(col)
                i = voxel_unique(X, a.voxel)
                xyz[:], col[:] = [X[i]], [C[i]]

    rng = np.random.default_rng(0)
    out = {}
    for name, xyz, col, n in (("scene", scene_xyz, scene_rgb, a.n_scene), ("ground", ground_xyz, ground_rgb, a.n_ground)):
        X, C = np.concatenate(xyz), np.concatenate(col)
        i = voxel_unique(X, a.voxel)
        X, C = X[i], C[i]
        if len(X) > n:
            j = rng.choice(len(X), n, replace=False)
            X, C = X[j], C[j]
        out[name] = (X.astype(np.float64), C)

    # ground: flatten to the front-camera height in the nearest front-camera frame (merge_depth_ground.py)
    G, GC = out["ground"]
    fposes = np.stack([np.array(f["camtoworld"]) for f in frames if "/CAM_FRONT/" in f["rgb_path"]])
    centres = fposes[:-1, :3, 3]
    nearest = np.empty(len(G), np.int64)
    for s0 in range(0, len(G), 50_000):
        d = ((G[s0:s0 + 50_000, None, :] - centres[None]) ** 2).sum(-1)
        nearest[s0:s0 + 50_000] = d.argmin(1)
    c2w, w2c = fposes[nearest], np.linalg.inv(fposes)[nearest]
    local = np.einsum("nij,nj->ni", w2c[:, :3, :3], G) + w2c[:, :3, 3]
    local[:, 1] = front_info["height"]
    G = np.einsum("nij,nj->ni", c2w[:, :3, :3], local) + c2w[:, :3, 3]

    os.makedirs(a.out, exist_ok=True)
    link_tree(a.src, a.out)
    write_ply(os.path.join(a.out, "points3d.ply"), out["scene"][0], out["scene"][1])
    write_ply(os.path.join(a.out, "ground_points3d.ply"), G, GC)
    stats.update(points3d=len(out["scene"][0]), ground_points3d=len(G))
    json.dump(stats, open(os.path.join(a.out, "lidar_points.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
