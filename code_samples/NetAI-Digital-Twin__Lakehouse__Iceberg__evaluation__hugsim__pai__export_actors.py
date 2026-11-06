"""export_actors.py — the scene's reconstructed vehicles as UnicyclePlanner actor assets.

HUGSIM's UnicyclePlanner replays a recorded vehicle along the unicycle path fitted to its
track in training (unicycle_<track>.pth), but it poses the actor as a 3DRealCar asset:
rotation Ry(-yaw), position (a, ground + height, b) (`sim/utils/plan.py`). Training placed
the same vehicle's Gaussians (dynamic_<track>.pth) with euler_to_rotmat('xzy', [ex, ez, ey])
= Ry(ey) Rz(ez) Rx(ex), ey = pi/2 - yaw, at the box centre (`gaussian_renderer.unicycle_b2w`).
The two differ by the constant C = Ry(pi/2) Rz(ez) Rx(ex), with ex, ez the track's median box
roll/pitch, so baking C into the Gaussians (means and rotations) lets the planner render the
vehicle as reconstructed. View-dependent colour (SH degree > 0) is dropped rather than rotated.

  pixi run python pai/export_actors.py data/scenes/pai/ac73935a     # → <scene>/actors/track_<id>/

Writes gs.pth + wlh.json per track (the asset layout HUGSIM expects) and actors.json with the
`height` to put in a scenario's plan_list (box centre relative to the ground model, y down).
"""
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR

sys.path.append(os.getcwd())
from sim.utils.sim_utils import dense_cam_poses  # noqa: E402
from utils.dynamic_utils import unicycle  # noqa: E402

scene = os.path.abspath(sys.argv[1])
out_root = os.path.join(scene, "actors")
cam_poses, cam_height, cmds = pickle.load(open(os.path.join(scene, "ground_param.pkl"), "rb"))
cam_poses, _ = dense_cam_poses(np.asarray(cam_poses), cmds)


def ground_height(u, v):  # sim/utils/plan.py planner.ground_height
    d = np.sqrt((cam_poses[:-1, 0, 3] - u) ** 2 + (cam_poses[:-1, 2, 3] - v) ** 2)
    c2w = cam_poses[np.argmin(d)]
    local = np.linalg.inv(c2w)[:3, :3] @ np.array([u, 0, v]) + np.linalg.inv(c2w)[:3, 3]
    local[1] = 0
    return (c2w[:3, :3] @ local + c2w[:3, 3])[1] + cam_height


summary = {}
tracks = sorted(int(f[len("dynamic_"):-4]) for f in os.listdir(scene) if f.startswith("dynamic_"))
for tid in tracks:
    uc_path = os.path.join(scene, f"unicycle_{tid}.pth")
    if not os.path.exists(uc_path):
        continue
    uc = unicycle.restore(torch.load(uc_path, weights_only=False))
    ex, ez = np.median(np.pi / 2 - uc.pitchroll.detach().cpu().numpy(), axis=0)
    C = SCR.from_euler("y", np.pi / 2).as_matrix() @ SCR.from_euler("xzy", [ex, ez, 0.0]).as_matrix()

    params, it = torch.load(os.path.join(scene, f"dynamic_{tid}.pth"), weights_only=False)
    sh, xyz, f_dc, f_rest, feats3d, scaling, rotation, opacity, lr_scale = params
    Ct = torch.tensor(C, dtype=xyz.dtype, device=xyz.device)
    xyz = xyz @ Ct.T
    q = rotation / rotation.norm(dim=-1, keepdim=True)  # (w, x, y, z), as GaussianModel stores it
    qc = SCR.from_matrix(C).as_quat()  # (x, y, z, w)
    w0, x0, y0, z0 = (torch.tensor(v, dtype=q.dtype, device=q.device) for v in (qc[3], qc[0], qc[1], qc[2]))
    w1, x1, y1, z1 = q.unbind(-1)
    rotation = torch.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dim=-1)
    f_rest = torch.zeros_like(f_rest)

    # extents after C: x should be the length (heading axis under Ry(-yaw)), y the height
    p = xyz.detach().cpu().numpy()
    ext = np.percentile(p, 98, axis=0) - np.percentile(p, 2, axis=0)
    ts = uc.train_timestamp.detach().cpu().numpy()
    hs = [float(uc.forward(torch.tensor(t, device=uc.train_timestamp.device))[5]) - ground_height(
        float(uc.forward(torch.tensor(t, device=uc.train_timestamp.device))[0]),
        float(uc.forward(torch.tensor(t, device=uc.train_timestamp.device))[1])) for t in ts]
    height = float(np.median(hs))
    wlh = [float(ext[2]), float(ext[0]), float(ext[1])]  # width (z), length (x), height (y)

    d = os.path.join(out_root, f"track_{tid}")
    os.makedirs(d, exist_ok=True)
    torch.save(([sh, xyz, f_dc, f_rest, feats3d, scaling, rotation, opacity, lr_scale], it), os.path.join(d, "gs.pth"))
    json.dump(wlh, open(os.path.join(d, "wlh.json"), "w"))
    summary[tid] = {"height": round(height, 3), "wlh": [round(v, 2) for v in wlh],
                    "box_roll_pitch_deg": [round(float(np.degrees(ex)), 1), round(float(np.degrees(ez)), 1)],
                    "track_s": [round(float(ts[0]), 2), round(float(ts[-1]), 2)], "n_gaussians": int(p.shape[0])}
    print(f"track {tid}: {summary[tid]}")
json.dump(summary, open(os.path.join(out_root, "actors.json"), "w"), indent=1)
