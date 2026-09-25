#!/usr/bin/env python3
"""DiffusionDrive (NAVSIM/Transfuser + diffusion head) as an evaluation policy.

An academic driving agent rather than a foundation model: ~700 MB of weights on
torch 2.0.1, no flash-attn, runs on any GPU. It is the counterweight to the 10B
Alpamayo runs — same contract, ~1/30th the compute.

Two reasons this model specifically:
  * The repo already established it partially transfers to PhysicalAI (open-loop
    L2 median 3.58 m, planning/diffusiondrive/VALIDITY_REPORT.md) using lidar BEV
    geometry rather than camera appearance, which is what sank SparseDrive.
  * That same report closed by saying a valid planning signal "must be
    scene-content based — e.g. rung-2 map-free PDMS (collision / TTC / progress
    against the detected agents) ... future work". This module is that work, so
    scoring DiffusionDrive through it closes the loop the report left open.

Feature construction mirrors planning/diffusiondrive/test_one_clip.py exactly
(same crops, same BEV binning, same 8-dim status) with one change: every input is
taken at the harness decision time t0 instead of a fixed mid-clip fraction, and
the camera and lidar come through `obs.sensors`, so the no-future guarantee holds.

Must run inside the container that has navsim + the checkpoint:
    docker run --rm --runtime=nvidia --gpus device=0 \\
      -v $PWD/netai-e2e:/mnt/netai-e2e -v $PWD/evaluation:/eval \\
      --user 1000:1007 -e HOME=/tmp -e AV_ROOT=/mnt/netai-e2e/nvidia-physicalai-av-subset \\
      --entrypoint python netai/diffusiondrive-runner:latest \\
      /eval/run_eval.py --policy policy_diffusiondrive:DiffusionDrivePolicy \\
        --limit 60 --workers 1 --require-sensors --out /eval/.results_diffusiondrive.parquet
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

CKPT = os.environ.get("DD_CKPT", "/workspace/DiffusionDrive/ckpt/diffusiondrive_navsim_88p1.pth")
CAM_L, CAM_F, CAM_R = ("camera_cross_left_120fov", "camera_front_wide_120fov",
                       "camera_cross_right_120fov")
LMINX = LMINY = -32.0
LMAXX = LMAXY = 32.0
SPLIT_H, MAXH, HISTMAX, BEV = 0.2, 100.0, 5.0, 256


class DiffusionDrivePolicy:
    name = "diffusiondrive"
    is_oracle = False
    needs_scenario = False
    needs_sensors = True

    def __init__(self, ckpt: str = CKPT, device: str = "cuda"):
        import numpy as np
        import torch
        self._np, self._torch = np, torch
        self.device = device

        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        sd = {k.replace("agent.", ""): v for k, v in sd.items()}
        anchor = "/tmp/plan_anchor.npy"
        np.save(anchor, sd["_transfuser_model._trajectory_head.plan_anchor"].cpu().numpy())
        from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        cfg = TransfuserConfig()
        cfg.plan_anchor_path = anchor
        self.agent = TransfuserAgent(cfg, lr=1e-4, checkpoint_path=ckpt)
        self.agent.eval().to(device)

    # ── features, all at t0 ─────────────────────────────────────────────────
    def _camera(self, obs):
        np, torch = self._np, self._torch
        imgs = []
        for cam in (CAM_L, CAM_F, CAM_R):
            fr = obs.sensors.frames(cam, [obs.t0_us])
            if not fr:
                return None
            imgs.append(fr[0])
        l, f, r = imgs
        # NAVSIM crops for 1920x1080 — identical to test_one_clip.camera_feature
        l = l[28:-28, 416:-416]; f = f[28:-28]; r = r[28:-28, 416:-416]
        import cv2
        stitched = cv2.resize(np.concatenate([l, f, r], axis=1), (1024, 256))
        return torch.from_numpy(stitched.transpose(2, 0, 1)).float() / 255.0

    def _lidar(self, obs):
        np, torch = self._np, self._torch
        pts = obs.sensors.lidar(obs.t0_us)
        if pts is None or len(pts) == 0:
            return None
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep = (z > SPLIT_H) & (z < MAXH)
        hist, _, _ = np.histogram2d(x[keep], y[keep], bins=BEV,
                                    range=[[LMINX, LMAXX], [LMINY, LMAXY]])
        hist = np.clip(hist, 0, HISTMAX) / HISTMAX
        return torch.from_numpy(hist[None].astype(np.float32))

    def _status(self, obs):
        """NAVSIM 8-dim status: command(4) + body-frame velocity(2) + accel(2).

        The adapter stores world-frame velocity, so it is rotated back into the
        body frame here — the exact inverse of what NvidiaAdapter._ego did.
        """
        torch = self._torch
        e = obs.ego_state
        c, s = math.cos(-e.yaw), math.sin(-e.yaw)
        vx_b = c * e.vx - s * e.vy
        vy_b = s * e.vx + c * e.vy
        cmd = [0.0, 1.0, 0.0, 0.0]              # straight; no route/command available
        return torch.tensor(cmd + [vx_b, vy_b, e.ax, e.ay], dtype=torch.float32)

    # ── policy interface ────────────────────────────────────────────────────
    def plan(self, obs) -> Optional[List[Tuple[float, float]]]:
        torch = self._torch
        if obs.sensors is None:
            return None
        cam, lid, st = self._camera(obs), self._lidar(obs), self._status(obs)
        if cam is None or lid is None:
            return None
        feats = {"camera_feature": cam[None].to(self.device),
                 "lidar_feature": lid[None].to(self.device),
                 "status_feature": st[None].to(self.device)}
        with torch.no_grad():
            out = self.agent.forward(feats)
        traj = out["trajectory"] if isinstance(out, dict) else out
        t = traj.detach().float().cpu().numpy()
        while t.ndim > 2:                       # (B, [modes,] T, 3) -> (T, 3)
            t = t[0]
        # NAVSIM emits 8 poses at 2 Hz over 4 s in the ego frame at t0, which is
        # exactly this harness's grid; resample anyway so a config change cannot
        # silently misalign them.
        out_pts = []
        for k in range(obs.n_steps):
            want = (k + 1) * obs.dt_s
            i = min(len(t) - 1, max(0, int(round(want / 0.5)) - 1))
            out_pts.append((float(t[i][0]), float(t[i][1])))
        return out_pts
