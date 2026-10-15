"""warmup.py — replay the recorded drive for the first seconds of a HUGSIM episode, as AlpaSim does.

AlpaSim forces the recorded (ground-truth) trajectory for `force_gt_duration_us` = 3.0 s before
the policy's plans take effect (`alpasim/runs/*/generated-user-config-0.yaml`); HUGSIM hands the
policy control at t = 0. A client that wants AlpaSim's protocol calls `RecordedPath.plan()`
while `info['timestamp'] < warmup_s` and sends that instead of its own plan: waypoints on the
recorded front-camera path (the ego reference point HUGSIM starts at), 0.5 s apart, in HUGSIM's
plan frame (x right, y forward of the current ego pose).

The recorded path comes from the clip's `data/pai/<short>/meta_data.json` (load_ncore.py output;
the same world frame the HUGS twin was trained and is simulated in, origin = first front-camera
pose). The episode's scene is read from the output directory name, `<scene>_<mode>`.
"""
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RecordedPath:
    def __init__(self, output_dir: str):
        scene = os.path.basename(os.path.normpath(output_dir)).rsplit("_", 2)[0]
        meta = json.load(open(os.path.join(H, "data", "pai", scene[:8], "meta_data.json")))
        front = [f for f in meta["frames"] if "/CAM_FRONT/" in f["rgb_path"]]
        self.t = np.array([f["timestamp"] for f in front], dtype=float)
        self.xz = np.array([[f["camtoworld"][0][3], f["camtoworld"][2][3]] for f in front], dtype=float)

    def position(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.t[0], self.t[-1]))
        return np.array([np.interp(t, self.t, self.xz[:, 0]), np.interp(t, self.t, self.xz[:, 1])])

    def speed(self, t: float, h: float = 0.1) -> float:
        return float(np.linalg.norm(self.position(t + h) - self.position(t - h)) / (2 * h))

    def plan(self, info: dict, n: int, dt: float = 0.5) -> np.ndarray:
        """The recorded motion over the next n x dt seconds, in the plan frame of the current ego pose.

        Waypoints are the recorded displacements from the recorded position at info['timestamp'],
        not targets relative to the ego's own position: when the recording stands still the plan is
        all zeros, which HUGSIM's controller holds exactly. (Targeting the recorded positions made a
        stopped ego chase centimetre offsets and turn on the spot, 12 degrees in 3 s on ac73935a.)
        """
        r1 = float(info["ego_rot"][1])                 # Euler XYZ; heading about the vertical axis
        fwd = np.array([math.sin(r1), math.cos(r1)])   # world (x, z) direction the ego faces
        right = np.array([math.cos(r1), -math.sin(r1)])
        t0 = float(info["timestamp"])
        p0 = self.position(t0)
        pts = []
        for k in range(n):
            d = self.position(t0 + dt * (k + 1)) - p0
            if np.linalg.norm(d) < 0.05:               # standing still: hold exactly
                d = np.zeros(2)
            pts.append([float(d @ right), float(d @ fwd)])
        return np.array(pts, dtype=np.float64)
