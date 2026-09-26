# SPDX-License-Identifier: Apache-2.0
"""Run this project's `Policy` objects inside AlpaSim, unmodified.

Purpose is validation, not deployment: MF-PDMS is open-loop and dataset-agnostic by
design, and AlpaSim is closed-loop and locked to NuRec scenes. Driving the SAME policy
through both is what yields a per-clip correlation between the cheap metric and a
closed-loop score — the external anchor MF-PDMS otherwise lacks (see ../ALPASIM.md).

Implemented as an AlpaSim **plugin** (entry-point `alpasim.models`), so nothing in the
AlpaSim checkout is modified and nothing here depends on AlpaSim internals beyond the
documented `BaseTrajectoryModel` contract.

IMPORTANT — what does and does not transfer:
  AlpaSim's `PredictionInput` carries cameras, command, speed, acceleration and ego
  pose history. It deliberately does NOT carry ground-truth agent boxes, because in
  closed-loop the policy is expected to perceive them. Our `Observation` does carry
  `agent_history`. So:
    * ego-only policies (stationary, constant_velocity, constant_turn_rate) transfer
      exactly;
    * camera policies (VaVAM, Alpamayo) transfer exactly;
    * track-consuming policies (reactive_idm) CANNOT transfer faithfully — they are
      given an empty agent set here and will behave as their track-free fallback.
      They are excluded rather than silently degraded; see HARNESS_POLICY_DENY.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig

logger = logging.getLogger(__name__)

# Policies that read ground-truth agent tracks cannot be run faithfully here.
HARNESS_POLICY_DENY = {"reactive_idm", "replay_human"}


class _AlpaSimSensorReader:
    """Adapts AlpaSim's decoded frames to the harness SensorReader contract.

    The harness reader is history-bounded by construction; AlpaSim only ever hands
    over frames at or before the decision time, so the bound is satisfied inherently.
    """

    def __init__(self, camera_images, t0_us: int):
        self._by_cam = camera_images
        self.max_t_us = t0_us

    def available(self):
        return list(self._by_cam)

    def frame_times(self, camera: str, t_us):
        frames = self._by_cam.get(camera) or []
        if not frames:
            return []
        return [min(frames, key=lambda f: abs(f.timestamp_us - t)).timestamp_us
                for t in t_us]

    def frames(self, camera: str, t_us):
        frames = self._by_cam.get(camera) or []
        if not frames:
            return []
        out = []
        for t in t_us:
            f = min(frames, key=lambda fr: abs(fr.timestamp_us - t))
            img = f.image
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            out.append(np.asarray(img, dtype=np.uint8))
        return out

    def lidar(self, t_us: int, sensor: str = "lidar_top_360fov"):
        return None            # AlpaSim's NuRec path renders cameras only


class HarnessPolicyModel(BaseTrajectoryModel):
    """AlpaSim model that delegates to a harness `Policy`.

    Selected with `model_cfg.checkpoint_path` holding the policy spec, either a
    builtin name (`constant_velocity`) or `module:Class` (`policy_vavam:VaVAMPolicy`),
    matching run_eval.py's `--policy` syntax exactly so the same string names the same
    policy in both systems.
    """

    def __init__(self, policy_spec: str, device: torch.device,
                 camera_ids: list[str], context_length: int,
                 output_frequency_hz: int):
        import sys
        harness_dir = os.environ.get("HARNESS_DIR")
        if harness_dir and harness_dir not in sys.path:
            sys.path.insert(0, harness_dir)
        self._policy = self._build_policy(policy_spec)
        self._device = device
        self._camera_ids = list(camera_ids)
        self._context_length = int(context_length or 1)
        self._freq = int(output_frequency_hz or 2)
        logger.info("HarnessPolicyModel wrapping %s", getattr(self._policy, "name", policy_spec))

    @staticmethod
    def _build_policy(spec: str):
        import policies as P
        name = spec.strip()
        if name in HARNESS_POLICY_DENY:
            raise ValueError(
                f"policy '{name}' consumes ground-truth agent tracks, which AlpaSim "
                "does not provide; it cannot be validated closed-loop. See module docstring.")
        if name in P.BUILTIN:
            return P.BUILTIN[name]()
        if ":" not in name:
            raise ValueError(f"unknown policy '{name}'; use a builtin or module:Class")
        mod, cls = name.split(":", 1)
        return getattr(importlib.import_module(mod), cls)()

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, device: torch.device,
                    camera_ids: list[str], context_length: int | None,
                    output_frequency_hz: int) -> "HarnessPolicyModel":
        spec = os.environ.get("HARNESS_POLICY") or model_cfg.checkpoint_path
        return cls(spec, device, camera_ids, context_length or 1, output_frequency_hz)

    def _encode_command(self, command: DriveCommand) -> Any:
        # The harness Observation has no command channel; policies assume STRAIGHT.
        return int(command)

    def _observation(self, pi: PredictionInput):
        """PredictionInput -> harness Observation, with NO agent tracks (see docstring)."""
        from scenario import EgoState, Observation

        hist = []
        for p in pi.ego_pose_history:
            t_us = int(getattr(p, "timestamp_us", 0))
            pos = getattr(p, "position", None)
            rot = getattr(p, "rotation", None)
            x, y, z = (float(pos[0]), float(pos[1]), float(pos[2])) if pos is not None else (0.0, 0.0, 0.0)
            yaw = float(np.arctan2(rot[1][0], rot[0][0])) if rot is not None else 0.0
            hist.append(EgoState(t_us, x, y, yaw, 0.0, 0.0, z))
        if not hist:
            raise ValueError("AlpaSim supplied an empty ego_pose_history")

        # speed/acceleration arrive as scalars in the rig frame; project onto heading
        e = hist[-1]
        vx = float(pi.speed) * float(np.cos(e.yaw))
        vy = float(pi.speed) * float(np.sin(e.yaw))
        hist[-1] = EgoState(e.t_us, e.x, e.y, e.yaw, vx, vy, e.z,
                            e.qx, e.qy, e.qz, e.qw, float(pi.acceleration), 0.0)

        t0 = hist[-1].t_us
        horizon_s = float(os.environ.get("EVAL_HORIZON_S", "4.0"))
        return Observation(
            clip_id=os.environ.get("ALPASIM_SCENE", "alpasim"),
            dataset="alpasim_nurec", t0_us=t0,
            ego_history=hist, agent_history={},          # deliberately empty
            ego_length=4.87, ego_width=2.12,
            horizon_s=horizon_s, dt_s=1.0 / self._freq,
            sensors=_AlpaSimSensorReader(pi.camera_images, t0),
        )

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        obs = self._observation(prediction_input)
        traj = self._policy.plan(obs)
        n = obs.n_steps
        if traj is None:
            traj = [(0.0, 0.0)] * n                      # refuse to move rather than guess
        pos = np.zeros((1, n, 3), dtype=np.float32)
        for k, (x, y) in enumerate(list(traj)[:n]):
            pos[0, k, 0] = x
            pos[0, k, 1] = y
        rot = np.tile(np.eye(3, dtype=np.float32), (1, n, 1, 1))
        return ModelPrediction(candidate_positions=pos, candidate_rotations=rot,
                               selected_index=0)

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._freq
