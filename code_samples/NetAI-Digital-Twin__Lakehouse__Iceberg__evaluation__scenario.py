#!/usr/bin/env python3
"""Dataset-agnostic scenario types and the adapter contract.

This is the seam that makes the evaluation pipeline multi-dataset. Everything
downstream (metrics, harness, policies) sees only these types; nothing downstream
imports a dataset module. Supporting a new dataset means writing one adapter that
produces `Scenario` objects — no changes to metrics or the harness.

Frame convention (the one thing an adapter must get right):
  All positions are in a single per-clip **world/odometry frame**, metres, with
  yaw in radians CCW from +x. Ego poses and agent boxes must be in the SAME frame.
  Datasets that store agents in a per-timestamp sensor/rig frame (NVIDIA PhysicalAI
  does) must lift them to world using the ego pose at that agent's reference
  timestamp — see `adapters.NvidiaAdapter`.

Time is integer microseconds on a single per-clip clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class EgoState:
    t_us: int
    x: float
    y: float
    yaw: float
    vx: float          # world-frame velocity
    vy: float


@dataclass(frozen=True)
class AgentBox:
    t_us: int
    track_id: str
    x: float
    y: float
    yaw: float
    length: float
    width: float
    label: str

    @property
    def is_vru(self) -> bool:
        return self.label in VRU_CLASSES


VRU_CLASSES = {"person", "pedestrian", "rider", "cyclist", "bicycle",
               "motorcycle", "stroller", "animal"}


@dataclass
class Scenario:
    """One clip: ego trajectory, agent tracks, and the ego footprint."""
    clip_id: str
    dataset: str
    ego: List[EgoState]                       # sorted by t_us
    agents: Dict[str, List[AgentBox]]         # track_id -> sorted boxes
    ego_length: float = 4.87
    ego_width: float = 2.12
    meta: dict = field(default_factory=dict)

    def ego_at(self, t_us: int) -> Optional[EgoState]:
        if not self.ego:
            return None
        return min(self.ego, key=lambda e: abs(e.t_us - t_us))

    def span_us(self) -> Tuple[int, int]:
        return (self.ego[0].t_us, self.ego[-1].t_us) if self.ego else (0, 0)

    def future_egoframe(self, t0_us: int, n: int, dt_s: float) -> List[Tuple[float, float]]:
        """The recorded future as a policy would return it: ego frame at t0.

        Lives on Scenario (not the harness) so the oracle policy needs no closure
        over an adapter — which keeps policies picklable for multiprocessing.
        """
        import math
        e0 = self.ego_at(t0_us)
        if e0 is None:
            return [(0.0, 0.0)] * n
        c, s_ = math.cos(-e0.yaw), math.sin(-e0.yaw)
        out = []
        for k in range(n):
            e = self.ego_at(t0_us + int((k + 1) * dt_s * 1e6))
            if e is None:
                break
            dx, dy = e.x - e0.x, e.y - e0.y
            out.append((c * dx - s_ * dy, s_ * dx + c * dy))
        while len(out) < n:
            out.append(out[-1] if out else (0.0, 0.0))
        return out


@dataclass
class Observation:
    """What a policy is allowed to see. Nothing here may come from after `t0_us`.

    Keeping the future out of the observation is the whole point of the type: a
    policy that accidentally reads ground-truth future would score ~1.0 and look
    like a triumph. `harness` builds these by truncation, and `policies.ReplayHuman`
    is the *deliberate* oracle that bypasses it, as a calibration reference.
    """
    clip_id: str
    dataset: str
    t0_us: int
    ego_history: List[EgoState]               # up to and including t0
    agent_history: Dict[str, List[AgentBox]]  # up to and including t0
    ego_length: float
    ego_width: float
    horizon_s: float
    dt_s: float

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_s / self.dt_s))

    @property
    def ego_state(self) -> EgoState:
        return self.ego_history[-1]


class DatasetAdapter(Protocol):
    """Implement this to plug a new dataset into the evaluation pipeline.

    Required source data is deliberately minimal — agent tracks and ego poses.
    Every AV dataset has both; no maps, no sensor data, no calibration beyond the
    ego footprint. That is what keeps Tier 1 dataset-agnostic (see FEASIBILITY.md).
    """

    name: str

    def list_clips(self) -> Sequence[str]:
        """Clip ids this adapter can serve."""
        ...

    def load(self, clip_id: str) -> Optional[Scenario]:
        """Build a Scenario, or None if the clip lacks required data."""
        ...
