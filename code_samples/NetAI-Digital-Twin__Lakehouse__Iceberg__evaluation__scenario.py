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
    # Full 3D pose. Planar metrics use x/y/yaw only, but real driving models want
    # the whole pose (Alpamayo consumes 3-D positions and 3x3 rotations), so an
    # adapter that has it should supply it. Defaults keep 2-D adapters valid.
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    # Body-frame linear acceleration. Some planners (DiffusionDrive/Transfuser's
    # 8-dim status vector) consume it directly, and finite-differencing velocity
    # is a poor substitute when the dataset records it.
    ax: float = 0.0
    ay: float = 0.0


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
    # Adapter-supplied factory: (clip_id, max_t_us) -> SensorReader | None. Held as
    # a callable rather than an open handle so a Scenario stays picklable for the
    # multiprocessing pool; readers are built inside the worker, per decision point.
    sensor_factory: Optional[object] = None
    # Lazy () -> (t_min_us, t_max_us) for sensor coverage. Lazy because it costs an
    # extra file read and only sensor-consuming policies need it.
    sensor_span_fn: Optional[object] = None

    def ego_at(self, t_us: int) -> Optional[EgoState]:
        if not self.ego:
            return None
        return min(self.ego, key=lambda e: abs(e.t_us - t_us))

    def span_us(self) -> Tuple[int, int]:
        return (self.ego[0].t_us, self.ego[-1].t_us) if self.ego else (0, 0)

    def agent_span_us(self) -> Optional[Tuple[int, int]]:
        """Time range actually covered by agent annotations.

        NOT the same as the ego span: egomotion here runs to ~140 s while obstacle
        labels and video cover only the first ~20 s. Scoring outside this window
        silently evaluates against an empty scene, so every collision-based metric
        comes back perfect and the benchmark looks meaningless-but-fine.
        """
        ts = [b.t_us for tr in self.agents.values() for b in tr]
        return (min(ts), max(ts)) if ts else None

    def sensor_span_us(self) -> Optional[Tuple[int, int]]:
        if self.sensor_span_fn is None:
            return None
        try:
            return self.sensor_span_fn()
        except Exception:
            return None

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
    sensors: Optional["SensorReader"] = None   # None when the dataset has no sensors

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon_s / self.dt_s))

    @property
    def ego_state(self) -> EgoState:
        return self.ego_history[-1]


class SensorReader(Protocol):
    """History-bounded access to raw sensor data for a single decision point.

    Track-and-pose metrics need no sensors, but a real driving policy is a vision
    model, so the Observation has to be able to carry pixels — otherwise the
    "plug in your model" contract only admits planners that already have
    perception. A reader is constructed by the adapter and bound to `max_t_us`
    (= the decision time), and MUST refuse requests past it: that is what keeps
    the no-future guarantee true once sensors are in play.

    Optional: adapters without sensor data simply do not provide one, and
    `Observation.sensors` stays None.
    """

    def available(self) -> Sequence[str]:
        """Sensor names this reader can serve, e.g. camera ids."""
        ...

    def frames(self, camera: str, t_us: Sequence[int]):
        """Nearest frame at or before each timestamp, as HxWx3 uint8 RGB arrays."""
        ...


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

    def sensor_reader(self, clip_id: str, max_t_us: int) -> Optional["SensorReader"]:
        """Optional. Return a reader bounded to `max_t_us`, or None if unsupported."""
        ...
