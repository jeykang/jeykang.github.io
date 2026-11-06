#!/usr/bin/env python3
"""Map-free PDMS (MF-PDMS) — the sub-metrics computable without an HD map.

**This is NOT EPDMS and must not be reported as it.** NAVSIM's EPDMS has nine
sub-metrics; four need map layers we do not have (see FEASIBILITY.md §3):
    DAC  drivable-area compliance   multiplier   needs drivable area
    DDC  driving-direction          multiplier   needs lane direction
    TLC  traffic-light compliance   multiplier   needs light states
    LK   lane keeping               weighted     needs lane graph
Because DAC/DDC/TLC are *multipliers*, dropping them does not degrade EPDMS — it
makes it undefined. MF-PDMS keeps EPDMS's structure and the five map-free terms:

    MF-PDMS = NC * (5*TTC + 5*EP + 2*HC + 2*EC) / 14

with weights inherited from EPDMS (LK's weight of 2 is removed from the
denominator). A score is comparable to other MF-PDMS scores, never to a published
EPDMS number.

Everything here consumes world-frame poses and boxes only, so it is dataset-
agnostic by construction.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from scenario import AgentBox, EgoState

# nuPlan/NAVSIM-derived comfort bounds. dt is 0.5 s here, so jerk is a coarse
# finite difference — treat EC as an indicator, not a ride-quality measurement.
MAX_LON_ACCEL = 2.40
MAX_LON_DECEL = 4.05
MAX_LAT_ACCEL = 4.89
MAX_JERK = 8.37
MAX_YAW_RATE = 0.95

TTC_HORIZON_S = 1.0     # constant-velocity projection used for the TTC check
TTC_DT = 0.2
COLLISION_BUFFER = 0.0  # metres added to both footprints; 0 = geometric contact


@dataclass
class Pose:
    t_us: int
    x: float
    y: float
    yaw: float


# ── geometry ────────────────────────────────────────────────────────────────
def _corners(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def obb_overlap(a, b) -> bool:
    """Separating-axis test on two convex quads. Exact for oriented boxes."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ax, ay = -(y2 - y1), (x2 - x1)
            pa = [ax * px + ay * py for px, py in a]
            pb = [ax * px + ay * py for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _boxes_hit(ex, ey, eyaw, el, ew, agent: AgentBox) -> bool:
    # Broad phase: bounding-circle rejection before the exact SAT. Most agents in a
    # scene are tens of metres away, so this removes almost all of the OBB work.
    r = 0.5 * (math.hypot(el, ew) + math.hypot(agent.length, agent.width)) + COLLISION_BUFFER
    if (ex - agent.x) ** 2 + (ey - agent.y) ** 2 > r * r:
        return False
    return obb_overlap(
        _corners(ex, ey, eyaw, el + COLLISION_BUFFER, ew + COLLISION_BUFFER),
        _corners(agent.x, agent.y, agent.yaw, agent.length, agent.width))


def agent_at(track: Sequence[AgentBox], t_us: int, win_us: int = 200_000) -> Optional[AgentBox]:
    """Nearest box within win_us. Binary search — tracks are sorted, and this is
    called once per (pose, track, ttc-substep), so a linear min() dominated runtime."""
    if not track:
        return None
    i = bisect.bisect_left(_TrackKeys(track), t_us)
    best = None
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(track):
            d = abs(track[j].t_us - t_us)
            if best is None or d < best[0]:
                best = (d, track[j])
    return best[1] if best and best[0] <= win_us else None


class _TrackKeys(Sequence):
    """Timestamp view over a track, so bisect can search without materialising."""
    __slots__ = ("_t",)

    def __init__(self, track):
        self._t = track

    def __len__(self):
        return len(self._t)

    def __getitem__(self, i):
        return self._t[i].t_us


def _agent_velocity(track: Sequence[AgentBox], t_us: int) -> tuple:
    """Finite-difference world velocity of a track around t_us."""
    if len(track) < 2:
        return (0.0, 0.0)
    i = min(range(len(track)), key=lambda k: abs(track[k].t_us - t_us))
    j = i + 1 if i + 1 < len(track) else i - 1
    dt = (track[j].t_us - track[i].t_us) / 1e6
    if abs(dt) < 1e-6:
        return (0.0, 0.0)
    return ((track[j].x - track[i].x) / dt, (track[j].y - track[i].y) / dt)


# ── sub-metrics ─────────────────────────────────────────────────────────────
def no_collision(poses: List[Pose], agents: Dict[str, List[AgentBox]],
                 el: float, ew: float) -> tuple:
    """NC multiplier in {0,1}, plus a diagnostic dict.

    At-fault approximation: a contact is NOT counted against the ego when the
    agent sits behind the ego's rear axle line at contact (i.e. it struck us from
    behind). Without a map or right-of-way we cannot do NAVSIM's full at-fault
    logic, so this is deliberately the conservative, well-defined part of it.
    """
    for p in poses:
        for track in agents.values():
            a = agent_at(track, p.t_us)
            if a is None:
                continue
            if not _boxes_hit(p.x, p.y, p.yaw, el, ew, a):
                continue
            c, s = math.cos(-p.yaw), math.sin(-p.yaw)
            dx = c * (a.x - p.x) - s * (a.y - p.y)      # agent in ego frame
            if dx < -el / 2.0:
                continue                                # struck from behind -> not at fault
            return 0.0, {"collision_t_us": p.t_us, "collision_track": a.track_id,
                         "collision_vru": a.is_vru}
    return 1.0, {}


def time_to_collision(poses: List[Pose], agents: Dict[str, List[AgentBox]],
                      el: float, ew: float) -> float:
    """TTC in {0,1}: 0 if a constant-velocity projection from any planned pose
    collides within TTC_HORIZON_S. Ego is projected along its own planned path."""
    n = len(poses)
    for i, p in enumerate(poses):
        if i + 1 < n:
            dt = max(1e-3, (poses[i + 1].t_us - p.t_us) / 1e6)
            evx, evy = (poses[i + 1].x - p.x) / dt, (poses[i + 1].y - p.y) / dt
        else:
            evx = evy = 0.0
        steps = int(TTC_HORIZON_S / TTC_DT)
        for track in agents.values():
            a = agent_at(track, p.t_us)
            if a is None:
                continue
            avx, avy = _agent_velocity(track, p.t_us)
            for k in range(1, steps + 1):
                tau = k * TTC_DT
                ex, ey = p.x + evx * tau, p.y + evy * tau
                ghost = AgentBox(a.t_us, a.track_id, a.x + avx * tau, a.y + avy * tau,
                                 a.yaw, a.length, a.width, a.label)
                if _boxes_hit(ex, ey, p.yaw, el, ew, ghost):
                    return 0.0
    return 1.0


def ego_progress(poses: List[Pose], human: List[Pose], min_progress_m: float = 1.0) -> float:
    """EP in [0,1]: planned path length over the human's, on the same horizon.

    Map-free substitute for NAVSIM's route-projected progress. If the human barely
    moved (stopped at a light), progress is not informative and EP is 1.0 —
    otherwise every policy would be rewarded for creeping forward at a red light.
    """
    def plen(ps):
        return sum(math.hypot(ps[i + 1].x - ps[i].x, ps[i + 1].y - ps[i].y)
                   for i in range(len(ps) - 1))
    h = plen(human)
    if h < min_progress_m:
        return 1.0
    return max(0.0, min(1.0, plen(poses) / h))


def _polyfit(ts: List[float], ys: List[float], deg: int) -> List[float]:
    """Least-squares polynomial fit via normal equations. Stdlib only.

    Returns coefficients c with y ~ sum(c[k] * t**k). Gaussian elimination on a
    (deg+1) system is ample here: deg <= 3 and the design matrix is tiny.
    """
    n = deg + 1
    A = [[sum(t ** (i + j) for t in ts) for j in range(n)] + [sum(y * t ** i for t, y in zip(ts, ys))]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            return [0.0] * n
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    return [A[i][n] for i in range(n)]


def _kinematics(poses: List[Pose]) -> dict:
    """Peak |lon accel|, |lat accel|, |jerk| and |yaw rate| over a pose list.

    Derivatives come from a least-squares CUBIC fit to the speed and heading
    profiles rather than from raw finite differences. That matters: differentiating
    sampled position three times amplifies pose noise by 1/dt^3, and the naive
    version scored the *recorded human trajectory* at EC=0.62 — i.e. it was
    measuring differentiation noise, not ride quality. The oracle policy exists to
    catch exactly this class of bug.
    """
    if len(poses) < 4:
        return {"lon_accel": 0.0, "lat_accel": 0.0, "jerk": 0.0, "yaw_rate": 0.0}
    t0 = poses[0].t_us
    ts, sp, yaw = [], [], []
    for i in range(len(poses) - 1):
        dt = max(1e-3, (poses[i + 1].t_us - poses[i].t_us) / 1e6)
        ts.append((poses[i].t_us - t0) / 1e6 + dt / 2.0)
        sp.append(math.hypot(poses[i + 1].x - poses[i].x, poses[i + 1].y - poses[i].y) / dt)
        h = math.atan2(poses[i + 1].y - poses[i].y, poses[i + 1].x - poses[i].x)
        if yaw:                                        # unwrap
            h = yaw[-1] + ((h - yaw[-1] + math.pi) % (2 * math.pi) - math.pi)
        yaw.append(h)

    deg = 3 if len(ts) >= 5 else 2
    cs = _polyfit(ts, sp, deg)
    cy = _polyfit(ts, yaw, min(deg, 2))

    def dv(c, t, order):
        out = 0.0
        for k in range(order, len(c)):
            f = 1.0
            for m in range(order):
                f *= (k - m)
            out += c[k] * f * t ** (k - order)
        return out

    accs = [dv(cs, t, 1) for t in ts]
    jrks = [dv(cs, t, 2) for t in ts]
    yrs = [dv(cy, t, 1) for t in ts]
    lat = [abs(s * y) for s, y in zip(sp, yrs)]
    return {"lon_accel": max(abs(a) for a in accs), "lat_accel": max(lat),
            "jerk": max(abs(j) for j in jrks), "yaw_rate": max(abs(y) for y in yrs)}


def extended_comfort(poses: List[Pose]) -> float:
    """EC in [0,1]: fraction of absolute comfort bounds respected."""
    k = _kinematics(poses)
    checks = [k["lon_accel"] <= max(MAX_LON_ACCEL, MAX_LON_DECEL),
              k["lat_accel"] <= MAX_LAT_ACCEL,
              k["jerk"] <= MAX_JERK,
              k["yaw_rate"] <= MAX_YAW_RATE]
    return sum(checks) / len(checks)


def history_comfort(poses: List[Pose], history: List[Pose]) -> float:
    """HC in [0,1]: comfort relative to how this clip was actually driven.

    A trajectory that is aggressive by absolute standards but no more aggressive
    than the recorded human on the same road should not be punished — that is what
    NAVSIM's history-relative comfort is for. Bound = max(absolute, 1.5x history).
    """
    if len(history) < 3:
        return extended_comfort(poses)
    h = _kinematics(history)
    k = _kinematics(poses)
    checks = [k["lon_accel"] <= max(MAX_LON_ACCEL, MAX_LON_DECEL, 1.5 * h["lon_accel"]),
              k["lat_accel"] <= max(MAX_LAT_ACCEL, 1.5 * h["lat_accel"]),
              k["jerk"] <= max(MAX_JERK, 1.5 * h["jerk"]),
              k["yaw_rate"] <= max(MAX_YAW_RATE, 1.5 * h["yaw_rate"])]
    return sum(checks) / len(checks)


WEIGHTS = {"ttc": 5.0, "ep": 5.0, "hc": 2.0, "ec": 2.0}


def mf_pdms(poses: List[Pose], human: List[Pose], history: List[Pose],
            agents: Dict[str, List[AgentBox]], el: float, ew: float) -> dict:
    """All five sub-metrics plus the aggregate, as a flat dict."""
    nc, diag = no_collision(poses, agents, el, ew)
    sub = {
        "nc": nc,
        "ttc": time_to_collision(poses, agents, el, ew),
        "ep": ego_progress(poses, human),
        "hc": history_comfort(poses, history),
        "ec": extended_comfort(poses),
    }
    weighted = sum(WEIGHTS[k] * sub[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    sub["mf_pdms"] = sub["nc"] * weighted
    sub.update(diag)
    return sub
