#!/usr/bin/env python3
"""Runs a policy over scenarios and produces MF-PDMS rows.

Responsibilities, deliberately narrow:
  - pick decision times within a clip
  - build Observations that contain NO future data
  - convert the policy's ego-frame trajectory into world poses
  - score with metrics.mf_pdms and aggregate per clip

The no-future guarantee lives here (`_observation` truncates by timestamp), which
is why policies receive an Observation rather than a Scenario.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from metrics import Pose, mf_pdms
from scenario import Observation, Scenario

DECISION_FRACS = (0.3, 0.5, 0.7)   # same convention as the difficulty runners
HORIZON_S = 4.0                    # NAVSIM's 4 s planning horizon
DT_S = 0.5
HISTORY_S = 2.0                    # NAVSIM conditions on 2 s of history


def _observation(sc: Scenario, t0_us: int) -> Optional[Observation]:
    h0 = t0_us - int(HISTORY_S * 1e6)
    ego_hist = [e for e in sc.ego if h0 <= e.t_us <= t0_us]
    if len(ego_hist) < 2:
        return None
    agent_hist = {}
    for k, tr in sc.agents.items():
        past = [b for b in tr if b.t_us <= t0_us]
        if past:
            agent_hist[k] = past
    return Observation(clip_id=sc.clip_id, dataset=sc.dataset, t0_us=t0_us,
                       ego_history=ego_hist, agent_history=agent_hist,
                       ego_length=sc.ego_length, ego_width=sc.ego_width,
                       horizon_s=HORIZON_S, dt_s=DT_S)


def _to_world(obs: Observation, traj) -> List[Pose]:
    """Ego-frame (x fwd, y left) points at t0 -> world poses with derived yaw."""
    e = obs.ego_state
    c, s = math.cos(e.yaw), math.sin(e.yaw)
    pts = [(e.x + c * px - s * py, e.y + s * px + c * py) for px, py in traj]
    poses, prev = [], (e.x, e.y)
    for k, (wx, wy) in enumerate(pts):
        yaw = math.atan2(wy - prev[1], wx - prev[0]) if (wx, wy) != prev else e.yaw
        poses.append(Pose(obs.t0_us + int((k + 1) * obs.dt_s * 1e6), wx, wy, yaw))
        prev = (wx, wy)
    return poses


def human_future(sc: Scenario, t0_us: int, n: int, dt_s: float) -> List[Pose]:
    out = []
    for k in range(n):
        t = t0_us + int((k + 1) * dt_s * 1e6)
        e = sc.ego_at(t)
        if e is None:
            break
        out.append(Pose(t, e.x, e.y, e.yaw))
    return out


def human_future_egoframe(sc: Scenario, t0_us: int, n: int, dt_s: float):
    """Back-compat alias; the implementation lives on Scenario."""
    return sc.future_egoframe(t0_us, n, dt_s)


def history_poses(obs: Observation) -> List[Pose]:
    return [Pose(e.t_us, e.x, e.y, e.yaw) for e in obs.ego_history]


def decision_times(sc: Scenario) -> List[int]:
    """Decision points that leave a full horizon of ground truth after them."""
    t_lo, t_hi = sc.span_us()
    need = int((HORIZON_S + 0.5) * 1e6)
    usable = t_hi - need
    if usable <= t_lo + int(HISTORY_S * 1e6):
        return []
    lo = t_lo + int(HISTORY_S * 1e6)
    return [lo + int(f * (usable - lo)) for f in DECISION_FRACS]


def evaluate_scenario(policy, sc: Scenario) -> Optional[dict]:
    """Mean MF-PDMS over this clip's decision times."""
    rows = []
    for t0 in decision_times(sc):
        obs = _observation(sc, t0)
        if obs is None:
            continue
        human = human_future(sc, t0, obs.n_steps, obs.dt_s)
        if len(human) < obs.n_steps:
            continue
        traj = (policy.plan(obs, sc) if getattr(policy, "needs_scenario", False)
                else policy.plan(obs))
        if traj is None or len(traj) < obs.n_steps:
            continue
        poses = _to_world(obs, list(traj)[:obs.n_steps])
        rows.append(mf_pdms(poses, human, history_poses(obs), sc.agents,
                            sc.ego_length, sc.ego_width))
    if not rows:
        return None
    keys = ("nc", "ttc", "ep", "hc", "ec", "mf_pdms")
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out.update(clip_id=sc.clip_id, dataset=sc.dataset, policy=policy.name,
               n_decisions=len(rows), n_tracks=len(sc.agents),
               is_oracle=bool(getattr(policy, "is_oracle", False)),
               horizon_s=HORIZON_S, dt_s=DT_S)
    return out


def _score_one(cid: str) -> Optional[dict]:
    try:
        sc = _W["adapter"].load(cid)
    except Exception as e:
        print(f"  [WARN] load {cid[:8]}: {str(e)[:70]}", flush=True)
        return None
    return evaluate_scenario(_W["policy"], sc) if sc is not None else None


_W: dict = {}


def _init_worker(adapter, policy):
    _W["adapter"], _W["policy"] = adapter, policy


def evaluate(policy, adapter, clip_ids: Sequence[str], progress_every: int = 200,
             workers: int = 1) -> List[dict]:
    """Score every clip. Loading is NFS-I/O-bound (~0.9 s/clip) while scoring is
    ~0.016 s/clip, so `workers` is almost pure speedup up to the mount's limit."""
    rows: List[dict] = []
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=(adapter, policy)) as pool:
            for i, r in enumerate(pool.imap_unordered(_score_one, clip_ids, chunksize=4)):
                if r:
                    rows.append(r)
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"[eval] {i+1}/{len(clip_ids)} clips, {len(rows)} scored", flush=True)
        return rows
    _init_worker(adapter, policy)
    for i, cid in enumerate(clip_ids):
        r = _score_one(cid)
        if r:
            rows.append(r)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[eval] {i+1}/{len(clip_ids)} clips, {len(rows)} scored", flush=True)
    return rows
