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
import os
import time
from typing import Dict, List, Optional, Sequence

from metrics import Pose, mf_pdms
from scenario import Observation, Scenario

DECISION_FRACS = (0.3, 0.5, 0.7)   # same convention as the difficulty runners
# Horizon is configurable because policies disagree: Alpamayo emits 6.4 s at 10 Hz
# (resampled to this grid), while VaVAM's action expert emits exactly 6 steps at 2 Hz
# = 3.0 s. Rather than extrapolate a model's output past what it predicts, run the
# whole ladder at the shorter horizon and report that as its own table.
HORIZON_S = float(os.environ.get("EVAL_HORIZON_S", "4.0"))
DT_S = float(os.environ.get("EVAL_DT_S", "0.5"))
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
    sensors = None
    if sc.sensor_factory is not None:
        try:
            sensors = sc.sensor_factory(sc.clip_id, t0_us)   # bounded to the decision time
        except Exception:
            sensors = None
    return Observation(clip_id=sc.clip_id, dataset=sc.dataset, t0_us=t0_us,
                       ego_history=ego_hist, agent_history=agent_hist,
                       ego_length=sc.ego_length, ego_width=sc.ego_width,
                       horizon_s=HORIZON_S, dt_s=DT_S, sensors=sensors)


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


def decision_times(sc: Scenario, require_sensors: bool = False) -> List[int]:
    """Decision points inside the window where the data actually supports scoring.

    The window is the INTERSECTION of ego, agent and (optionally) sensor coverage —
    not the ego span. On this dataset egomotion runs to ~140 s while obstacle labels
    and video stop at ~20 s, so using the ego span put 89% of decision points in a
    region with no annotated agents, where every collision metric is trivially
    perfect. Intersecting is what makes a score mean something.
    """
    t_lo, t_hi = sc.span_us()
    a = sc.agent_span_us()
    if a is None:
        return []
    t_lo, t_hi = max(t_lo, a[0]), min(t_hi, a[1])
    if require_sensors:
        sp = sc.sensor_span_us()
        if sp is None:
            return []
        t_lo, t_hi = max(t_lo, sp[0]), min(t_hi, sp[1])
    need = int((HORIZON_S + 0.5) * 1e6)
    usable = t_hi - need
    lo = t_lo + int(HISTORY_S * 1e6)
    if usable <= lo:
        return []
    return [lo + int(f * (usable - lo)) for f in DECISION_FRACS]


def evaluate_scenario(policy, sc: Scenario) -> Optional[dict]:
    """Mean MF-PDMS over this clip's decision times."""
    t_start = time.time()
    rows, plan_s = [], []
    for t0 in decision_times(sc, require_sensors=getattr(policy, "needs_sensors", False)):
        obs = _observation(sc, t0)
        if obs is None:
            continue
        human = human_future(sc, t0, obs.n_steps, obs.dt_s)
        if len(human) < obs.n_steps:
            continue
        _tp = time.time()
        traj = (policy.plan(obs, sc) if getattr(policy, "needs_scenario", False)
                else policy.plan(obs))
        plan_s.append(time.time() - _tp)
        if traj is None or len(traj) < obs.n_steps:
            continue
        poses = _to_world(obs, list(traj)[:obs.n_steps])
        rows.append(mf_pdms(poses, human, history_poses(obs), sc.agents,
                            sc.ego_length, sc.ego_width))
    if not rows:
        return None
    keys = ("nc", "ttc", "ep", "hc", "ec", "mf_pdms")
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    total = time.time() - t_start
    out.update(wall_s=round(total, 4),
               plan_s=round(sum(plan_s), 4),
               # everything that is not the policy: sensor decode, label reads, metrics
               overhead_s=round(max(0.0, total - sum(plan_s)), 4),
               clip_id=sc.clip_id, dataset=sc.dataset, policy=policy.name,
               n_decisions=len(rows), n_tracks=len(sc.agents),
               is_oracle=bool(getattr(policy, "is_oracle", False)),
               horizon_s=HORIZON_S, dt_s=DT_S)
    return out


def _score_one(cid: str) -> Optional[dict]:
    _t = time.time()
    try:
        sc = _W["adapter"].load(cid)
    except Exception as e:
        print(f"  [WARN] load {cid[:8]}: {str(e)[:70]}", flush=True)
        return None
    load_s = time.time() - _t
    if sc is None:
        return None
    row = evaluate_scenario(_W["policy"], sc)
    if row is not None:
        row["scenario_load_s"] = round(load_s, 4)
    return row


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
