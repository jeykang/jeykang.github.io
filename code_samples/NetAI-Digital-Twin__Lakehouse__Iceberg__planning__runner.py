#!/usr/bin/env python3
"""Driving-difficulty scorer — rung 1: constant-velocity open-loop planner.

Detachable module (see MEDALLION_PROGRESS.md §13). Computes a per-clip
"how hard was this to drive" signal as the open-loop error of a *trivial*
constant-velocity planner against the ground-truth ego trajectory:

  at each decision frame, predict the pose 3 s ahead by extrapolating the
  (central-difference) velocity; L2 vs the actual future pose; average over
  the clip. Larger error = the route deviated more from a straight constant-
  speed path = harder to anticipate.

This is the gate-proven rung (Spearman 0.69 vs ego_dynamics — correlated but
NOT redundant; ~half its variance is independent). CPU + NFS only — no GPU, no
catalog, no learned model. Later rungs (learned planner, map-free PDMS
collision/progress) replace/augment this scorer but keep the same output
contract.

Output (file-drop interface, read by edge_case_scorer._load_planning_scores):
  <NFS>/.planning/planning_shard_00_of_01.parquet
  columns: clip_id, planning_score ∈ [0,1], cv_l2_3s_m, n_points, scored_at

Removal: delete this dir + the _load_planning_scores hook + the `planning`
weight in _SCENE_WEIGHTS. Nothing in the core Spark pipeline imports it.

Usage:
    python3 runner.py [--egomotion-root DIR] [--output-dir DIR]
                      [--horizon 3.0] [--scale 10.0] [--workers 32]
                      [--max-clips N]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_EGO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "netai-e2e", "nvidia-physicalai-av-subset", "labels", "egomotion",
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "netai-e2e", "nvidia-physicalai-av-subset", ".planning",
)
STEP = 15      # decision frames ~ every 1 s at 15 Hz
W = 5          # central-difference half-window (~0.33 s) for velocity


def cv_open_loop_l2(path: str, horizon_s: float):
    """Mean L2 (m) @horizon of a constant-velocity open-loop planner for one
    clip. Returns (l2_mean, n_decision_points) or (None, 0) on failure."""
    try:
        d = pq.read_table(path, columns=["timestamp", "x", "y"]).to_pydict()
    except Exception:
        return None, 0
    ts = d["timestamp"]
    if len(ts) < 60:
        return None, 0
    o = sorted(range(len(ts)), key=lambda k: ts[k])
    ts = [d["timestamp"][k] for k in o]
    xs = [d["x"][k] for k in o]
    ys = [d["y"][k] for k in o]
    horizon_us = horizon_s * 1e6
    n = len(ts)
    errs = []
    for i in range(W, n - 1, STEP):
        if i + W >= n:
            break
        dt = (ts[i + W] - ts[i - W]) / 1e6
        if dt <= 0:
            continue
        vx = (xs[i + W] - xs[i - W]) / dt
        vy = (ys[i + W] - ys[i - W]) / dt
        tgt = ts[i] + horizon_us
        if tgt > ts[-1]:
            break
        j = i
        while j < n - 1 and ts[j] < tgt:
            j += 1
        h = (ts[j] - ts[i]) / 1e6
        errs.append(math.hypot(xs[i] + vx * h - xs[j], ys[i] + vy * h - ys[j]))
    if not errs:
        return None, 0
    return st.mean(errs), len(errs)


def clip_id_of(path: str) -> str:
    return os.path.basename(path).split(".", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egomotion-root", default=DEFAULT_EGO)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--scale", type=float, default=10.0,
                    help="planning_score = min(1, cv_l2_3s / scale)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(
        os.path.normpath(args.egomotion_root), "*", "*.parquet")))
    if args.max_clips:
        files = files[:args.max_clips]
    print(f"[planning] {len(files):,} egomotion clips; horizon={args.horizon}s "
          f"scale={args.scale}m workers={args.workers}", flush=True)

    rows = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cv_open_loop_l2, f, args.horizon): f for f in files}
        for fut in futs:
            f = futs[fut]
            l2, npts = fut.result()
            done += 1
            if l2 is not None:
                rows.append({"clip_id": clip_id_of(f),
                             "cv_l2_3s_m": float(l2), "n_points": int(npts)})
            if done % 5000 == 0:
                print(f"[planning] {done:,}/{len(files):,} "
                      f"({done/(time.time()-t0):.0f}/s)", flush=True)

    # planning_score: fixed-scale normalization (absolute, not population-relative)
    for r in rows:
        r["planning_score"] = min(1.0, r["cv_l2_3s_m"] / args.scale)
        r["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "planning_shard_00_of_01.parquet")
    pq.write_table(pa.Table.from_pylist(rows), out)

    ps = [r["planning_score"] for r in rows]
    l2 = [r["cv_l2_3s_m"] for r in rows]
    print(f"[planning] scored {len(rows):,} clips in {time.time()-t0:.0f}s", flush=True)
    if ps:
        print(f"[planning] planning_score: min={min(ps):.3f} max={max(ps):.3f} "
              f"mean={st.mean(ps):.3f} | cv_l2_3s median={st.median(l2):.2f}m", flush=True)
    print(f"[planning] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
