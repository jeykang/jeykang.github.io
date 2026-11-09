#!/usr/bin/env python3
"""Downstream-utility test: does adding cosmos3 to the selection surface more
failures of the camera-only consumer model?

PRE-REGISTERED before any outcome was computed (2026-08-11). This is the anchor that
needs no OOD labels and no human panel — it measures the merit the difficulty axis
actually exists to serve: mining clips that break the consumer's camera-only stack.

Independence of the outcome (the thing that makes this test valid):
  the production perceptual axis IS `camera_low_conf`, built with **yolo11x** at
  frame fractions **0.3/0.5/0.7** (planning/camera_perception_runner.py). Scoring the
  outcome with that same detector on those same frames would be scoring the baseline
  against its own input. So the consumer proxy here is deliberately different on both
  axes: **yolov8n** (different family, and a realistic stand-in for the small model a
  scalable camera-only product would deploy) at fractions **0.15/0.45/0.85**.

Outcome definitions, fixed in advance:
  agents_present = behavioral says agents exist (same rule as write_camera_gated.py) —
                   without it, "no detections" on an empty road scores as a failure.
  consumer_conf  = mean over sampled frames of the max detection confidence
                   over AD-relevant classes.
  FAILURE        = agents_present AND consumer_conf < 0.5
                   i.e. the consumer misses a scene that provably contains agents.
  primary   : failure rate within the selected top-N
  secondary : mean consumer_conf within the selected top-N (lower = harder)

Comparison: top-N by PRODUCTION = OR(conflict, max(darkness, camera_gated))
        vs  top-N by PRODUCTION + cosmos3 (noisy-OR, rank-normed) — and, sharper,
        the DISJOINT sets: clips cosmos3 swaps IN vs the clips it pushes OUT.

Usage: ./c3_venv/bin/python downstream_utility.py [--frac 0.10] [--dev 1]
"""
import argparse
import glob
import json
import os
import random
import statistics as st

import cv2
import numpy as np
import pyarrow.parquet as pq

import analyze_axis as A
import cosmos3_scorer as cs

_HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(_HERE, ".cohort_scores.json")
OUTCOME = os.path.join(_HERE, ".cohort_outcome.json")
CONSUMER_W = os.path.join(_HERE, "..", "..", "yolov8n.pt")
FRACS = [0.15, 0.45, 0.85]          # deliberately NOT the axis's 0.3/0.5/0.7
AD = [0, 1, 2, 3, 5, 7, 9, 11]      # person,bicycle,car,motorcycle,bus,truck,light,sign
FAIL_CONF = 0.5
N_BOOT = 10000
SEED = 0

cv2.setNumThreads(2)


def agents_present():
    p = os.path.join(cs.ROOT, ".behavioral", "behavioral_shard_00_of_01.parquet")
    t = pq.read_table(p, columns=["clip_id", "conflict", "vru", "multidir", "closing"]).to_pydict()
    return {c: (cf > 0 or v > 0 or m > 1 or cl > 0) for c, cf, v, m, cl
            in zip(t["clip_id"], t["conflict"], t["vru"], t["multidir"], t["closing"])}


def grab(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in FRACS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def run_consumer(clips, dev):
    from ultralytics import YOLO
    done = json.load(open(OUTCOME)) if os.path.exists(OUTCOME) else {}
    todo = [c for c in clips if c not in done]
    if not todo:
        return done
    print(f"[consumer] yolov8n on {len(todo)} clips (dev {dev}, fracs {FRACS})", flush=True)
    model = YOLO(CONSUMER_W)
    idx = cs.clip_index()
    for k, cid in enumerate(todo):
        try:
            frames = grab(idx[cid])
            if not frames:
                continue
            res = model.predict(frames, conf=0.05, classes=AD, verbose=False, device=dev)
            confs = []
            for r in res:
                c = r.boxes.conf.cpu().numpy()
                confs.append(float(c.max()) if len(c) else 0.0)
            done[cid] = round(float(np.mean(confs)), 4)
        except Exception as e:
            print(f"  [WARN] {cid[:8]}: {str(e)[:60]}", flush=True)
        if (k + 1) % 200 == 0:
            print(f"[consumer] {k+1}/{len(todo)}", flush=True)
            json.dump(done, open(OUTCOME, "w"))
    json.dump(done, open(OUTCOME, "w"))
    return done


def boot_diff(a, b, n=N_BOOT):
    rng = random.Random(SEED)
    ds = []
    for _ in range(n):
        ds.append(st.mean([rng.choice(a) for _ in a]) - st.mean([rng.choice(b) for _ in b]))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10, help="selection fraction (top-N)")
    ap.add_argument("--dev", default="1", help="GPU for the consumer detector")
    a = ap.parse_args()

    c3 = json.load(open(SCORES))
    conflict = A.load_axis(".conflict", "conflict_score")
    camera = A.load_axis(".camera_perception", "low_conf")
    dark = A.load_darkness()
    present = agents_present()
    clips = sorted(c for c in c3 if c in conflict and c in camera and c in dark)
    print(f"[downstream] {len(clips)} clips with cosmos3 + all production axes")

    outcome = run_consumer(clips, a.dev)
    clips = [c for c in clips if c in outcome]

    cf = A.rank_norm([conflict[c] for c in clips])
    perc = A.rank_norm([max(dark[c], camera[c]) for c in clips])
    c3r = A.rank_norm([c3[c] for c in clips])
    prod = [1 - (1 - x) * (1 - y) for x, y in zip(cf, perc)]
    both = [1 - (1 - x) * (1 - y) for x, y in zip(prod, c3r)]

    n = max(1, int(len(clips) * a.frac))
    def topn(scores):
        return set(sorted(range(len(clips)), key=lambda i: -scores[i])[:n])
    Sp, Sb = topn(prod), topn(both)

    def fail(i):
        return 1 if (present.get(clips[i], True) and outcome[clips[i]] < FAIL_CONF) else 0
    def rate(S):
        return sum(fail(i) for i in S) / len(S)
    def conf(S):
        return st.mean(outcome[clips[i]] for i in S)

    base = sum(fail(i) for i in range(len(clips))) / len(clips)
    print(f"\n===== DOWNSTREAM UTILITY (top {a.frac:.0%} = {n} clips, "
          f"consumer=yolov8n@{FRACS}) =====")
    print(f"  cohort base failure rate            {base:.3f}  (n={len(clips)})")
    print(f"  PRODUCTION selection  failure rate  {rate(Sp):.3f}   mean conf {conf(Sp):.3f}")
    print(f"  PRODUCTION+cosmos3    failure rate  {rate(Sb):.3f}   mean conf {conf(Sb):.3f}")
    print(f"  overlap between selections          {len(Sp & Sb)}/{n} "
          f"({len(Sp & Sb)/n:.0%})")

    swapped_in = sorted(Sb - Sp)
    pushed_out = sorted(Sp - Sb)
    print(f"\n  --- the part cosmos3 actually changes ---")
    if swapped_in and pushed_out:
        fi = [fail(i) for i in swapped_in]
        fo = [fail(i) for i in pushed_out]
        lo, hi = boot_diff(fi, fo)
        print(f"  swapped IN  by cosmos3  n={len(fi):3d}  failure {st.mean(fi):.3f}  "
              f"mean conf {st.mean([outcome[clips[i]] for i in swapped_in]):.3f}")
        print(f"  pushed OUT              n={len(fo):3d}  failure {st.mean(fo):.3f}  "
              f"mean conf {st.mean([outcome[clips[i]] for i in pushed_out]):.3f}")
        print(f"  difference {st.mean(fi)-st.mean(fo):+.3f}  bootstrap 95% CI "
              f"[{lo:+.3f}, {hi:+.3f}]  "
              f"{'IMPROVES selection' if lo > 0 else 'DEGRADES selection' if hi < 0 else 'no detectable effect'}")
    else:
        print("  selections identical — cosmos3 changes nothing at this fraction")

    print(f"\n  cosmos3 alone, top {a.frac:.0%}      failure {rate(topn(c3r)):.3f}")
    print(f"  conflict alone, top {a.frac:.0%}     failure {rate(topn(cf)):.3f}")
    print(f"  perceptual alone, top {a.frac:.0%}   failure {rate(topn(perc)):.3f}")


if __name__ == "__main__":
    main()
