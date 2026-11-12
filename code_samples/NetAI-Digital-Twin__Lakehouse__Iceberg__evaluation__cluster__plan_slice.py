#!/usr/bin/env python3
"""Choose an evaluation slice and emit a manifest — runs locally, needs NO dataset.

The NVIDIA dataset is published on HuggingFace as one ZIP per (sensor, chunk), not
per clip, so a slice is selected at CHUNK granularity: pulling a chunk costs ~5.5 GB
(4 cameras + labels) and yields ~100 clips. That is why this picks a few clips from
each of several chunks rather than N clips at random — a random draw over 33k clips
would touch ~60 different chunks and cost ~330 GB for 60 clips.

Spreading across several chunks matters for representativeness: a chunk is one
collection batch, so a single-chunk slice is correlated in place and time. It is fine
for comparing models on identical clips, and NOT a random sample of the cohort — say
so when reporting.

Uses only the cached clip index (`planning/cosmos3_reason/.clip_index.json`), so it
works while the NFS mount is down.

    python plan_slice.py --chunks 6 --per-chunk 10 --out slice_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "..", "..", "planning", "cosmos3_reason", ".clip_index.json")
CAM4 = ["camera_cross_left_120fov", "camera_front_wide_120fov",
        "camera_cross_right_120fov", "camera_front_tele_30fov"]
CAM7 = CAM4 + ["camera_rear_left_70fov", "camera_rear_right_70fov",
               "camera_rear_tele_30fov"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--per-chunk", type=int, default=10)
    ap.add_argument("--cameras", type=int, choices=(4, 7), default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", default=os.path.join(HERE, "slice_manifest.json"))
    a = ap.parse_args()

    idx = json.load(open(a.index))
    by_chunk = collections.defaultdict(list)
    for cid, path in idx.items():
        by_chunk[path.split("chunk_")[1][:4]].append(cid)

    # prefer well-populated chunks, then spread the picks evenly across the dataset
    dense = sorted((c for c in by_chunk if len(by_chunk[c]) >= a.per_chunk), key=int)
    if not dense:
        raise SystemExit("no chunk holds enough clips")
    step = max(1, len(dense) // a.chunks)
    chosen = dense[::step][:a.chunks]

    rng = random.Random(a.seed)
    clips, per = [], {}
    for ch in chosen:
        pool = sorted(by_chunk[ch])
        rng.shuffle(pool)
        picked = sorted(pool[:a.per_chunk])
        per[ch] = picked
        clips += picked

    cams = CAM4 if a.cameras == 4 else CAM7
    est = len(chosen) * (1.35 * len(cams) + 0.08)
    man = {"chunks": chosen, "clips_by_chunk": per, "clips": sorted(clips),
           "cameras": cams, "estimated_download_gb": round(est, 1)}
    json.dump(man, open(a.out, "w"), indent=1)
    print(f"[plan] {len(clips)} clips across {len(chosen)} chunks "
          f"(of {len(dense)} eligible) -> {a.out}")
    print(f"[plan] chunks: {', '.join(chosen)}")
    print(f"[plan] estimated download ~{est:.0f} GB ({len(cams)} cameras + labels)")
    print(f"[plan] NOTE chunk-clustered, not a random cohort sample — see docstring")


if __name__ == "__main__":
    main()
