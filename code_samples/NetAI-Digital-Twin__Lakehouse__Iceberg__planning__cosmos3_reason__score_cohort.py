#!/usr/bin/env python3
"""Score a random cohort sample with the cosmos3 axis — production-shaped, no controls.

The 452-clip gate set is label-enriched (45% OOD vs a ~10% cohort base), so a top-N
selection experiment run on it would not resemble production selection. This draws a
RANDOM sample of the cohort and scores it, which is also exactly what deploying the
axis would look like.

Writes .cohort_scores.json (clip_id -> reasoned logit-EV difficulty).
Usage: ./c3_venv/bin/python score_cohort.py [N]
"""
import json
import os
import random
import sys
import time

import cosmos3_scorer as cs
import analyze_axis as A

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(_HERE, ".cohort_sample.json")
OUT = os.path.join(_HERE, ".cohort_scores.json")
SEED = 7


def build_sample(n):
    """Clips that have every axis the comparison needs, plus a decodable mp4."""
    if os.path.exists(SAMPLE):
        return json.load(open(SAMPLE))
    conflict = A.load_axis(".conflict", "conflict_score")
    camera = A.load_axis(".camera_perception", "low_conf")
    dark = A.load_darkness()
    idx = cs.clip_index()
    pool = sorted(set(conflict) & set(camera) & set(dark) & set(idx))
    rng = random.Random(SEED)
    rng.shuffle(pool)
    sample = pool[:n]
    json.dump(sample, open(SAMPLE, "w"))
    print(f"[cohort] sampled {len(sample)} of {len(pool)} eligible clips (seed {SEED})")
    return sample


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    sample = build_sample(n)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [c for c in sample if c not in done]
    print(f"[cohort] {len(done)} already scored, {len(todo)} to go", flush=True)
    if not todo:
        return
    model, proc = cs.load_model()
    t0 = time.time()
    for k, cid in enumerate(todo):
        try:
            r = cs.reasoned_score(model, proc, cid)
            done[cid] = round(r["score"], 6)
        except Exception as e:
            print(f"  [WARN] {cid[:8]}: {str(e)[:70]}", flush=True)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[cohort] {k+1}/{len(todo)} ({el/(k+1):.2f}s/clip, "
                  f"eta {(len(todo)-k-1)*el/(k+1)/60:.0f} min)", flush=True)
            json.dump(done, open(OUT, "w"))
    json.dump(done, open(OUT, "w"))
    print(f"[cohort] wrote {OUT} ({len(done)} clips)")


if __name__ == "__main__":
    main()
