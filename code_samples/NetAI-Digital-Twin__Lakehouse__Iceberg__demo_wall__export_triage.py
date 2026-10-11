#!/usr/bin/env python3
"""export_triage.py — per-clip rollout-triage scores for the wall's Triage chapter.

Reproduces the screen reported in evaluation/SKIP.md (VaVAM, target
offroad_or_collision_at_fault, open-loop ladder features, leave-one-out logistic
regression) with skip.py's own functions, so the wall shows the same ranking the
doc reports, and writes one row per labelled clip to user_data/wall_triage.json.
Refuses to write if recall at a 50 % budget no longer matches SKIP.md.

Needs pandas + scikit-learn, i.e. the evaluation venv:
    evaluation/.skip_venv/bin/python demo_wall/export_triage.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Lakehouse/Iceberg
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import skip  # noqa: E402

POLICY, TARGET, FEATURE_SET = "vavam", "offroad_or_collision_at_fault", "openloop"
EXPECTED_RECALL_AT_HALF = 27 / 41      # SKIP.md: 0.66 at a 50 % budget
CLOSED_LOOP = os.path.join(ROOT, "evaluation", ".cl_vavam_perclip.parquet")


def main():
    df = pd.read_parquet(os.path.join(ROOT, "evaluation", ".skip_features.parquet"))
    tcol = f"cl_{POLICY}_{TARGET}"
    lab = df[df[f"has_cl_{POLICY}"] & df[tcol].notna()].copy()
    y = (lab[tcol].to_numpy() > 0).astype(int)
    X = lab[skip._feature_cols(lab, FEATURE_SET, POLICY)].to_numpy(dtype=float)
    p = skip.loo_scores(X, y)

    rec = skip.recall_at_budget(p, y, [0.2, 0.5, 0.8])
    if abs(rec[1] - EXPECTED_RECALL_AT_HALF) > 1e-9:
        sys.exit(f"recall@0.5 = {rec[1]:.4f}, SKIP.md says {EXPECTED_RECALL_AT_HALF:.4f}; not exporting")

    # outcome class per clip from the closed-loop rows (collision beats off-road)
    cl = pd.read_parquet(CLOSED_LOOP).set_index("clip_id")
    outcome = {}
    for cid, r in cl.iterrows():
        outcome[cid] = ("collision" if r["collision_any"] > 0 else
                        "offroad" if r["offroad"] > 0 else "clean")

    order = np.argsort(-p, kind="stable")          # the screen's rollout order
    clips = []
    for rank, i in enumerate(order):
        cid = lab["clip_id"].iloc[i]
        clips.append({"rank": rank, "clip": cid[:8], "p_fail": round(float(p[i]), 4),
                      "at_fault": int(y[i]), "outcome": outcome.get(cid, "unknown")})
    out = {"policy": POLICY, "target": TARGET, "feature_set": FEATURE_SET,
           "n": int(len(y)), "positives": int(y.sum()),
           "recall": {"0.2": rec[0], "0.5": rec[1], "0.8": rec[2]}, "clips": clips}
    dst = os.path.join(ROOT, "user_data", "wall_triage.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[triage] n={out['n']} positives={out['positives']} recall@.2/.5/.8="
          f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} -> {dst}")


if __name__ == "__main__":
    main()
