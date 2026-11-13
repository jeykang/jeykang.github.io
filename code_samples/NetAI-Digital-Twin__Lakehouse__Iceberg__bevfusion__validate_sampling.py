"""Sampling-adequacy calibration for the BEVFusion difficulty scorer.

Question: is N=10 frames per clip sufficient to determine relative
difficulty for Gold-subset selection, or do we need more?

Method: take a 200-clip random subset of Silver, run runner.py at
N ∈ {10, 20, 40} frames/clip, compute:
  (a) Spearman rank correlation between sample sizes
  (b) Jaccard overlap of top-10% sets between sample sizes

Pass criteria for committing to N=10 on the full 33,719-clip Gold run:
  Spearman ρ(10, 40) ≥ 0.85   AND   Jaccard(top10%@10, top10%@40) ≥ 0.80

If either threshold fails, climb to N=20 and re-test against N=40.

This script READS three parquets produced by runner.py invocations at
the three N values, computes the metrics, and prints a verdict.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_scores(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    return df.set_index("clip_id")["perception_score"]


def jaccard_topk(s1: pd.Series, s2: pd.Series, top_pct: float) -> float:
    common = s1.index.intersection(s2.index)
    n_top = max(1, int(len(common) * top_pct / 100))
    a = set(s1.loc[common].nlargest(n_top).index)
    b = set(s2.loc[common].nlargest(n_top).index)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-10", required=True)
    ap.add_argument("--scores-20", required=True)
    ap.add_argument("--scores-40", required=True)
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--rho-thresh", type=float, default=0.85)
    ap.add_argument("--jaccard-thresh", type=float, default=0.80)
    args = ap.parse_args()

    s10 = load_scores(args.scores_10)
    s20 = load_scores(args.scores_20)
    s40 = load_scores(args.scores_40)
    common = s10.index.intersection(s20.index).intersection(s40.index)
    print(f"Validation set: {len(common)} clips scored at all three N values\n")

    rows = []
    for (a_name, a), (b_name, b) in [
        (("N=10", s10), ("N=40", s40)),
        (("N=20", s20), ("N=40", s40)),
        (("N=10", s10), ("N=20", s20)),
    ]:
        rho, _ = spearmanr(a.loc[common], b.loc[common])
        jac = jaccard_topk(a.loc[common], b.loc[common], args.top_pct)
        rows.append({"a": a_name, "b": b_name, "spearman_rho": rho,
                     f"jaccard_top{args.top_pct:.0f}": jac})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    rho_10_40, _ = spearmanr(s10.loc[common], s40.loc[common])
    jac_10_40 = jaccard_topk(s10.loc[common], s40.loc[common], args.top_pct)
    pass_rho = rho_10_40 >= args.rho_thresh
    pass_jac = jac_10_40 >= args.jaccard_thresh
    if pass_rho and pass_jac:
        print(f"VERDICT: N=10 sufficient (ρ={rho_10_40:.3f} ≥ {args.rho_thresh}, "
              f"Jaccard={jac_10_40:.3f} ≥ {args.jaccard_thresh})")
        return 0
    print(f"VERDICT: N=10 insufficient (ρ={rho_10_40:.3f}, Jaccard={jac_10_40:.3f}). "
          f"Re-test with N=20.")
    rho_20_40, _ = spearmanr(s20.loc[common], s40.loc[common])
    jac_20_40 = jaccard_topk(s20.loc[common], s40.loc[common], args.top_pct)
    if rho_20_40 >= args.rho_thresh and jac_20_40 >= args.jaccard_thresh:
        print(f"  N=20 OK (ρ={rho_20_40:.3f}, Jaccard={jac_20_40:.3f}) — use 20.")
    else:
        print(f"  N=20 also insufficient (ρ={rho_20_40:.3f}, Jaccard={jac_20_40:.3f}). "
              f"Use N=40 for the full run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
