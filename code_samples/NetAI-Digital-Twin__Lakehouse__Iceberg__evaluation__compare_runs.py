#!/usr/bin/env python3
"""Paired comparison of two policy runs over the SAME clips.

Unpaired means differ by slice as much as by model — on this data the slice effect
was ~7x the model effect — so any claim that model A beats model B must come from a
paired test on identical clips, with a CI.

    python compare_runs.py a.parquet b.parquet
"""
import random
import statistics as st
import sys

import pyarrow.parquet as pq

N_BOOT = 10000
KEYS = ("mf_pdms", "nc", "ttc", "ep", "hc", "ec")


def load(p):
    d = pq.read_table(p).to_pylist()
    return {r["clip_id"]: r for r in d}, (d[0]["policy"] if d else p)


def boot_ci(diffs, n=N_BOOT, seed=0):
    rng = random.Random(seed)
    m = sorted(st.mean([rng.choice(diffs) for _ in diffs]) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    (A, na), (B, nb) = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(A) & set(B))
    print(f"===== PAIRED: {na} vs {nb}  (n={len(common)} shared clips) =====")
    for k in KEYS:
        da = [A[c][k] for c in common]
        db = [B[c][k] for c in common]
        diffs = [x - y for x, y in zip(da, db)]
        lo, hi = boot_ci(diffs)
        verdict = ("A better" if lo > 0 else "B better" if hi < 0 else "no difference")
        print(f"  {k:8s} {na[:14]:>14s}={st.mean(da):.3f}  {nb[:14]:>14s}={st.mean(db):.3f}  "
              f"diff={st.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    wins = sum(1 for c in common if A[c]["mf_pdms"] > B[c]["mf_pdms"])
    ties = sum(1 for c in common if A[c]["mf_pdms"] == B[c]["mf_pdms"])
    print(f"\n  per-clip mf_pdms: {na} wins {wins}, {nb} wins {len(common)-wins-ties}, ties {ties}")


if __name__ == "__main__":
    main()
