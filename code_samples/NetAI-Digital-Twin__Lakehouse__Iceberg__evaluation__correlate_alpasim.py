"""Correlate per-clip open-loop MF-PDMS against per-clip closed-loop AlpaSim.

MF-PDMS is cheap and dataset-agnostic; AlpaSim is expensive and tied to the
NVIDIA PhysicalAI NuRec artifacts. The question this answers is how much the
open-loop metric tells us about closed-loop outcomes on the *same clips with
the same policy* -- i.e. whether MF-PDMS is usable as a proxy, and where it
disagrees.

Usage:
    python correlate_alpasim.py --mfpdms .results_nurec_<policy>.parquet \
        --alpasim /tmp/alpasim_per_clip.parquet
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

# Closed-loop metrics worth correlating. `higher_is_better` only sets the sign
# we expect; it does not change the computation.
ALPASIM_METRICS = {
    "progress": True,
    "progress_rel": True,
    "dist_traveled_m": True,
    "duration_frac_20s": True,
    "dist_to_gt_trajectory": False,
    "offroad": False,
    "collision_any": False,
    "offroad_or_collision_at_fault": False,
}


def bootstrap_ci(x, y, fn, n=10000, seed=0):
    """Percentile CI for a correlation statistic; wide is expected at small n."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    stats_ = np.array([fn(x[i], y[i]) for i in idx])
    stats_ = stats_[np.isfinite(stats_)]
    if stats_.size == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(stats_, [2.5, 97.5]))


def _spearman(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return stats.spearmanr(a, b).statistic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mfpdms", type=pathlib.Path, required=True)
    ap.add_argument("--alpasim", type=pathlib.Path, required=True)
    ap.add_argument("--open-loop-metric", default="mf_pdms")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    ol = pd.read_parquet(args.mfpdms)
    cl = pd.read_parquet(args.alpasim)
    joined = ol.merge(cl, on="clip_id", how="inner", suffixes=("_ol", "_cl"))

    policy = ol["policy"].iloc[0] if "policy" in ol else "?"
    print(f"policy            {policy}")
    print(f"open-loop clips   {len(ol)}")
    print(f"closed-loop clips {len(cl)}")
    print(f"overlap           {len(joined)}")
    if len(joined) < 3:
        raise SystemExit("\nToo few overlapping clips to correlate.")

    x = joined[args.open_loop_metric].to_numpy(dtype=float)
    print(
        f"\n{args.open_loop_metric} over overlap: "
        f"mean={x.mean():.3f} sd={x.std(ddof=1):.3f} "
        f"min={x.min():.3f} max={x.max():.3f}"
    )

    rows = []
    for metric, higher_better in ALPASIM_METRICS.items():
        if metric not in joined:
            continue
        y = joined[metric].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        if len(xs) < 3 or np.std(ys) == 0:
            rows.append(
                {
                    "alpasim_metric": metric,
                    "n": len(xs),
                    "spearman": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "pearson": np.nan,
                    "p": np.nan,
                    "note": "constant" if len(xs) >= 3 else "too few finite",
                }
            )
            continue
        rho = stats.spearmanr(xs, ys)
        lo, hi = bootstrap_ci(xs, ys, _spearman)
        rows.append(
            {
                "alpasim_metric": metric,
                "n": len(xs),
                "spearman": rho.statistic,
                "ci_lo": lo,
                "ci_hi": hi,
                "pearson": stats.pearsonr(xs, ys).statistic,
                "p": rho.pvalue,
                "note": "expect +" if higher_better else "expect -",
            }
        )

    res = pd.DataFrame(rows)
    print("\n--- per-clip correlation: open-loop vs closed-loop ---")
    print(res.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    sig = res.dropna(subset=["p"]).query("p < 0.05")
    print(
        f"\n{len(sig)} of {res['p'].notna().sum()} correlations significant at p<0.05"
        + (f": {', '.join(sig.alpasim_metric)}" if len(sig) else "")
    )

    if args.out:
        res.to_parquet(args.out)
        joined.to_parquet(str(args.out).replace(".parquet", ".joined.parquet"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
