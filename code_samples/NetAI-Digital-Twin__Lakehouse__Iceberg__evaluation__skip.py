#!/usr/bin/env python3
"""skip.py — rollout triage: which curated clips earn a closed-loop rollout.

Open-loop MF-PDMS carries little per-clip information about closed-loop outcome
(ALPASIM.md, n=40), so "run closed-loop on whatever open-loop flags" is not a
policy. This tool builds and evaluates a *screen* — a per-clip predictor of
closed-loop failure fitted on cheap features: the curation axes (conflict,
behavioral, camera) and the open-loop reference ladder — and turns it into a
budget dial: run the top fraction of clips by predicted failure, and report what
recall of the real failures that fraction buys.

Three subcommands, all offline and CPU-only:

  features   join cheap per-clip features to closed-loop outcomes -> one parquet
  fit        leave-one-out screen; recall-vs-budget curve against the baselines
             (random, open-loop MF-PDMS ascending, single curation axes)
  select     rank every featured clip, labelled or not, and emit the rollout
             list for a budget — the input to alpasim/run_scene.sh or run_eval.py

Nothing here imports the evaluator (harness.py) or AlpaSim; it consumes their outputs:
  curation axes   <axes-root>/.conflict, .behavioral, .camera_perception
                  (planning/*_runner.py with NFS_ROOT=<axes-root>)
  open-loop       .results_<tag>_<policy>.parquet from run_eval.py
  closed-loop     per-clip parquet from alpasim/per_clip.py -o

Column prefixes in the feature table: ax_ (curation), ol_<policy>_ (open-loop),
cl_<name>_ (closed-loop outcome). Needs numpy/pandas/scikit-learn (.skip_venv).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AXES = os.path.join(HERE, ".av_slice_nurec")
PROD_CONFLICT = "/mnt/netai-e2e/nvidia-physicalai-av-subset/.conflict/conflict_shard_00_of_01.parquet"
OL_COLS = ["nc", "ttc", "ep", "hc", "ec", "mf_pdms", "n_decisions"]
CL_COLS = ["progress", "progress_rel", "dist_traveled_m", "dist_to_gt_trajectory",
           "dist_to_gt_location", "offroad", "collision_any", "collision_at_fault",
           "collision_rear", "offroad_or_collision", "offroad_or_collision_at_fault",
           "duration_frac_20s", "min_distance_to_obstacle_m"]


def _read(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


# ----------------------------------------------------------------------------- features
def build_features(axes_root: str, open_loop: list[str], closed: list[tuple[str, str]],
                   prod_conflict: str | None) -> pd.DataFrame:
    frames = []

    p = f"{axes_root}/.conflict/conflict_shard_00_of_01.parquet"
    if os.path.exists(p):
        c = _read(p)[["clip_id", "conflict_load", "conflict_score"]]
        if prod_conflict and os.path.exists(prod_conflict):
            # rank the slice's raw load against the production population, so the
            # score is on the same scale Gold thresholds use rather than slice-internal
            prod = np.sort(_read(prod_conflict)["conflict_load"].to_numpy())
            c["conflict_prod_rank"] = np.searchsorted(prod, c["conflict_load"].to_numpy(), side="right") / len(prod)
        frames.append(c.add_prefix("ax_").rename(columns={"ax_clip_id": "clip_id"}))

    p = f"{axes_root}/.behavioral/behavioral_shard_00_of_01.parquet"
    if os.path.exists(p):
        b = _read(p).drop(columns=["scored_at", "active_axes"], errors="ignore")
        # conflict is the same axis as .conflict's raw load; keep the extra ones only
        b = b.drop(columns=["conflict", "conflict_rank"], errors="ignore")
        frames.append(b.add_prefix("ax_").rename(columns={"ax_clip_id": "clip_id"}))

    p = f"{axes_root}/.camera_perception/camera_perception.parquet"
    if os.path.exists(p):
        frames.append(_read(p).add_prefix("ax_").rename(columns={"ax_clip_id": "clip_id"}))
    p = f"{axes_root}/.camera_perception/camera_gated.parquet"
    if os.path.exists(p):
        g = _read(p)[["clip_id", "low_conf"]].rename(columns={"low_conf": "ax_cam_gated_low_conf"})
        frames.append(g)

    have_tracks = False
    for f in open_loop:
        t = _read(f)
        pol = t["policy"].iloc[0] if "policy" in t else os.path.basename(f).split("_")[-1].split(".")[0]
        keep = ["clip_id"] + [c for c in OL_COLS if c in t.columns]
        o = t[keep].add_prefix(f"ol_{pol}_").rename(columns={f"ol_{pol}_clip_id": "clip_id"})
        if "n_tracks" in t.columns and not have_tracks:   # scene property, same in every file
            o["ol_n_tracks"] = t["n_tracks"].to_numpy(); have_tracks = True
        frames.append(o)

    for name, f in closed:
        t = _read(f)
        keep = ["clip_id"] + [c for c in CL_COLS if c in t.columns]
        cl = t[keep].add_prefix(f"cl_{name}_").rename(columns={f"cl_{name}_clip_id": "clip_id"})
        cl[f"has_cl_{name}"] = True
        frames.append(cl)

    if not frames:
        sys.exit("no inputs found")
    df = frames[0]
    for fr in frames[1:]:
        df = df.merge(fr, on="clip_id", how="outer")
    for col in [c for c in df.columns if c.startswith("has_cl_")]:
        df[col] = df[col].fillna(False).astype(bool)
    return df


# ----------------------------------------------------------------------------- fit
def _feature_cols(df: pd.DataFrame, feature_set: str, policy: str) -> list[str]:
    ax = [c for c in df.columns if c.startswith("ax_") and c != "ax_conflict_score"]
    ol = [c for c in df.columns if c.startswith("ol_")]
    if feature_set == "axes":
        return ax
    if feature_set == "openloop":
        return ol
    if feature_set == "openloop-self":       # only the policy's own open-loop score
        return [c for c in ol if c.startswith(f"ol_{policy}_")]
    if feature_set == "all":
        return ax + ol
    return [c.strip() for c in feature_set.split(",")]


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def _impute(Xtr: np.ndarray, Xte: np.ndarray):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


def loo_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predicted failure probability for every labelled clip."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if y[tr].min() == y[tr].max():
            p[i] = y[tr].mean(); continue
        Xtr, Xte = _impute(X[tr], X[i:i + 1])
        m = _model().fit(Xtr, y[tr])
        p[i] = m.predict_proba(Xte)[0, 1]
    return p


def recall_at_budget(score: np.ndarray, y: np.ndarray, budgets: list[float]) -> list[float]:
    """Recall of positives when the top `budget` fraction of clips (by score) is run."""
    order = np.argsort(-score, kind="stable")
    npos = y.sum()
    out = []
    for b in budgets:
        k = max(1, int(round(b * len(y))))
        out.append(float(y[order[:k]].sum() / npos) if npos else float("nan"))
    return out


def auc_ci(score: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0):
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return float("nan"), (float("nan"), float("nan"))
    a = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        bs.append(roc_auc_score(y[idx], score[idx]))
    return a, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fit(df: pd.DataFrame, policy: str, target: str, feature_set: str, budgets: list[float],
        out_prefix: str) -> dict:
    tcol = f"cl_{policy}_{target}"
    lab = df[df[f"has_cl_{policy}"] & df[tcol].notna()].copy()
    y = (lab[tcol].to_numpy() > 0).astype(int)
    cols = _feature_cols(lab, feature_set, policy)
    X = lab[cols].to_numpy(dtype=float)
    print(f"labelled clips: {len(lab)}  positives ({target}): {y.sum()}  features[{feature_set}]: {len(cols)}")

    rankings = {"screen (LOO logistic)": loo_scores(X, y)}
    if f"ol_{policy}_mf_pdms" in lab:
        rankings["open-loop MF-PDMS, worst first"] = -lab[f"ol_{policy}_mf_pdms"].to_numpy(dtype=float)
    for c, label in [("ax_conflict_load", "conflict load, highest first"),
                     ("ax_behavioral_score", "behavioral score, highest first"),
                     ("ax_cam_gated_low_conf", "camera gated low-conf, highest first"),
                     ("ol_constant_velocity_mf_pdms", "constant-velocity open-loop, worst first")]:
        if c in lab:
            v = lab[c].to_numpy(dtype=float)
            rankings[label] = -v if c.endswith("mf_pdms") else v
    for k in rankings:
        rankings[k] = np.where(np.isnan(rankings[k]), np.nanmin(rankings[k]), rankings[k])

    rows = []
    hdr = f"{'ranking':44s} {'AUC':>5s} {'95% CI':>14s} " + " ".join(f"@{b:.0%}".rjust(6) for b in budgets)
    print(hdr)
    print("random (expected)".ljust(44), " 0.500", "              ", " ".join(f"{b:6.2f}" for b in budgets))
    for name, s in rankings.items():
        a, (lo, hi) = auc_ci(s, y)
        rec = recall_at_budget(s, y, budgets)
        print(f"{name:44s} {a:5.3f} [{lo:5.3f},{hi:5.3f}] " + " ".join(f"{r:6.2f}" for r in rec))
        rows.append({"ranking": name, "auc": a, "auc_lo": lo, "auc_hi": hi,
                     **{f"recall@{b}": r for b, r in zip(budgets, rec)}})
    curve = pd.DataFrame(rows)
    curve.to_csv(f"{out_prefix}_curve.csv", index=False)

    # full refit for `select`, coefficients reported for the record
    Xf, _ = _impute(X, X)
    m = _model().fit(Xf, y)
    coef = dict(zip(cols, m[-1].coef_[0].round(3).tolist()))
    summary = {"policy": policy, "target": target, "feature_set": feature_set, "n": int(len(lab)),
               "positives": int(y.sum()), "features": cols, "coef_std": coef,
               "budgets": budgets, "rankings": rows}
    json.dump(summary, open(f"{out_prefix}_fit.json", "w"), indent=1)
    print(f"wrote {out_prefix}_curve.csv, {out_prefix}_fit.json")
    return summary


# ----------------------------------------------------------------------------- select
def select(df: pd.DataFrame, policy: str, target: str, feature_set: str, budget: float,
           exclude_labelled: bool, out: str, scene_ids: bool):
    tcol = f"cl_{policy}_{target}"
    lab = df[df[f"has_cl_{policy}"] & df[tcol].notna()]
    cols = _feature_cols(df, feature_set, policy)
    y = (lab[tcol].to_numpy() > 0).astype(int)
    Xtr, Xall = _impute(lab[cols].to_numpy(dtype=float), df[cols].to_numpy(dtype=float))
    m = _model().fit(Xtr, y)
    df = df.assign(p_fail=m.predict_proba(Xall)[:, 1]).sort_values("p_fail", ascending=False)
    pool = df[~df[f"has_cl_{policy}"]] if exclude_labelled else df
    k = int(round(budget * len(pool))) if budget <= 1 else int(budget)
    pick = pool.head(max(1, k))
    ids = [f"clipgt-{c}" if scene_ids else c for c in pick["clip_id"]]
    with open(out, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"budget {budget}: {len(pick)} of {len(pool)} clips -> {out}  "
          f"(p_fail {pick['p_fail'].min():.2f}..{pick['p_fail'].max():.2f}; fitted on {len(lab)} labelled)")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features")
    f.add_argument("--axes-root", default=DEFAULT_AXES)
    f.add_argument("--open-loop", default=os.path.join(HERE, ".results_nurec_*.parquet"),
                   help="glob of run_eval.py result parquets")
    f.add_argument("--closed-loop", nargs=2, action="append", metavar=("NAME", "PARQUET"), default=[],
                   help="per-clip closed-loop table from alpasim/per_clip.py -o; repeatable")
    f.add_argument("--prod-conflict", default=PROD_CONFLICT)
    f.add_argument("--out", default=os.path.join(HERE, ".skip_features.parquet"))

    for name in ("fit", "select"):
        s = sub.add_parser(name)
        s.add_argument("--features", default=os.path.join(HERE, ".skip_features.parquet"))
        s.add_argument("--policy", default="vavam")
        s.add_argument("--target", default="offroad_or_collision_at_fault")
        s.add_argument("--feature-set", default="all",
                       help="axes | openloop | openloop-self | all | comma-separated columns")
        if name == "fit":
            s.add_argument("--budgets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.8")
            s.add_argument("--out-prefix", default=os.path.join(HERE, ".skip"))
        else:
            s.add_argument("--budget", type=float, required=True, help="fraction (<=1) or count")
            s.add_argument("--exclude-labelled", action="store_true",
                           help="only clips without a closed-loop result (the next batch to run)")
            s.add_argument("--scene-ids", action="store_true", help="write clipgt-<id> for run_scene.sh")
            s.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "features":
        df = build_features(a.axes_root, sorted(glob.glob(a.open_loop)), a.closed_loop, a.prod_conflict)
        df.to_parquet(a.out, index=False)
        has = {c[7:]: int(df[c].sum()) for c in df.columns if c.startswith("has_cl_")}
        print(f"wrote {a.out}: {len(df)} clips, {df.shape[1]} columns; closed-loop labelled: {has}")
    elif a.cmd == "fit":
        fit(pd.read_parquet(a.features), a.policy, a.target, a.feature_set,
            [float(b) for b in a.budgets.split(",")], a.out_prefix)
    else:
        select(pd.read_parquet(a.features), a.policy, a.target, a.feature_set, a.budget,
               a.exclude_labelled, a.out, a.scene_ids)


if __name__ == "__main__":
    main()
