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
  predict    freeze the screen: fit on every labelled clip and record p_fail for
             every clip (plus the feature file's hash) BEFORE the next batch runs,
             so that batch is a prospective test rather than another LOO estimate
  prospective  score a frozen ranking against the batch's closed-loop outcomes: AUC and
             recall at each budget, against random and MF-PDMS, with label noise
             measured from repeated rollouts (n_rollouts > 1)

Nothing here imports the evaluator (harness.py) or AlpaSim; it consumes their outputs:
  curation axes   <axes-root>/.conflict, .behavioral, .camera_perception
                  (planning/*_runner.py with NFS_ROOT=<axes-root>)
  open-loop       .results_<tag>_<policy>.parquet from run_eval.py
  closed-loop     per-clip parquet from alpasim/per_clip.py -o

Column prefixes in the feature table: ax_ (curation), ol_<policy>_ (open-loop, the per-clip
mean over decision points), od_<policy>_<metric>_{min,first} (open-loop per decision point,
from run_eval.py --per-decision: the worst decision and the earliest one), cl_<name>_
(closed-loop outcome). Needs numpy/pandas/scikit-learn (.skip_venv).
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
                   prod_conflict: str | None, per_decision: list[str] | None = None) -> pd.DataFrame:
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

    for f in per_decision or []:
        t = _read(f)
        pol = t["policy"].iloc[0]
        t = t.sort_values(["clip_id", "t0_us"])
        keys = [c for c in OL_COLS if c in t.columns and c != "n_decisions"]
        g = t.groupby("clip_id")[keys]
        agg = pd.concat([g.min().add_suffix("_min"), g.first().add_suffix("_first")], axis=1)
        frames.append(agg.add_prefix(f"od_{pol}_").reset_index())

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
    od = [c for c in df.columns if c.startswith("od_")]
    if feature_set == "axes":
        return ax
    if feature_set == "openloop":
        return ol
    if feature_set == "decisions":
        return od
    if feature_set == "openloop+decisions":
        return ol + od
    if feature_set in ("openloop+min", "openloop+first"):
        return ol + [c for c in od if c.endswith("_" + feature_set.split("+")[1])]
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



# ----------------------------------------------------------------------------- predict (freeze)
def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(df: pd.DataFrame, policy: str, target: str, feature_set: str, out: str,
            features_path: str, budget: float = 0.5) -> pd.DataFrame:
    """Fit on every labelled clip; record p_fail for every clip before new rollouts exist.

    Writes <out> (one row per clip: labelled, complete_features, p_fail, rank among the
    unlabelled, selected at `budget`) and <out minus .parquet>.json (fit metadata and the
    sha256 of the feature file). Same model and imputation as `select`, so the clips it
    marks as selected are the ones `select --exclude-labelled --budget <budget>` writes.
    """
    import datetime
    import sklearn
    tcol = f"cl_{policy}_{target}"
    labelled = (df[f"has_cl_{policy}"] & df[tcol].notna()).to_numpy()
    lab = df[labelled]
    cols = _feature_cols(df, feature_set, policy)
    y = (lab[tcol].to_numpy() > 0).astype(int)
    Xtr, Xall = _impute(lab[cols].to_numpy(dtype=float), df[cols].to_numpy(dtype=float))
    m = _model().fit(Xtr, y)
    rec = pd.DataFrame({"clip_id": df["clip_id"].to_numpy(), "labelled": labelled,
                        "complete_features": df[cols].notna().all(axis=1).to_numpy(),
                        "p_fail": m.predict_proba(Xall)[:, 1]})
    pool = rec[~rec["labelled"]].sort_values("p_fail", ascending=False, kind="stable")
    k = int(round(budget * len(pool))) if budget <= 1 else int(budget)
    rec["rank_in_pool"] = pd.Series(np.arange(1, len(pool) + 1), index=pool.index)
    rec["selected"] = rec["rank_in_pool"].le(max(1, k)).fillna(False).astype(bool)
    rec.to_parquet(out, index=False)
    meta = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "purpose": "screen frozen before the unlabelled clips are rolled out (prospective test)",
            "features_file": os.path.abspath(features_path), "features_sha256": _sha256(features_path),
            "policy": policy, "target": target, "feature_set": feature_set, "features": cols,
            "n_labelled": int(labelled.sum()), "positives": int(y.sum()),
            "n_unlabelled": int(len(pool)), "unlabelled_complete_features": int(rec.loc[~rec["labelled"], "complete_features"].sum()),
            "budget": budget, "n_selected": int(rec["selected"].sum()),
            "model": "StandardScaler + LogisticRegression(C=0.5, balanced), median imputation from labelled clips",
            "sklearn": sklearn.__version__,
            "coef_std": dict(zip(cols, m[-1].coef_[0].round(4).tolist()))}
    meta_path = (out[:-8] if out.endswith(".parquet") else out) + ".json"
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"froze {len(rec)} clips ({meta['n_labelled']} labelled, {meta['n_unlabelled']} unlabelled, "
          f"{meta['n_selected']} selected at budget {budget}) -> {out}, {meta_path}")
    return rec



# ----------------------------------------------------------------------------- prospective
def prospective(frozen_path: str, closed: list[str], features_path: str, policy: str, target: str,
                budgets: list[float], out_prefix: str) -> dict:
    """Score a frozen ranking (predict) on rollouts that ran after it was frozen.

    Labels per clip, from the batch's per-(clip, rollout) rows (alpasim/per_clip.py):
      single    one rollout per clip, the lowest rollout_id — a single draw, like batches 1-3
      majority  more than half of the clip's rollouts fail
    With >1 rollout per clip, label noise is reported two ways: the share of clips whose
    rollouts disagree, and a ceiling — the expected AUC of a screen that knew each clip's true
    failure propensity q, scored against single-rollout labels y ~ Bernoulli(q). q is modelled
    as Beta(a, b) fitted to the rollout counts by the method of moments (beta-binomial, the
    intra-clip correlation rho = 1 / (a + b + 1)); the expectation is simulated. (Ranking each
    rollout by the failure rate of the clip's other rollouts is biased low with 2-3 rollouts.)
    """
    from sklearn.metrics import roc_auc_score
    fr = pd.read_parquet(frozen_path)
    meta_path = (frozen_path[:-8] if frozen_path.endswith(".parquet") else frozen_path) + ".json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    cl = pd.concat([_read(c) for c in closed], ignore_index=True)
    if target not in cl:
        sys.exit(f"{target} not in the closed-loop tables")
    cl = cl[cl[target].notna()].copy()
    cl["fail"] = (cl[target] > 0).astype(int)
    cl = cl.sort_values(["clip_id", "rollout_id"])
    per = cl.groupby("clip_id").agg(n_rollouts=("fail", "size"), rate=("fail", "mean"),
                                    single=("fail", "first")).reset_index()
    per["majority"] = (per["rate"] > 0.5).astype(int)
    df = fr[~fr["labelled"]].merge(per, on="clip_id", how="inner")
    feats = pd.read_parquet(features_path)[["clip_id", f"ol_{policy}_mf_pdms"]] \
        if features_path and os.path.exists(features_path) else None
    if feats is not None:
        df = df.merge(feats, on="clip_id", how="left")
    print(f"frozen {meta.get('frozen_utc', '?')} on n={meta.get('n_labelled', '?')}; "
          f"batch: {len(per)} clips with outcomes, {len(df)} of them in the frozen pool "
          f"({int(df['complete_features'].sum())} with complete features); rollouts per clip: "
          f"{per['n_rollouts'].min()}-{per['n_rollouts'].max()}")

    out = {"frozen": meta, "target": target, "n_clips": int(len(df)), "budgets": budgets, "results": []}
    hdr = f"{'subset':10s} {'label':9s} {'ranking':28s} {'n':>3s} {'pos':>3s} {'AUC':>5s} {'95% CI':>14s} " + \
          " ".join(f"@{b:.0%}".rjust(6) for b in budgets)
    print(hdr)
    for subset, sub in (("all", df), ("complete", df[df["complete_features"]])):
        for label in ("single", "majority"):
            y = sub[label].to_numpy()
            ranks = {"screen (frozen)": sub["p_fail"].to_numpy()}
            if f"ol_{policy}_mf_pdms" in sub and sub[f"ol_{policy}_mf_pdms"].notna().any():
                v = -sub[f"ol_{policy}_mf_pdms"].to_numpy(dtype=float)
                ranks["open-loop MF-PDMS, worst first"] = np.where(np.isnan(v), np.nanmin(v), v)
            for name, sc in ranks.items():
                a, (lo, hi) = auc_ci(sc, y)
                rec = recall_at_budget(sc, y, budgets)
                print(f"{subset:10s} {label:9s} {name:28s} {len(y):3d} {int(y.sum()):3d} {a:5.3f} [{lo:5.3f},{hi:5.3f}] "
                      + " ".join(f"{r:6.2f}" for r in rec))
                out["results"].append({"subset": subset, "label": label, "ranking": name, "n": int(len(y)),
                                       "positives": int(y.sum()), "auc": a, "auc_lo": lo, "auc_hi": hi,
                                       **{f"recall@{b}": r for b, r in zip(budgets, rec)}})
    # the dial as frozen: recall of the clips it selected
    for label in ("single", "majority"):
        y, sel = df[label].to_numpy(), df["selected"].to_numpy()
        out[f"dial_recall_{label}"] = float(y[sel].sum() / max(1, y.sum()))
        print(f"dial as frozen ({int(sel.sum())} of {len(df)} selected): recall {out[f'dial_recall_{label}']:.2f} ({label} labels)")
    # label noise
    multi = per[per["n_rollouts"] > 1]
    if len(multi):
        g = cl[cl["clip_id"].isin(multi["clip_id"])].groupby("clip_id")["fail"]
        out["share_clips_disagreeing"] = float((g.nunique() > 1).mean())
        rng = np.random.default_rng(0)

        def ceiling(rates: np.ndarray, n: float, sims: int) -> tuple[float, float]:
            mu, var = float(rates.mean()), float(rates.var(ddof=1))
            rho = (n * var / max(mu * (1 - mu), 1e-9) - 1) / max(n - 1, 1e-9)
            rho = float(np.clip(rho, 1e-3, 0.999))
            ab = 1 / rho - 1
            q = rng.beta(max(mu * ab, 1e-3), max((1 - mu) * ab, 1e-3), size=(sims, len(df)))
            y = rng.random(q.shape) < q
            r = q.argsort(axis=1).argsort(axis=1) + 1                     # ranks (ties: measure zero)
            npos = y.sum(axis=1); ok = (npos > 0) & (npos < y.shape[1])
            auc = ((r * y).sum(axis=1) - npos * (npos + 1) / 2) / np.maximum(npos * (y.shape[1] - npos), 1)
            return rho, float(auc[ok].mean())

        n = float(multi["n_rollouts"].mean()); rates = multi["rate"].to_numpy()
        rho, ceil = ceiling(rates, n, 4000)
        boot = [ceiling(rng.choice(rates, len(rates)), n, 200) for _ in range(300)]
        lo, hi = np.percentile([b[1] for b in boot], [2.5, 97.5])
        out.update(intra_clip_correlation=rho, label_noise_ceiling_auc=ceil,
                   label_noise_ceiling_ci=[float(lo), float(hi)])
        print(f"label noise: {out['share_clips_disagreeing']:.2f} of clips have disagreeing rollouts; intra-clip "
              f"correlation {rho:.2f}; a screen knowing each clip's true failure propensity would reach "
              f"AUC {ceil:.3f} [{lo:.3f}, {hi:.3f}] against single-rollout labels (beta-binomial fit)")
    json.dump(out, open(f"{out_prefix}.json", "w"), indent=1, default=float)
    pd.DataFrame(out["results"]).to_csv(f"{out_prefix}.csv", index=False)
    print(f"wrote {out_prefix}.json, {out_prefix}.csv")
    return out


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
    f.add_argument("--per-decision", default=None, help="glob of run_eval.py --per-decision parquets")
    f.add_argument("--out", default=os.path.join(HERE, ".skip_features.parquet"))

    for name in ("fit", "select", "predict"):
        s = sub.add_parser(name)
        s.add_argument("--features", default=os.path.join(HERE, ".skip_features.parquet"))
        s.add_argument("--policy", default="vavam")
        s.add_argument("--target", default="offroad_or_collision_at_fault")
        s.add_argument("--feature-set", default="all",
                       help="axes | openloop | openloop-self | decisions | openloop+decisions | "
                            "openloop+min | openloop+first | all | comma-separated columns")
        if name == "fit":
            s.add_argument("--budgets", default="0.1,0.2,0.3,0.4,0.5,0.6,0.8")
            s.add_argument("--out-prefix", default=os.path.join(HERE, ".skip"))
        elif name == "predict":
            s.add_argument("--budget", type=float, default=0.5, help="the dial setting to record (fraction or count)")
            s.add_argument("--out", required=True, help="frozen predictions parquet; metadata goes beside it as .json")
        else:
            s.add_argument("--budget", type=float, required=True, help="fraction (<=1) or count")
            s.add_argument("--exclude-labelled", action="store_true",
                           help="only clips without a closed-loop result (the next batch to run)")
            s.add_argument("--scene-ids", action="store_true", help="write clipgt-<id> for run_scene.sh")
            s.add_argument("--out", required=True)
    pr = sub.add_parser("prospective")
    pr.add_argument("--frozen", required=True, help="parquet written by `predict` before the batch ran")
    pr.add_argument("--closed-loop", nargs="+", required=True, help="the batch's per_clip.parquet files")
    pr.add_argument("--features", default=os.path.join(HERE, ".skip_features.parquet"),
                    help="for the MF-PDMS baseline")
    pr.add_argument("--policy", default="vavam")
    pr.add_argument("--target", default="offroad_or_collision_at_fault")
    pr.add_argument("--budgets", default="0.2,0.5,0.8")
    pr.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    if a.cmd == "prospective":
        prospective(a.frozen, a.closed_loop, a.features, a.policy, a.target,
                    [float(b) for b in a.budgets.split(",")], a.out_prefix)
    elif a.cmd == "features":
        df = build_features(a.axes_root, sorted(glob.glob(a.open_loop)), a.closed_loop, a.prod_conflict,
                            sorted(glob.glob(a.per_decision)) if a.per_decision else None)
        df.to_parquet(a.out, index=False)
        has = {c[7:]: int(df[c].sum()) for c in df.columns if c.startswith("has_cl_")}
        print(f"wrote {a.out}: {len(df)} clips, {df.shape[1]} columns; closed-loop labelled: {has}")
    elif a.cmd == "fit":
        fit(pd.read_parquet(a.features), a.policy, a.target, a.feature_set,
            [float(b) for b in a.budgets.split(",")], a.out_prefix)
    elif a.cmd == "predict":
        predict(pd.read_parquet(a.features), a.policy, a.target, a.feature_set, a.out, a.features, a.budget)
    else:
        select(pd.read_parquet(a.features), a.policy, a.target, a.feature_set, a.budget,
               a.exclude_labelled, a.out, a.scene_ids)


if __name__ == "__main__":
    main()
