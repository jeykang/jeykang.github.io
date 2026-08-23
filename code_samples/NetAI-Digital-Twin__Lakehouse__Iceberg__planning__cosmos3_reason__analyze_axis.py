#!/usr/bin/env python3
"""Does the Cosmos3-Edge axis EARN a place next to the production axes?

A better AUC alone is not the bar. The bar this repo has settled on (see
nvidia_ingestion/VALIDITY_BATTERY_FINDINGS.md and the n=40 -> N=200 collapse in
planning/alpamayo/FINDINGS.md Addendum 2) is:
  1. is it scene-grounded?           -> swapped-frame control, in gate_runner.py
  2. is it better than what we have? -> AUC vs conflict, with a bootstrap CI on the
                                        DELTA, because a point estimate is exactly
                                        what produced the 0.706 -> 0.535 false positive
  3. is it ADDITIVE?                 -> the axes combine by noisy-OR, so the question
                                        is whether OR(conflict, cosmos3) beats conflict,
                                        not whether cosmos3 beats it alone

Usage: ./c3_venv/bin/python analyze_axis.py [.gate_cold_452.json ...]
"""
import glob
import json
import os
import random
import sys

import pyarrow.parquet as pq

import cosmos3_scorer as cs

CLIPS = os.environ.get("C3_CLIPS", "/tmp/conf/clips.txt")
N_BOOT = 2000
SEED = 0


def rank_norm(vals):
    """Rank-normalize to [0,1] — the same treatment the production union applies."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    for r, i in enumerate(order):
        out[i] = r / max(1, len(vals) - 1)
    return out


def auc(scores, labels):
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rsum = sum(r + 1 for r, i in enumerate(order) if labels[i])
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)


def boot_delta(a, b, labels, n=N_BOOT):
    """Bootstrap CI for AUC(a) - AUC(b), paired over clips."""
    rng = random.Random(SEED)
    idx = range(len(labels))
    ds = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        la = [labels[i] for i in s]
        if not (0 < sum(la) < len(la)):
            continue
        ds.append(auc([a[i] for i in s], la) - auc([b[i] for i in s], la))
    ds.sort()
    return ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]


# nvidia_ingestion.edge_case_scorer._HOUR_DIFFICULTY, inlined so this module stays
# detachable (no import into the base pipeline).
_HOUR_DIFFICULTY = {**{h: 1.0 for h in range(0, 6)},
                    **{h: 0.7 for h in [6, 7, 18, 19, 20]},
                    **{h: 0.5 for h in [8, 9]},
                    **{h: 0.4 for h in [16, 17]},
                    **{h: 0.2 for h in range(10, 16)},
                    **{h: 0.8 for h in [21, 22, 23]}}


def load_darkness():
    """time_of_day axis — the OTHER half of the production perceptual leg.

    Without it the 'production' comparison is a strawman: cosmos3 plainly reacts to
    lighting and weather, so it could look additive purely by re-deriving darkness,
    which production already has for free from metadata.
    """
    t = pq.read_table(os.path.join(cs.ROOT, "metadata", "data_collection.parquet"),
                      columns=["clip_id", "hour_of_day"]).to_pydict()
    return {c: _HOUR_DIFFICULTY.get(h, 0.3)
            for c, h in zip(t["clip_id"], t["hour_of_day"])}


def load_axis(subdir, col):
    out = {}
    for p in glob.glob(os.path.join(cs.ROOT, subdir, "*.parquet")):
        try:
            d = pq.read_table(p, columns=["clip_id", col]).to_pydict()
            out.update(dict(zip(d["clip_id"], d[col])))
        except Exception:
            pass
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".gate_*.json")))
    conflict = load_axis(".conflict", "conflict_score")
    camera = load_axis(".camera_perception", "low_conf")   # agent-gated variant
    darkness = load_darkness()

    for path in paths:
        rows = [r for r in json.load(open(path)) if r.get("score") is not None]
        rows = [r for r in rows if r["clip"] in conflict and r["clip"] in camera
                and r["clip"] in darkness]
        if len(rows) < 50:
            print(f"{path}: only {len(rows)} clips with all axes — skipped")
            continue
        y = [r["ood"] for r in rows]
        c3 = rank_norm([r["score"] for r in rows])
        cf = rank_norm([conflict[r["clip"]] for r in rows])
        cam = rank_norm([camera[r["clip"]] for r in rows])
        dk = [darkness[r["clip"]] for r in rows]
        # Production: perceptual = max(darkness, camera_low_conf), then noisy-OR
        # with behavioral. Rank-normed, as the production main loop does.
        perc = rank_norm([max(a, b) for a, b in zip(dk, cam)])
        prod = [1 - (1 - a) * (1 - b) for a, b in zip(cf, perc)]
        combos = {
            "cosmos3 alone":                 c3,
            "conflict alone":                cf,
            "camera_gated alone":            cam,
            "darkness alone":                rank_norm(dk),
            "PRODUCTION OR(conflict, perc)":  prod,
            "OR(conflict, cosmos3)":         [1 - (1 - a) * (1 - b) for a, b in zip(cf, c3)],
            "PRODUCTION + cosmos3":          [1 - (1 - a) * (1 - b) for a, b in zip(prod, c3)],
        }
        print(f"\n===== {os.path.basename(path)}  (n={len(rows)}, "
              f"{sum(y)} OOD / {len(y)-sum(y)} not) =====")
        for name, v in combos.items():
            print(f"  {name:<32} AUC {auc(v, y):.3f}")

        print(f"\n  bootstrap 95% CI on AUC delta ({N_BOOT} resamples):")
        for name, base in (("cosmos3 - conflict", "conflict alone"),
                           ("OR(conflict,cosmos3) - conflict", "conflict alone"),
                           ("PRODUCTION+cosmos3 - PRODUCTION", "PRODUCTION OR(conflict, perc)")):
            test = combos["cosmos3 alone"] if name.startswith("cosmos3") else (
                combos["OR(conflict, cosmos3)"] if name.startswith("OR(conflict,cosmos3)")
                else combos["PRODUCTION + cosmos3"])
            lo, hi = boot_delta(test, combos[base], y)
            verdict = "SIGNIFICANT" if lo > 0 else ("negative" if hi < 0 else "not significant")
            print(f"    {name:<38} [{lo:+.3f}, {hi:+.3f}]  {verdict}")

        # Where does cosmos3 disagree with conflict? That is where it can add value.
        hi_c3_lo_cf = [r for r, a, b in zip(rows, c3, cf) if a > 0.75 and b < 0.25]
        hi_cf_lo_c3 = [r for r, a, b in zip(rows, c3, cf) if b > 0.75 and a < 0.25]
        for label, sel in (("cosmos3-hard / conflict-easy", hi_c3_lo_cf),
                           ("conflict-hard / cosmos3-easy", hi_cf_lo_c3)):
            if sel:
                print(f"  {label:<30} n={len(sel):3d}  OOD rate="
                      f"{sum(r['ood'] for r in sel)/len(sel):.2f} "
                      f"(base {sum(y)/len(y):.2f})")




# ── cross-model comparison (the circularity control) ────────────────────────
def compare(paths):
    """Are two backends measuring the same construct, or does one carry a private signal?

    The circularity worry is that Cosmos3-Edge scores well because the OOD labels are
    Alpamayo/Cosmos-Reason2-lineage. If that were the whole story, an out-of-family
    model would show no union lift, and Cosmos3's signal would be largely ORTHOGONAL
    to the out-of-family one. If instead the two agree strongly and Cosmos3's private
    residual carries little AUC, the shared construct is doing the work and the family
    gap is capability, not leakage.
    """
    conflict = load_axis(".conflict", "conflict_score")
    camera = load_axis(".camera_perception", "low_conf")
    darkness = load_darkness()
    runs = {}
    for p in paths:
        rows = {r["clip"]: r for r in json.load(open(p)) if r.get("score") is not None}
        runs[os.path.basename(p).replace(".gate_", "").replace(".json", "")] = rows
    common = set.intersection(*[set(r) for r in runs.values()])
    common &= set(conflict) & set(camera) & set(darkness)
    common = sorted(common)
    names = list(runs)
    any_run = runs[names[0]]
    y = [any_run[c]["ood"] for c in common]
    S = {n: rank_norm([runs[n][c]["score"] for c in common]) for n in names}
    cf = rank_norm([conflict[c] for c in common])
    perc = rank_norm([max(darkness[c], camera[c]) for c in common])
    prod = [1 - (1 - a) * (1 - b) for a, b in zip(cf, perc)]

    print(f"\n===== CROSS-MODEL (n={len(common)}) =====")
    for n in names:
        print(f"  {n:<42} AUC {auc(S[n], y):.3f}")
    print("\n  agreement between backends (Spearman):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"    {a} <-> {b}: {spearman_rank(S[a], S[b]):+.3f}")

    if len(names) == 2:
        a, b = names
        ens = [(x + z) / 2 for x, z in zip(S[a], S[b])]
        print(f"\n  ensemble (mean rank)                       AUC {auc(ens, y):.3f}")
        lo, hi = boot_delta([1 - (1 - p_) * (1 - e) for p_, e in zip(prod, ens)], prod, y)
        print(f"  PRODUCTION+ensemble - PRODUCTION           [{lo:+.3f}, {hi:+.3f}]"
              f"  {'SIGNIFICANT' if lo > 0 else 'not significant'}")
        for x, other in ((a, b), (b, a)):
            res = residual(S[x], S[other])
            print(f"  {x} residual after removing {other}: AUC {auc(res, y):.3f}")


def spearman_rank(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def residual(a, b):
    """a with its least-squares projection onto b removed, then rank-normed."""
    n = len(a)
    mb = sum(b) / n
    ma = sum(a) / n
    var = sum((x - mb) ** 2 for x in b)
    beta = (sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / var) if var else 0.0
    return rank_norm([a[i] - ma - beta * (b[i] - mb) for i in range(n)])


if __name__ == "__main__":
    if "--compare" in sys.argv:
        args = [x for x in sys.argv[1:] if x != "--compare"]
        compare(args)
    else:
        main()
