#!/usr/bin/env python3
"""Unblind and score the human spot-review.

Decisive test: strata A (cosmos3-hard / conflict-easy) and B (both-easy) are BOTH
conflict-easy, so a difference in human ratings between them is attributable to
cosmos3 alone — and it is measured against human judgement, not against the
Alpamayo-lineage labels every other number in RESULTS.md depends on.

Guard first: the both-hard anchor must rate hardest. If it does not, the rating scale
did not work and nothing else here should be read.

Usage: ./c3_venv/bin/python score_review.py ratings.json
"""
import json
import os
import random
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(_HERE, ".review_key.json")
N_BOOT = 10000
SEED = 0
LABELS = {0: "trivial", 1: "easy", 2: "moderate", 3: "hard"}


def mannwhitney_u(a, b):
    """Rank-biserial + normal-approx p, ties handled. Small-n friendly."""
    allv = sorted(a + b)
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2 + 1
        ranks[allv[i]] = r
        i = j + 1
    ra = sum(ranks[x] for x in a)
    na, nb = len(a), len(b)
    u = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = (na * nb * (na + nb + 1) / 12) ** 0.5
    z = (u - mu) / sd if sd else 0.0
    # two-sided normal approximation
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / 2 ** 0.5)))
    return u, z, p, u / (na * nb)   # last = common-language effect size


def erf(x):
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * pow(2.718281828, -x * x)
    return y


def boot_diff(a, b, n=N_BOOT):
    rng = random.Random(SEED)
    ds = []
    for _ in range(n):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        ds.append(st.mean(sa) - st.mean(sb))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


def spearman(a, b):
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def auc(scores, labels):
    pos = sum(labels); neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rsum = sum(r + 1 for r, i in enumerate(order) if labels[i])
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ratings = json.load(open(sys.argv[1]))
    key = {r["clip"]: r for r in json.load(open(KEY))}
    rows = [{**key[c], "human": v} for c, v in ratings.items() if c in key]
    missing = [c for c in key if c not in ratings]
    print(f"===== HUMAN SPOT-REVIEW (rated {len(rows)}/{len(key)}) =====")
    if missing:
        print(f"  [warn] {len(missing)} clips unrated — excluded")

    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r["human"])

    print("\n  mean human difficulty by stratum (0 trivial .. 3 hard):")
    for s in ("A_c3hard_cfeasy", "B_c3easy_cfeasy", "anchor_bothhard", "rev_cfhard_c3easy"):
        v = by.get(s, [])
        if v:
            dist = "/".join(str(sum(1 for x in v if x == k)) for k in range(4))
            print(f"    {s:<22} n={len(v):3d}  mean={st.mean(v):.2f}  "
                  f"[counts 0/1/2/3 = {dist}]")

    # ---- scale-validity guard ----
    anc, ctrl = by.get("anchor_bothhard", []), by.get("B_c3easy_cfeasy", [])
    if anc and ctrl:
        ok = st.mean(anc) > st.mean(ctrl)
        print(f"\n  scale guard: both-hard anchor {st.mean(anc):.2f} vs both-easy "
              f"{st.mean(ctrl):.2f} -> {'PASS' if ok else 'FAIL — ratings unusable'}")
        if not ok:
            return

    # ---- the decisive contrast ----
    a, b = by.get("A_c3hard_cfeasy", []), by.get("B_c3easy_cfeasy", [])
    if a and b:
        d = st.mean(a) - st.mean(b)
        lo, hi = boot_diff(a, b)
        u, z, p, cles = mannwhitney_u(a, b)
        print(f"\n  DECISIVE  A (cosmos3-hard, conflict-easy) vs B (both-easy):")
        print(f"    mean difference   {d:+.2f}   bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"    Mann-Whitney      U={u:.0f}  z={z:+.2f}  p={p:.4f}")
        print(f"    effect size       {cles:.2f} (P[random A rated harder than random B])")
        v = ("CONFIRMED — cosmos3's unique flag tracks human-judged difficulty"
             if lo > 0 else
             "NOT CONFIRMED — cannot distinguish A from B on human judgement")
        print(f"    verdict           {v}")

    # ---- secondary: does human judgement track the model, and the labels? ----
    hu = [r["human"] for r in rows]
    print(f"\n  secondary:")
    print(f"    spearman(human, cosmos3 rank) = {spearman(hu, [r['c3'] for r in rows]):+.3f}")
    print(f"    spearman(human, conflict rank) = {spearman(hu, [r['cf'] for r in rows]):+.3f}")
    print(f"    AUC(human -> ood_reasoning label) = {auc(hu, [r['ood'] for r in rows]):.3f}"
          f"   [how well the HUMAN reproduces the label set]")
    print(f"    AUC(cosmos3 -> ood) on these 60 = {auc([r['c3'] for r in rows], [r['ood'] for r in rows]):.3f}")

    out = os.path.join(_HERE, ".review_scored.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
