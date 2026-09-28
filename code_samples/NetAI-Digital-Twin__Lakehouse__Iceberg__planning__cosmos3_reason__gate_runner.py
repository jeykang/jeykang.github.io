#!/usr/bin/env python3
"""Validity battery for the Cosmos3-Edge difficulty axis.

Deliberately the SAME battery planning/alpamayo/gate_runner.py ran, so the numbers
are directly comparable to the shelved Alpamayo result rather than to a new
yardstick:
  - parse success   : logit-EV is always numeric by construction
  - throughput      : s/clip, extrapolated to the ~33k cohort
  - OOD AUC         : vs the 452-clip human-hard labels (circularity caveat below)
  - negative control: TWO of them.
      blank : frames zeroed. Kept only for comparability with the old number --
              it is a weak control here, because a black frame is itself a
              plausible "pitch dark / zero visibility" scene, so a model with a
              dark=hard prior SCORES IT UP. Observed on the first smoke-test clip
              (real 0.767 vs blank 0.943), i.e. the control is not just weak but
              inverted, and the old +0.10/+0.21/+0.03 blank margins should be read
              with that in mind.
      swap  : frames from a DIFFERENT clip, label kept. This is the control
              planning/alpamayo/FINDINGS.md flagged as needed ("stronger negative
              control (shuffled-frame, not just blank)") and never ran. If the axis
              is scene-grounded, AUC over swapped frames must collapse toward 0.5;
              whatever AUC survives the swap is prior, not scene.
  - determinism     : same-input repeat |delta| (want 0.000)
  - convergent      : Spearman vs the production `conflict` axis

Circularity caveat carried over: `ood_reasoning` labels are Alpamayo-lineage, so
AUC here is generous to VLM-family scorers, not independent. It is kept only
because the old number was measured the same way.

Usage:  ./c3_venv/bin/python gate_runner.py [N] [--mode cold|reasoned]
"""
import argparse
import glob
import json
import os
import statistics as st
import time

import cosmos3_scorer as cs

CLIPS = os.environ.get("C3_CLIPS", "/tmp/conf/clips.txt")
_HERE = os.path.dirname(os.path.abspath(__file__))

# Reference numbers from planning/alpamayo/FINDINGS.md (same battery, same label set).
BASELINES = {
    "conflict (production)":       {"auc": 0.651, "note": "GPU-free"},
    "alpamayo cold VQA (10B)":     {"auc": 0.437, "note": "6.6 s/clip, 1 frame"},
    "alpamayo reasoned VQA (10B)": {"auc": 0.565, "note": "~8 s/clip, 1 frame"},
    "alpamayo CoC rollout (10B)":  {"auc": 0.604, "note": "66.5 s/clip, neg-control +0.03"},
}


def auc(rows, key):
    vals = [r for r in rows if r.get(key) is not None]
    pos = [r for r in vals if r["ood"]]
    neg = [r for r in vals if not r["ood"]]
    if not pos or not neg:
        return float("nan")
    al = sorted(vals, key=lambda r: r[key])
    rsum = sum(i + 1 for i, r in enumerate(al) if r["ood"])
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


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


def spearman(a, b):
    if len(a) < 8:
        return float("nan")
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def load_axis(subdir, col):
    """Load a production axis parquet keyed by clip_id."""
    import pyarrow.parquet as pq
    out = {}
    for p in glob.glob(os.path.join(cs.ROOT, subdir, "*.parquet")):
        try:
            d = pq.read_table(p, columns=["clip_id", col]).to_pydict()
            out.update(dict(zip(d["clip_id"], d[col])))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=150)
    ap.add_argument("--mode", choices=["cold", "reasoned"], default="cold")
    ap.add_argument("--model", default=None,
                    help=f"backend: {'|'.join(cs.MODELS)} or an HF id (default cosmos3-edge)")
    ap.add_argument("--n-blank", type=int, default=40, help="blank neg-control subsample")
    ap.add_argument("--n-swap", type=int, default=0,
                    help="mismatched-frame neg-control subsample (0 = all clips)")
    ap.add_argument("--n-repeat", type=int, default=10, help="determinism subsample")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.model:
        cs.MODEL = a.model
        cs.MODEL_ID = cs.MODELS.get(a.model, a.model)
    clips = [l.strip().split(",") for l in open(CLIPS) if l.strip()][:a.n]
    scorer = cs.score_clip if a.mode == "cold" else cs.reasoned_score

    print(f"[gate] loading {cs.MODEL_ID} on {cs.pick_device()} "
          f"({cs.N_FRAMES} frames @ {cs.LONG_SIDE}px, mode={a.mode})", flush=True)
    model, proc = cs.load_model()
    print(f"[gate] scoring {len(clips)} clips", flush=True)

    n_swap = a.n_swap or len(clips)
    swap_off = max(1, len(clips) // 2)   # deterministic donor, always a different clip
    rows, negs, swaps, deltas, examples = [], [], [], [], []
    blank_scores = []
    t0 = time.time()
    for k, (cid, ood) in enumerate(clips):
        try:
            frames = cs.load_frames(cid)
            r = scorer(model, proc, cid, frames=frames)
        except Exception as e:
            print(f"  [WARN] {cid[:8]}: {str(e)[:90]}", flush=True)
            rows.append({"clip": cid, "ood": int(ood), "score": None})
            continue
        rec = {"clip": cid, "ood": int(ood), "score": r["score"], "pmax": r["digit_pmax"]}
        if r.get("rationale"):
            rec["rationale"] = r["rationale"]
            if k < 3:
                examples.append((ood, round(r["score"], 3), r["rationale"][:110]))
        if k < a.n_blank:
            rb = scorer(model, proc, cid, frames=frames, blank=True)
            rec["blank"] = rb["score"]
            negs.append(r["score"] - rb["score"])
            blank_scores.append(rb["score"])
        if k < n_swap:
            donor = clips[(k + swap_off) % len(clips)][0]
            try:
                rs = scorer(model, proc, cid, frames=cs.load_frames(donor))
                rec["swap"] = rs["score"]
                swaps.append(r["score"] - rs["score"])
            except Exception as e:
                print(f"  [WARN] swap {donor[:8]}: {str(e)[:60]}", flush=True)
        if k < a.n_repeat:
            r2 = scorer(model, proc, cid, frames=frames)
            deltas.append(abs(r2["score"] - r["score"]))
        rows.append(rec)
        if (k + 1) % 20 == 0:
            print(f"[gate] {k+1}/{len(clips)} ({(time.time()-t0)/(k+1):.1f}s/clip)", flush=True)

    el = time.time() - t0
    scored = [r for r in rows if r.get("score") is not None]
    conflict = load_axis(".conflict", "conflict_score")
    camera = load_axis(".camera_perception", "camera_low_conf")

    a_auc = auc(scored, "score")
    print(f"\n===== VLM DIFFICULTY GATE  model={cs.MODEL}  N={len(rows)}  mode={a.mode} =====")
    print(f"parse success    : {len(scored)}/{len(rows)} = "
          f"{100*len(scored)/max(1,len(rows)):.0f}%  (logit-EV: numeric by construction)")
    print(f"throughput       : {el/max(1,len(rows)):.2f}s/clip"
          f"  -> ~33k cohort ~= {33000*el/max(1,len(rows))/3600:.1f}h single GPU")
    if scored:
        ss = [r["score"] for r in scored]
        print(f"score range      : min={min(ss):.3f} max={max(ss):.3f} "
              f"mean={st.mean(ss):.3f} sd={st.pstdev(ss):.3f}")
    print(f"OOD AUC          : {a_auc:.3f}")
    if negs:
        print(f"neg-control      : real-blank mean={st.mean(negs):+.3f} "
              f"(blank mean={st.mean(blank_scores):.3f}; "
              f"{sum(1 for d in negs if d>0)}/{len(negs)} real>blank)  [want >0]")
    if swaps:
        sw = [r for r in scored if r.get("swap") is not None]
        swap_auc = auc([{"ood": r["ood"], "score": r["swap"]} for r in sw], "score")
        print(f"neg-control SWAP : real-swap mean={st.mean(swaps):+.3f} "
              f"({sum(1 for d in swaps if d>0)}/{len(swaps)} real>swap)")
        print(f"                   AUC on swapped frames={swap_auc:.3f} vs real={a_auc:.3f} "
              f"-> scene-attributable={a_auc-swap_auc:+.3f}  [want swap ~0.500]")
    if deltas:
        print(f"determinism      : max|delta|={max(deltas):.4f} over {len(deltas)} repeats")
    for name, col, src in ((".conflict", "conflict_score", conflict),
                           (".camera_perception", "camera_low_conf", camera)):
        pairs = [(r["score"], src[r["clip"]]) for r in scored if r["clip"] in src]
        if len(pairs) >= 8:
            rho = spearman([x for x, _ in pairs], [y for _, y in pairs])
            print(f"convergent       : spearman(cosmos3, {col})={rho:+.3f} (n={len(pairs)})")

    print("\n--- reference (planning/alpamayo/FINDINGS.md, same battery) ---")
    for k_, v in BASELINES.items():
        flag = ""
        if a_auc == a_auc:
            flag = "  <-- BEATEN" if a_auc > v["auc"] else ""
        print(f"  {k_:<28} AUC {v['auc']:.3f}   {v['note']}{flag}")

    for o, s, r in examples:
        print(f"  ood={o} score={s} :: {r!r}")

    tag = cs.MODEL.replace("/", "_")
    out = a.out or os.path.join(_HERE, f".gate_{tag}_{a.mode}_{len(rows)}.json")
    json.dump(rows, open(out, "w"))
    print(f"\nwrote {out}\n>>> GATE DONE")


if __name__ == "__main__":
    main()
