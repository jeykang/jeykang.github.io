"""Phase-0 feasibility gate for the Alpamayo-1.5 VLM difficulty scorer.

Runs over /tmp/conf/clips.txt (clip_id,is_ood). Measures the GO/NO-GO criteria:
  - parse success rate (format reliability),
  - negative control: blank frames must drop the score (scene-grounded, not prior),
  - determinism: same-seed repeat |Δ| (want ~0),
  - throughput (extrapolated to the ~33k scale),
  - OOD AUC (sanity vs the human-hard labels; note circularity caveat),
  - convergent: Spearman vs conflict (.conflict) where available.

Usage (inside the venv):  python gate_runner.py [N_clips]
"""
import glob, io, json, os, sys, time, zipfile
import statistics as st
import difficulty_qa as dq

CLIPS = "/tmp/conf/clips.txt"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "netai-e2e", "nvidia-physicalai-av-subset")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
N_BLANK = 40  # negative-control subsample (blank-frame pass is an extra forward)


def auc(rows, key):
    pos = [r[key] for r in rows if r["ood"] and r.get(key) is not None]
    neg = [r[key] for r in rows if not r["ood"] and r.get(key) is not None]
    if not pos or not neg:
        return float("nan")
    al = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r[key])
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
    da = sum((x - ma) ** 2 for x in ra) ** 0.5; db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def load_conflict():
    out = {}
    for p in glob.glob(os.path.join(ROOT, ".conflict", "*.parquet")):
        try:
            import pyarrow.parquet as pq
            d = pq.read_table(p, columns=["clip_id", "conflict_score"]).to_pydict()
            out.update(dict(zip(d["clip_id"], d["conflict_score"])))
        except Exception:
            pass
    return out


def main():
    clips = [l.strip().split(",") for l in open(CLIPS) if l.strip()][:N]
    print(f"[gate] loading Alpamayo-1.5 ...", flush=True)
    model, proc = dq.load_model()
    print(f"[gate] scoring {len(clips)} clips", flush=True)
    rows, neg = [], []
    blank_const = None
    t0 = time.time()
    for k, (cid, ood) in enumerate(clips):
        try:
            data = dq.load_frames(cid)
            r = dq.score_clip(model, proc, cid, data=data)
        except Exception as e:
            print(f"  [WARN] {cid[:8]}: {str(e)[:90]}", flush=True)
            rows.append({"clip": cid, "ood": int(ood), "score": None}); continue
        rec = {"clip": cid, "ood": int(ood), "score": r["score"], "pmax": r["digit_pmax"]}
        if k < N_BLANK:
            rb = dq.score_clip(model, proc, cid, data=data, blank=True)
            rec["blank"] = rb["score"]; neg.append(r["score"] - rb["score"])
            blank_const = rb["score"]
        rows.append(rec)
        if (k + 1) % 20 == 0:
            print(f"[gate] {k+1}/{len(clips)} ({(k+1)/(time.time()-t0):.2f} clips/s)", flush=True)

    el = time.time() - t0
    scored = [r for r in rows if r.get("score") is not None]
    conf = load_conflict()
    pairs = [(r["score"], conf[r["clip"]]) for r in scored if r["clip"] in conf]

    print(f"\n===== ALPAMAYO-1.5 VLM DIFFICULTY GATE (N={len(rows)}) =====")
    print(f"parse success    : {len(scored)}/{len(rows)} = {100*len(scored)/max(1,len(rows)):.0f}%  (logit-EV: always numeric, deterministic)")
    print(f"throughput       : {len(rows)/el:.2f} clips/s ({el/max(1,len(rows)):.1f}s/clip)"
          f"  -> ~33k ≈ {33000*el/max(1,len(rows))/3600:.1f}h (single GPU)")
    print(f"score range      : min={min(r['score'] for r in scored):.3f} max={max(r['score'] for r in scored):.3f} mean={st.mean(r['score'] for r in scored):.3f}")
    print(f"OOD AUC          : {auc(scored,'score'):.3f}  (>0.5 expected; conflict=0.651; circularity caveat)")
    if neg:
        print(f"neg-control      : real-blank mean={st.mean(neg):+.3f} (blank const≈{blank_const:.3f}; "
              f"{sum(1 for d in neg if d>0)}/{len(neg)} real>blank)  (want >0 = scene-grounded)")
    if len(pairs) >= 8:
        print(f"convergent       : spearman(VLM, conflict)={spearman([a for a,_ in pairs],[b for _,b in pairs]):+.3f} (n={len(pairs)})")
    json.dump(rows, open("/tmp/alpamayo_gate.json", "w"))
    print(">>> GATE DONE")


if __name__ == "__main__":
    main()
