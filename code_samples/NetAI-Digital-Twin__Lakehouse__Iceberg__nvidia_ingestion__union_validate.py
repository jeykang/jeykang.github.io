"""Confirm the noisy-OR union behaves as intended for edge-case mining.

- behavioral axis (conflict) still validates on the (behavioral) OOD labels.
- composite AUC vs OOD is LOWER than conflict-alone BY DESIGN (it now also keeps
  perceptually-hard clips the daytime-only OOD set doesn't label).
- the dark-clip inversion is FIXED: spearman(darkness, composite) flips from
  negative (conflict-alone) to positive.
- the union RESCUES perceptually-hard clips conflict-alone would have stripped.
"""
import json, sys
sys.path.insert(0, "/opt/spark")
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

OOD_FILE = "/opt/spark/nvidia_ingestion/_ood_clips.txt"


def auc(pairs):
    pos = [v for v, o in pairs if o]; neg = [v for v, o in pairs if not o]
    if not pos or not neg:
        return float("nan")
    al = sorted(pairs, key=lambda p: p[0])
    rsum = sum(i + 1 for i, p in enumerate(al) if p[1])
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
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5; db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def pct(xs, q):
    s = sorted(xs); return s[min(len(s) - 1, int(len(s) * q))]


def main():
    config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="union-validate")
    ood = set(l.strip() for l in open(OOD_FILE) if l.strip())
    rows = spark.table(f"{config.spark_catalog_name}.{config.nvidia.namespace_gold}.clip_scores") \
        .select("clip_id", "difficulty_score", "detail", "sensor_covered").collect()
    R = []
    for r in rows:
        if not r["sensor_covered"]:
            continue
        s = json.loads(r["detail"]).get("sub_scores", {})
        R.append(dict(comp=float(r["difficulty_score"]), ood=r["clip_id"] in ood,
                      tod=s.get("time_of_day") or 0.0, conf=s.get("conflict") or 0.0,
                      perc=s.get("perceptual_axis") or 0.0))
    n = len(R); nood = sum(r["ood"] for r in R)
    print(f"[union] covered={n}, ood overlap={nood}\n")

    print("===== AUC vs OOD (behavioral labels; covered tier) =====")
    print(f"  conflict (behavioral)  AUC={auc([(r['conf'], r['ood']) for r in R]):.3f}  (should hold ~0.65)")
    print(f"  perceptual axis        AUC={auc([(r['perc'], r['ood']) for r in R]):.3f}  (expected <0.5: OOD is daytime)")
    print(f"  composite (noisy-OR)   AUC={auc([(r['comp'], r['ood']) for r in R]):.3f}  (BY DESIGN < conflict: also keeps perceptual-hard)")

    print("\n===== dark-clip inversion fixed? (spearman vs darkness) =====")
    tod = [r["tod"] for r in R]
    print(f"  spearman(darkness, conflict)   = {spearman(tod,[r['conf'] for r in R]):+.3f}  (was negative: dark = agent-sparse)")
    print(f"  spearman(darkness, composite)  = {spearman(tod,[r['comp'] for r in R]):+.3f}  (should be POSITIVE now)")

    print("\n===== Gold tier composition (top 10% composite) =====")
    thr_c = pct([r["comp"] for r in R], 0.90)
    thr_conf = pct([r["conf"] for r in R], 0.90)  # conflict-only threshold
    gold = [r for r in R if r["comp"] >= thr_c]
    dark = [r for r in gold if r["tod"] >= 0.7]
    hiconf = [r for r in gold if r["conf"] >= 0.7]
    rescued = [r for r in gold if r["conf"] < thr_conf]   # in by union, not by conflict-only
    print(f"  Gold clips: {len(gold)}")
    print(f"    dark (time_of_day>=0.7):        {len(dark)} ({100*len(dark)/len(gold):.0f}%)")
    print(f"    high-conflict (>=0.7):          {len(hiconf)} ({100*len(hiconf)/len(gold):.0f}%)")
    print(f"    RESCUED by union (conflict below conflict-only top-10%): "
          f"{len(rescued)} ({100*len(rescued)/len(gold):.0f}%)")
    print("\n>>> UNION VALIDATE DONE", flush=True)


if __name__ == "__main__":
    main()
