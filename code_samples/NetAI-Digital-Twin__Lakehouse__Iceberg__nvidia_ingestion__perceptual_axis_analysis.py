"""Is the perceptual axis (darkness) real and currently under-served?

(A) Per-cluster OOD AUC: which sub-score predicts which event_cluster. Shows the
    facets — conflict should own the agent clusters; if NO cluster is perceptual,
    ood_reasoning can't validate darkness at all (by construction).
(B) Darkness -> perception degradation: correlate time_of_day (darkness) with the
    raw BEVFusion stats (mean_max_conf, mean_n_detections) over the perception
    cohort. If confidence/detections drop in the dark, darkness is a genuine
    perceptual-difficulty axis in OUR data — and we check whether the current
    perception_score + conflict capture it or invert it.

Reads gold.clip_scores detail (time_of_day, hour, conflict, perception sub-scores)
+ .perception parquet (raw stats) + _ood_clusters.csv (clip -> event_cluster).
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, "/opt/spark")
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

CLUSTERS_FILE = "/opt/spark/nvidia_ingestion/_ood_clusters.csv"


def auc(pairs):
    pos = [v for v, o in pairs if o]; neg = [v for v, o in pairs if not o]
    if not pos or not neg:
        return float("nan"), len(pos)
    al = sorted(pairs, key=lambda p: p[0])
    rsum = sum(i + 1 for i, p in enumerate(al) if p[1])
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), len(pos)


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 10:
        return float("nan")
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5; db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="perceptual-axis")
    source = config.nvidia.source_path

    # clip -> cluster
    clus = {}
    for ln in open(CLUSTERS_FILE):
        cid, _, c = ln.strip().partition(",")
        if cid:
            clus[cid] = c
    ood = set(clus)

    # gold.clip_scores detail
    rows = spark.table(f"{config.spark_catalog_name}.{config.nvidia.namespace_gold}.clip_scores") \
        .select("clip_id", "difficulty_score", "detail").collect()
    sc = {}
    for r in rows:
        try:
            d = json.loads(r["detail"]); sub = d.get("sub_scores", {})
        except Exception:
            d, sub = {}, {}
        sc[r["clip_id"]] = {"comp": float(r["difficulty_score"]), "hour": d.get("hour"),
                            "time_of_day": sub.get("time_of_day"),
                            "conflict": sub.get("conflict"), "perception": sub.get("perception")}
    print(f"[perc] {len(sc)} scored clips, {sum(1 for c in sc if c in ood)} ood overlap\n", flush=True)

    # ---- (A) per-cluster AUC ----
    print("===== (A) per-cluster OOD AUC (positives = cluster; negatives = non-ood) =====")
    print(f"  {'cluster':38s} {'n_on_disk':>9s} {'time_of_day':>11s} {'conflict':>9s} {'composite':>10s}")
    by_clu = defaultdict(list)
    for cid, c in clus.items():
        by_clu[c].append(cid)
    nonood_ids = [cid for cid in sc if cid not in ood]
    for c in sorted(by_clu, key=lambda k: -len(by_clu[k])):
        pos = [cid for cid in by_clu[c] if cid in sc]
        out = {}
        for dim in ["time_of_day", "conflict", "comp"]:
            pairs = [(sc[cid][dim], True) for cid in pos if sc[cid].get(dim) is not None]
            pairs += [(sc[cid][dim], False) for cid in nonood_ids if sc[cid].get(dim) is not None]
            a, npos = auc(pairs); out[dim] = (a, npos)
        nd = out["conflict"][1]  # on-disk count (conflict present)
        print(f"  {c:38s} {nd:9d} {out['time_of_day'][0]:11.3f} "
              f"{out['conflict'][0]:9.3f} {out['comp'][0]:10.3f}")

    # ---- (B) darkness -> perception degradation ----
    print("\n===== (B) darkness vs perception (BEVFusion cohort) =====")
    pdir = os.path.join(source, ".perception")
    paths = [f"file://{os.path.join(pdir, f)}" for f in sorted(os.listdir(pdir)) if f.endswith(".parquet")]
    pr = spark.read.parquet(*paths).select(
        "clip_id", "mean_n_detections", "mean_max_conf", "driving_obj_count", "perception_score").collect()
    P = {r["clip_id"]: r for r in pr}
    # join with time_of_day / hour
    tod, conf, ndet, pscore, confl = [], [], [], [], []
    for cid, r in P.items():
        s = sc.get(cid)
        if not s or s.get("time_of_day") is None:
            continue
        tod.append(s["time_of_day"]); conf.append(r["mean_max_conf"])
        ndet.append(r["mean_n_detections"]); pscore.append(r["perception_score"])
        confl.append(s["conflict"] if s.get("conflict") is not None else 0.0)
    n = len(tod)
    print(f"  cohort n={n}")
    print(f"  spearman(time_of_day, mean_max_conf)     = {spearman(tod, conf):+.3f}   (neg = dark degrades confidence)")
    print(f"  spearman(time_of_day, mean_n_detections) = {spearman(tod, ndet):+.3f}   (neg = dark = fewer detections)")
    print(f"  spearman(time_of_day, perception_score)  = {spearman(tod, pscore):+.3f}   (does the damper call dark harder?)")
    print(f"  spearman(time_of_day, conflict)          = {spearman(tod, confl):+.3f}   (are dark clips agent-sparse?)")
    # day vs dark buckets (median split on time_of_day)
    order = sorted(range(n), key=lambda i: tod[i]); half = n // 2
    day, dark = order[:half], order[half:]
    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    print(f"\n  DAY  (low time_of_day):  mean_conf={mean([conf[i] for i in day]):.3f} "
          f"mean_det={mean([ndet[i] for i in day]):.2f} perception_score={mean([pscore[i] for i in day]):.3f}")
    print(f"  DARK (high time_of_day): mean_conf={mean([conf[i] for i in dark]):.3f} "
          f"mean_det={mean([ndet[i] for i in dark]):.2f} perception_score={mean([pscore[i] for i in dark]):.3f}")
    print("\n>>> PERCEPTUAL ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
