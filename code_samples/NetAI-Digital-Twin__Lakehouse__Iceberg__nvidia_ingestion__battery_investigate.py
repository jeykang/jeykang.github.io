"""Follow-up investigation before re-weighting Gold (per battery findings).

(1) AUCs on the lidar-covered subset (where conflict/perception exist) + a
    conflict-centered what-if composite — the achievable ceiling on covered clips.
(2) Debug sensor_coverage anti-alignment (ood vs non distribution; are hard clips
    just full-rig clips?).
(3) Debug season_geography ~ ego_dynamics rho=-0.844 (value degeneracy?).

Reads iceberg.nvidia_gold.clip_scores detail JSON. Pure-python stats.
"""
import json, sys
from collections import Counter
sys.path.insert(0, "/opt/spark")
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

OOD_FILE = "/opt/spark/nvidia_ingestion/_ood_clips.txt"
DIMS = ["time_of_day", "season_geography", "sensor_coverage",
        "ego_dynamics", "perception", "conflict"]
# conflict-centered what-if weights (renormalized over present dims)
WHATIF = {"conflict": 0.45, "perception": 0.25, "ego_dynamics": 0.10,
          "time_of_day": 0.10, "season_geography": 0.05, "sensor_coverage": 0.05}


def auc(pairs):
    pos = [v for v, o in pairs if o]; neg = [v for v, o in pairs if not o]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    al = sorted(pairs, key=lambda p: p[0])
    rsum = sum(i + 1 for i, p in enumerate(al) if p[1])
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), len(pos), len(neg)


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return (sum(xs) / n, xs[n // 2], xs[0], xs[-1]) if n else (0, 0, 0, 0)


def main():
    config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="battery-investigate")
    ood = set(l.strip() for l in open(OOD_FILE) if l.strip())
    rows = spark.table(f"{config.spark_catalog_name}.{config.nvidia.namespace_gold}.clip_scores") \
        .select("clip_id", "difficulty_score", "detail").collect()
    recs = []
    for r in rows:
        try:
            sub = json.loads(r["detail"]).get("sub_scores", {})
        except Exception:
            sub = {}
        recs.append((r["clip_id"], float(r["difficulty_score"]), sub, r["clip_id"] in ood))
    print(f"[inv] {len(recs)} clips, {sum(o for *_,o in recs)} ood\n", flush=True)

    # ---- (1) lidar-covered subset (conflict present) ----
    lid = [(c, comp, s, o) for c, comp, s, o in recs if s.get("conflict") is not None]
    print(f"===== (1) LIDAR-COVERED SUBSET (conflict present): {len(lid)} clips, "
          f"{sum(o for *_,o in lid)} ood =====")
    for dim in DIMS + ["difficulty_score"]:
        if dim == "difficulty_score":
            pairs = [(comp, o) for _, comp, s, o in lid]
        else:
            pairs = [(s[dim], o) for _, _, s, o in lid if s.get(dim) is not None]
        a, npos, nneg = auc(pairs)
        print(f"  {dim:18s} AUC={a:.3f}  (n_ood={npos})")
    # conflict-centered what-if composite on this subset
    wpairs = []
    for _, _, s, o in lid:
        num = den = 0.0
        for d, w in WHATIF.items():
            if s.get(d) is not None:
                num += w * s[d]; den += w
        if den:
            wpairs.append((num / den, o))
    a, npos, _ = auc(wpairs)
    print(f"  {'WHATIF(conflict-centered)':18s} AUC={a:.3f}  (n_ood={npos})  "
          f"weights={WHATIF}")

    # ---- (2) sensor_coverage debug ----
    print(f"\n===== (2) sensor_coverage debug =====")
    sc_ood = [s["sensor_coverage"] for _, _, s, o in recs if o and s.get("sensor_coverage") is not None]
    sc_non = [s["sensor_coverage"] for _, _, s, o in recs if not o and s.get("sensor_coverage") is not None]
    print(f"  ood : mean={stats(sc_ood)[0]:.3f} median={stats(sc_ood)[1]:.3f}")
    print(f"  non : mean={stats(sc_non)[0]:.3f} median={stats(sc_non)[1]:.3f}")
    hist = Counter(round(s["sensor_coverage"], 3) for _, _, s, _ in recs if s.get("sensor_coverage") is not None)
    print(f"  value histogram (all): {dict(sorted(hist.items()))}")
    hist_o = Counter(round(v, 3) for v in sc_ood)
    print(f"  value histogram (ood): {dict(sorted(hist_o.items()))}")

    # ---- (3) season_geography ~ ego_dynamics degeneracy ----
    print(f"\n===== (3) season_geography & ego_dynamics distributions =====")
    seas = [s["season_geography"] for _, _, s, _ in recs if s.get("season_geography") is not None]
    ego = [s["ego_dynamics"] for _, _, s, _ in recs if s.get("ego_dynamics") is not None]
    print(f"  season_geography: distinct={len(set(round(x,3) for x in seas))} "
          f"hist={dict(sorted(Counter(round(x,3) for x in seas).most_common(8)))}")
    print(f"  ego_dynamics    : distinct={len(set(round(x,3) for x in ego))} "
          f"mean={stats(ego)[0]:.3f} median={stats(ego)[1]:.3f} "
          f"top={dict(sorted(Counter(round(x,2) for x in ego).most_common(6)))}")
    # joint: mean ego per season value
    joint = {}
    for _, _, s, _ in recs:
        if s.get("season_geography") is not None and s.get("ego_dynamics") is not None:
            joint.setdefault(round(s["season_geography"], 3), []).append(s["ego_dynamics"])
    print("  mean ego_dynamics by season_geography value:")
    for k in sorted(joint):
        v = joint[k]
        print(f"    season={k:.3f}  n={len(v):7d}  mean_ego={sum(v)/len(v):.3f}")
    print("\n>>> INVESTIGATE DONE", flush=True)


if __name__ == "__main__":
    main()
