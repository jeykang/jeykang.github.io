"""Validity battery for the EXISTING Gold difficulty sub-scores.

Runs the same external-label + convergent/discriminant tests that refuted
`mode_spread` and confirmed `agent-conflict`, but on the dimensions already in
production: time_of_day, season_geography, sensor_coverage, ego_dynamics,
perception, conflict (+ the composite difficulty_score).

- External-label OOD AUC: rank each sub-score against the 1,740 human-flagged
  hard clips (ood_reasoning). AUC > 0.5 = aligns with "hard"; ~0.5 = no signal;
  < 0.5 = ANTI-aligned (invalid, like mode_spread's 0.37).
- Convergent/discriminant: Spearman among sub-scores + each vs the composite
  (a dimension ~0 with everything contributes nothing; one ~1.0 with another is
  redundant).

Reads sub-scores from iceberg.nvidia_gold.clip_scores `detail` JSON (no re-score
needed). OOD ids from a one-per-line file. Pure-python stats (no scipy).
"""
import json, sys
sys.path.insert(0, "/opt/spark")
from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session

OOD_FILE = "/opt/spark/nvidia_ingestion/_ood_clips.txt"
DIMS = ["time_of_day", "season_geography", "sensor_coverage",
        "ego_dynamics", "perception", "conflict"]


def auc(pairs):
    """pairs = list of (value, is_ood). Mann-Whitney AUC = P(ood ranks above non)."""
    pos = [v for v, o in pairs if o]; neg = [v for v, o in pairs if not o]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    al = sorted(pairs, key=lambda p: p[0])
    rsum = sum(i + 1 for i, p in enumerate(al) if p[1])
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), len(pos), len(neg)


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main():
    config = NvidiaPipelineConfig()
    spark = build_spark_session(config, app_name="validity-battery")
    ood = set(l.strip() for l in open(OOD_FILE) if l.strip())
    print(f"[battery] {len(ood)} OOD clip ids loaded", flush=True)

    rows = spark.table(f"{config.spark_catalog_name}.{config.nvidia.namespace_gold}.clip_scores") \
        .select("clip_id", "difficulty_score", "detail").collect()
    print(f"[battery] {len(rows)} scored clips", flush=True)

    # parse: clip -> {dim: val}, composite, ood
    recs = []
    for r in rows:
        try:
            sub = json.loads(r["detail"]).get("sub_scores", {})
        except Exception:
            sub = {}
        recs.append((r["clip_id"], float(r["difficulty_score"]), sub, r["clip_id"] in ood))
    n_ood = sum(1 for *_ , o in recs if o)
    print(f"[battery] OOD overlap in scored set: {n_ood} / {len(recs)}\n", flush=True)

    print("===== EXTERNAL-LABEL OOD AUC (per sub-score) =====")
    print(f"  {'dimension':18s} {'AUC':>6s}  {'n_ood':>6s} {'n_non':>7s}   verdict")
    results = {}
    for dim in DIMS + ["difficulty_score"]:
        if dim == "difficulty_score":
            pairs = [(c, o) for _, c, _, o in recs]
        else:
            pairs = [(s[dim], o) for _, _, s, o in recs if s.get(dim) is not None]
        a, npos, nneg = auc(pairs)
        results[dim] = a
        if a != a:
            verdict = "no data"
        elif a < 0.45:
            verdict = "!! ANTI-ALIGNED (invalid)"
        elif a < 0.52:
            verdict = "~ weak / no signal"
        elif a < 0.60:
            verdict = "ok (modest)"
        else:
            verdict = "GOOD"
        print(f"  {dim:18s} {a:6.3f}  {npos:6d} {nneg:7d}   {verdict}")

    print("\n===== CONVERGENT / DISCRIMINANT (Spearman) =====")
    # use clips where ALL of the always-present metadata dims exist; perception/
    # conflict correlated on their own support
    base = [(c, comp, s) for c, comp, s, o in recs]
    cols = {}
    for dim in DIMS:
        cols[dim] = [(s.get(dim), comp) for _, comp, s in base]
    print(f"  {'dimension':18s} {'rho_vs_composite':>17s}")
    for dim in DIMS:
        xy = [(v, comp) for v, comp in cols[dim] if v is not None]
        if len(xy) > 10:
            rho = spearman([v for v, _ in xy], [comp for _, comp in xy])
            print(f"  {dim:18s} {rho:17.3f}")
    # pairwise among always-present metadata dims
    meta = ["time_of_day", "season_geography", "sensor_coverage", "ego_dynamics"]
    print(f"\n  pairwise (metadata dims):")
    vals = {d: [s.get(d) for _, _, s in base] for d in meta}
    for i, di in enumerate(meta):
        for dj in meta[i + 1:]:
            xy = [(vals[di][k], vals[dj][k]) for k in range(len(base))
                  if vals[di][k] is not None and vals[dj][k] is not None]
            if len(xy) > 10:
                rho = spearman([a for a, _ in xy], [b for _, b in xy])
                print(f"    {di:18s} ~ {dj:18s} {rho:6.3f}")
    print("\n>>> BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
