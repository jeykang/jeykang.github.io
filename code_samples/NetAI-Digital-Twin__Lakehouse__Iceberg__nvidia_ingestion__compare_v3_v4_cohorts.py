"""Compare Gold v4 (with perception) vs counterfactual v3 (without perception).

v3 score for each clip is reconstructed from v4's detail JSON by dropping the
perception sub-score and renormalising the remaining active weights to 1.0.
Produces:

  * rank-correlation between v3 and v4 scores
  * Jaccard overlap between the top-10% cohort under each scheme
  * Profile of clips that moved into / out of the top cohort

Runs inside spark-iceberg.
"""
from __future__ import annotations

import json
from pyspark.sql import functions as F, types as T

from nvidia_ingestion.config import NvidiaPipelineConfig, build_spark_session
from nvidia_ingestion.edge_case_scorer import _SCENE_WEIGHTS


def _v3_udf():
    weights = dict(_SCENE_WEIGHTS)

    def _score(detail_json: str) -> float:
        if not detail_json:
            return -1.0
        try:
            d = json.loads(detail_json)
            sub = d.get("sub_scores") or {}
        except Exception:
            return -1.0
        active = {k: v for k, v in sub.items()
                  if k != "perception" and v is not None}
        if not active:
            return -1.0
        total_w = sum(weights[k] for k in active)
        return float(sum(v * weights[k] / total_w for k, v in active.items()))

    return F.udf(_score, T.DoubleType())


def main() -> None:
    cfg = NvidiaPipelineConfig()
    spark = build_spark_session(cfg, app_name="gold-v3-v4-cohort-compare")
    try:
        fq = f"{cfg.spark_catalog_name}.{cfg.nvidia.namespace_gold}.clip_scores"
        scores = spark.table(fq).filter(F.col("difficulty_score") >= 0)
        total = scores.count()
        print(f"clip_scores rows: {total:,}")

        v3 = _v3_udf()
        scored = scores.select(
            "clip_id",
            F.col("difficulty_score").alias("v4"),
            v3(F.col("detail")).alias("v3"),
            "detail",
        ).cache()

        n_with_perc = scored.filter(
            F.col("detail").contains('"perception":') &
            ~F.col("detail").contains('"perception": null')
        ).count()
        print(f"clips where v4 used perception (non-null): {n_with_perc:,}"
              f"  ({100*n_with_perc/total:.1f}%)")

        # Score distribution
        stats = scored.select(
            F.mean("v3").alias("v3_mean"), F.stddev("v3").alias("v3_std"),
            F.min("v3").alias("v3_min"), F.max("v3").alias("v3_max"),
            F.mean("v4").alias("v4_mean"), F.stddev("v4").alias("v4_std"),
            F.min("v4").alias("v4_min"), F.max("v4").alias("v4_max"),
        ).collect()[0].asDict()
        print("score distribution:")
        for k, v in stats.items():
            print(f"  {k:12s}: {v:.4f}")

        # Rank correlation (spearman ≈ pearson on ranks)
        corr = scored.stat.corr("v3", "v4")
        print(f"pearson(v3, v4) = {corr:.4f}")

        # Cohort: top 10% under each scheme
        cohort_n = int(round(total * 0.10))
        v4_thresh = scored.approxQuantile("v4", [0.90], 0.001)[0]
        v3_thresh = scored.approxQuantile("v3", [0.90], 0.001)[0]
        print(f"top-10% threshold  v4={v4_thresh:.4f}  v3={v3_thresh:.4f}")

        top_v4 = scored.filter(F.col("v4") >= v4_thresh).select("clip_id")
        top_v3 = scored.filter(F.col("v3") >= v3_thresh).select("clip_id")
        n_v4 = top_v4.count()
        n_v3 = top_v3.count()
        intersect = top_v4.intersect(top_v3).count()
        union = n_v4 + n_v3 - intersect
        jac = intersect / union if union else 0.0
        print(f"cohort sizes (approx top-10%): v4={n_v4:,}  v3={n_v3:,}")
        print(f"intersect={intersect:,}  jaccard={jac:.4f}")
        print(f"promoted by v4 (new-in-v4): {n_v4 - intersect:,}")
        print(f"demoted by v4 (only-in-v3): {n_v3 - intersect:,}")

        # Profile: clips moved INTO v4 top cohort — what sub-scores drove it?
        def _dominant(detail_json):
            if not detail_json:
                return None
            try:
                sub = (json.loads(detail_json).get("sub_scores") or {})
            except Exception:
                return None
            active = {k: v for k, v in sub.items() if v is not None}
            if not active:
                return None
            return max(active.items(), key=lambda kv: kv[1])[0]
        dom_udf = F.udf(_dominant, T.StringType())

        moved_in = (scored
            .filter((F.col("v4") >= v4_thresh) & (F.col("v3") < v3_thresh))
            .select("clip_id", "detail"))
        moved_out = (scored
            .filter((F.col("v3") >= v3_thresh) & (F.col("v4") < v4_thresh))
            .select("clip_id", "detail"))

        print("\ndominant-factor distribution — clips promoted by v4:")
        (moved_in.withColumn("dom", dom_udf("detail"))
                 .groupBy("dom").count()
                 .orderBy(F.col("count").desc())
                 .show(20, truncate=False))

        print("dominant-factor distribution — clips demoted by v4:")
        (moved_out.withColumn("dom", dom_udf("detail"))
                  .groupBy("dom").count()
                  .orderBy(F.col("count").desc())
                  .show(20, truncate=False))

        # Write a small JSON summary to user_data
        summary = {
            "total_clips": total,
            "clips_with_perception": n_with_perc,
            "v3_stats": {k: v for k, v in stats.items() if k.startswith("v3_")},
            "v4_stats": {k: v for k, v in stats.items() if k.startswith("v4_")},
            "pearson_corr": corr,
            "cohort_top10_pct": {
                "v4_threshold": v4_thresh,
                "v3_threshold": v3_thresh,
                "v4_size": n_v4,
                "v3_size": n_v3,
                "intersect": intersect,
                "jaccard": jac,
                "promoted_by_v4": n_v4 - intersect,
                "demoted_by_v4": n_v3 - intersect,
            },
        }
        with open("/user_data/v3_v4_cohort_compare.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote /user_data/v3_v4_cohort_compare.json")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
