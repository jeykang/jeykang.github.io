"""Extract per-clip AlpaSim scores from a run directory.

The published `metrics_results.txt` is averaged over every clip in the run,
which is too coarse to compare against a per-clip open-loop metric such as
MF-PDMS. AlpaSim's aggregation pipeline already computes a per-trajectory table
on the way to that summary (`df_wide_avg_t`), so this delegates to it rather
than reimplementing the modifier chain.

That chain is not optional detail: it drops timesteps before `eval_relevant`,
truncates each rollout at the first offroad/collision *and* once the ego
deviates `max_dist_to_gt_trajectory` metres from the GT path. Aggregating
without it overstates distance travelled and offroad rate.

Metrics are read from the per-rollout `metrics.parquet` files, which is the
same source `eval.aggregation.main` uses.

Must run inside the alpasim uv environment:
    uv run --project repo python per_clip.py <run_dir> [-o out.parquet]
"""

import argparse
import pathlib

import polars as pl
from eval.aggregation.modifiers import RemoveTimestepsAfterEvent
from eval.aggregation.processing import aggregate_and_write_metrics_results_txt

# Matches wizard `base_config.yaml: eval.aggregation_modifiers`.
MAX_DIST_TO_GT_TRAJECTORY = 4.0

# Metrics that are meaningful per clip; the rest are run-level or diagnostic.
PER_CLIP_METRICS = [
    "progress",
    "progress_rel",
    "progress_rel_to_total",
    "dist_traveled_m",
    "gt_dist_traveled_m",
    "dist_to_gt_trajectory",
    "dist_to_gt_location",
    "offroad",
    "collision_any",
    "collision_at_fault",
    "collision_rear",
    "offroad_or_collision",
    "offroad_or_collision_at_fault",
    "duration_frac_20s",
    "min_distance_to_obstacle_m",
    "min_distance_to_lane_boundary_m",
]


def per_clip_scores(
    run_dir: pathlib.Path,
    max_dist_to_gt_trajectory: float = MAX_DIST_TO_GT_TRAJECTORY,
) -> pl.DataFrame:
    """Return one row per (clip, rollout) with AlpaSim's own aggregation applied."""
    metrics = pl.read_parquet(str(run_dir / "rollouts" / "**" / "metrics.parquet"))

    processed = aggregate_and_write_metrics_results_txt(
        metrics,
        force_same_run=True,
        additional_modifiers=[
            RemoveTimestepsAfterEvent(
                pl.col("dist_to_gt_trajectory") >= max_dist_to_gt_trajectory
            )
        ],
    )
    df = processed.df_wide_avg_t

    df = df.select(
        ["clipgt_id", "rollout_id"] + [m for m in PER_CLIP_METRICS if m in df.columns]
    )
    # Strip the `clipgt-` prefix so ids join against lakehouse clip uuids.
    return df.with_columns(
        pl.col("clipgt_id").str.strip_prefix("clipgt-").alias("clip_id")
    )


def check_against_run_summary(
    run_dir: pathlib.Path, df: pl.DataFrame, tol: float = 5e-3
) -> list[str]:
    """Compare per-clip means against the run's own summary parquet.

    The summary is the mean over clips, so any disagreement means the modifier
    chain here has drifted from the one the run used.
    """
    summary_path = run_dir / "aggregate" / "metrics_results.parquet"
    if not summary_path.exists():
        return [f"no run summary at {summary_path}, skipped validation"]

    summary = pl.read_parquet(summary_path).row(0, named=True)
    problems = []
    for metric in df.columns:
        if metric not in summary or df.schema[metric] == pl.String:
            continue
        mine, theirs = df[metric].mean(), summary[metric]
        if mine is None or theirs is None:
            continue
        if abs(mine - theirs) > tol * max(1.0, abs(theirs)):
            problems.append(f"{metric}: per-clip mean {mine:.4f} != run {theirs:.4f}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path)
    args = ap.parse_args()

    df = per_clip_scores(args.run_dir)
    print(df.drop("clipgt_id").to_pandas().to_string(index=False))

    problems = check_against_run_summary(args.run_dir, df)
    print(
        f"\nvalidation vs run summary: {'OK' if not problems else 'MISMATCH'}"
        + "".join(f"\n  {p}" for p in problems)
    )

    if args.out:
        df.write_parquet(args.out)
        print(f"wrote {args.out}  ({df.height} clips)")


if __name__ == "__main__":
    main()
