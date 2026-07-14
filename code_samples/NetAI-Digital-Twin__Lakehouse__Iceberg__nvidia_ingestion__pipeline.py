"""
Unified medallion pipeline: Bronze → Silver → Gold.

Under the redefined architecture:
  Bronze = register all data as-is via Iceberg add_files()
  Silver = exclude broken/unusable data (quality checks)
  Gold   = curated edge-case subset (AV model scoring)

Usage:
    # Full pipeline with metadata scoring (no GPU needed)
    python -m nvidia_ingestion.pipeline --all

    # Bronze only (re-registration)
    python -m nvidia_ingestion.pipeline --bronze

    # Silver quality checks only
    python -m nvidia_ingestion.pipeline --silver

    # Gold scoring only (requires Silver to exist)
    python -m nvidia_ingestion.pipeline --gold --backend metadata --top-pct 10

    # Full pipeline with BEVFusion (GPU)
    python -m nvidia_ingestion.pipeline --all --backend bevfusion --gpu 0
"""

import argparse
import json
import time
from typing import Any, Dict

from .config import NvidiaPipelineConfig


def run_pipeline(
    stages: Dict[str, bool],
    config: NvidiaPipelineConfig,
    backend_name: str = "metadata",
    gpu_id: int = 0,
    top_pct: float = 10.0,
    score_limit: int = 0,
    bronze_mode: str = "nfs",
    fuse_root: str = "/mnt/nvidia-fuse",
    backend_kwargs: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Run the specified pipeline stages.

    Args:
        stages: dict of {stage_name: enabled}
        config: pipeline configuration
        backend_name: AV model backend for Gold scoring
        gpu_id: GPU device ID
        top_pct: % of hardest clips for Gold
        score_limit: max clips to score (0 = all)
        fuse_root: FUSE mount root (Bronze registration)
        backend_kwargs: extra kwargs for model backend

    Returns:
        Dict with timing and result summaries per stage.
    """
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {},
    }
    t_total = time.time()

    # ── BRONZE ────────────────────────────────────────────────────────
    if stages.get("bronze", False):
        print(f"\n{'='*70}")
        print("STAGE 1: BRONZE — Register all data as-is")
        print(f"{'='*70}\n")
        t0 = time.time()

        from .register_bronze import run_fuse_registration, run_nfs_registration
        from .benchmark import BenchmarkTracker

        tracker = BenchmarkTracker("bronze-registration")
        if bronze_mode == "nfs":
            bronze_results = run_nfs_registration(config=config, tracker=tracker)
        else:
            bronze_results = run_fuse_registration(
                config=config, tracker=tracker, fuse_root=fuse_root)
        tracker.print_summary()

        elapsed = time.time() - t0
        report["stages"]["bronze"] = {
            "elapsed_s": round(elapsed, 2),
            "tables": {k: v for k, v in bronze_results.items()},
            "total_rows": sum(v for v in bronze_results.values() if v > 0),
        }
        print(f"\nBronze complete: {elapsed:.1f}s, "
              f"{sum(1 for v in bronze_results.values() if v > 0)} tables registered")

    # ── SILVER ────────────────────────────────────────────────────────
    if stages.get("silver", False):
        print(f"\n{'='*70}")
        print("STAGE 2: SILVER — Quality filtering")
        print(f"{'='*70}\n")
        t0 = time.time()

        from .quality_checks import run_quality_pipeline

        report_df, view_counts = run_quality_pipeline(
            config=config, build_views=True)

        elapsed = time.time() - t0
        report["stages"]["silver"] = {
            "elapsed_s": round(elapsed, 2),
            "views": {k: v for k, v in view_counts.items()},
            "total_views": len(view_counts),
        }
        print(f"\nSilver complete: {elapsed:.1f}s, "
              f"{len(view_counts)} quality-filtered views")

    # ── GOLD ──────────────────────────────────────────────────────────
    if stages.get("gold", False):
        print(f"\n{'='*70}")
        print(f"STAGE 3: GOLD — Edge-case scoring ({backend_name})")
        print(f"{'='*70}\n")
        t0 = time.time()

        from .edge_case_scorer import run_gold_scoring

        scores_df, gold_results = run_gold_scoring(
            config=config,
            backend_name=backend_name,
            gpu_id=gpu_id,
            top_pct=top_pct,
            limit=score_limit,
            backend_kwargs=backend_kwargs or {},
        )

        elapsed = time.time() - t0
        report["stages"]["gold"] = {
            "elapsed_s": round(elapsed, 2),
            "backend": backend_name,
            "gpu_id": gpu_id,
            "top_pct": top_pct,
            "tables": {k: v for k, v in gold_results.items()},
        }
        print(f"\nGold complete: {elapsed:.1f}s, "
              f"{len(gold_results)} Gold views created")

    # ── SUMMARY ───────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    report["total_elapsed_s"] = round(total_elapsed, 2)

    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    for stage, data in report["stages"].items():
        print(f"  {stage.upper()}: {data['elapsed_s']:.1f}s")
    print(f"  TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save report
    report_path = "/tmp/nvidia_pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Nvidia PhysicalAI medallion pipeline: Bronze → Silver → Gold"
    )
    # Stage selection
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--bronze", action="store_true", help="Run Bronze registration")
    parser.add_argument("--silver", action="store_true", help="Run Silver quality checks")
    parser.add_argument("--gold", action="store_true", help="Run Gold scoring")

    # Gold options
    parser.add_argument("--backend", default="metadata",
                        help="Scoring backend (metadata, bevfusion, dummy)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--top-pct", type=float, default=10.0,
                        help="Top %% hardest clips for Gold")
    parser.add_argument("--score-limit", type=int, default=0,
                        help="Max clips to score (0 = all)")

    # Bronze options
    parser.add_argument("--bronze-mode", default="nfs", choices=["nfs", "fuse"],
                        help="Bronze registration source (default: nfs)")
    parser.add_argument("--fuse-root", default="/mnt/nvidia-fuse",
                        help="FUSE mount root for Bronze registration (fuse mode only)")

    # BEVFusion options
    parser.add_argument("--bevfusion-config", default=None)
    parser.add_argument("--bevfusion-checkpoint", default=None)

    args = parser.parse_args()

    # Determine stages
    if args.all:
        stages = {"bronze": True, "silver": True, "gold": True}
    else:
        stages = {
            "bronze": args.bronze,
            "silver": args.silver,
            "gold": args.gold,
        }
    if not any(stages.values()):
        parser.error("Specify at least one stage: --all, --bronze, --silver, --gold")

    config = NvidiaPipelineConfig()

    backend_kwargs = {}
    if args.backend == "bevfusion":
        backend_kwargs = {
            "config_path": args.bevfusion_config,
            "checkpoint_path": args.bevfusion_checkpoint,
        }

    run_pipeline(
        stages=stages,
        config=config,
        backend_name=args.backend,
        gpu_id=args.gpu,
        top_pct=args.top_pct,
        score_limit=args.score_limit,
        bronze_mode=args.bronze_mode,
        fuse_root=args.fuse_root,
        backend_kwargs=backend_kwargs,
    )


if __name__ == "__main__":
    main()
