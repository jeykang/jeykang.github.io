#!/usr/bin/env python3
"""
CLI Runner for Nvidia PhysicalAI Dataset Ingestion Pipeline.

Runs the full Bronze → Silver → Gold medallion pipeline against the
NFS-mounted Nvidia dataset (zip archives read in-place, no extraction).

Usage (inside spark-iceberg container or with PYTHONPATH set):

    # Full pipeline — Draco blobs (default)
    python -m nvidia_ingestion.nvidia_runner all

    # Full pipeline — decoded point arrays
    NVIDIA_LIDAR_MODE=decoded python -m nvidia_ingestion.nvidia_runner all

    # Individual layers
    python -m nvidia_ingestion.nvidia_runner bronze
    python -m nvidia_ingestion.nvidia_runner silver
    python -m nvidia_ingestion.nvidia_runner gold

    # Quick smoke-test (2 zip chunks, 5 clips each)
    python -m nvidia_ingestion.nvidia_runner all --max-chunks 2 --max-clips 5

    # Dual-mode benchmark (blob then decoded, results saved)
    python -m nvidia_ingestion.nvidia_runner benchmark

    # --- Zero-copy pipelines (new) ---

    # Option 1: FUSE-based zero-copy (existing NFS dataset, benchmarking)
    # Step 1: Mount zips on HOST first:
    #   python -m nvidia_ingestion.fuse_mount mount --sensor radar --max-chunks 2
    # Step 2: Register in Iceberg (inside Spark container):
    python -m nvidia_ingestion.nvidia_runner register-fuse

    # Option 2: Byte-stream to MinIO (incoming real-world data)
    # Step 1: Upload Parquet bytes from zips to MinIO:
    python -m nvidia_ingestion.nvidia_runner upload-s3 --max-chunks 2
    # Step 2: Register uploaded files in Iceberg:
    python -m nvidia_ingestion.nvidia_runner register-s3

    # Full zero-copy medallion pipeline (FUSE: register + silver + gold)
    python -m nvidia_ingestion.nvidia_runner zerocopy-fuse

    # Full streaming medallion pipeline (upload + register + silver + gold)
    python -m nvidia_ingestion.nvidia_runner zerocopy-s3
"""

import argparse
import sys
import time
from typing import Dict

from .benchmark import BenchmarkTracker
from .config import NvidiaPipelineConfig, NvidiaConfig


def _config_from_args(args) -> NvidiaPipelineConfig:
    cfg = NvidiaPipelineConfig()
    if args.max_chunks:
        cfg.nvidia.max_zip_chunks = args.max_chunks
    if args.max_clips:
        cfg.nvidia.max_clips_per_chunk = args.max_clips
    if args.lidar_mode:
        cfg.nvidia.lidar_mode = args.lidar_mode
    if getattr(args, "silver_mode", None):
        cfg.nvidia.silver_mode = args.silver_mode
    if getattr(args, "gold_mode", None):
        cfg.nvidia.gold_mode = args.gold_mode
    return cfg


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_bronze(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker) -> Dict[str, int]:
    from .ingest_bronze import run_bronze_ingestion
    return run_bronze_ingestion(config=cfg, tracker=tracker)


def run_silver(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker) -> Dict[str, int]:
    from .transform_silver import run_silver_transformation
    return run_silver_transformation(config=cfg, tracker=tracker)


def run_gold(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker) -> Dict[str, int]:
    from .build_gold import run_gold_build
    return run_gold_build(config=cfg, tracker=tracker)


def run_register_fuse(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker,
                      fuse_root: str = "/mnt/nvidia-fuse") -> Dict[str, int]:
    from .register_bronze import run_fuse_registration
    return run_fuse_registration(config=cfg, tracker=tracker, fuse_root=fuse_root)


def run_register_s3(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker,
                    s3_prefix: str = "nvidia_bronze") -> Dict[str, int]:
    from .register_bronze import run_s3_registration
    return run_s3_registration(config=cfg, tracker=tracker, s3_prefix=s3_prefix)


def run_upload_s3(cfg: NvidiaPipelineConfig, args) -> Dict:
    from .stream_to_s3 import upload_all
    return upload_all(
        source=cfg.nvidia.source_path,
        max_chunks=cfg.nvidia.max_zip_chunks,
        max_clips=cfg.nvidia.max_clips_per_chunk,
        dry_run=getattr(args, "dry_run", False),
    )


def _print_results(phase: str, results: Dict[str, int]):
    print(f"\n{'=' * 60}")
    print(f"  {phase} SUMMARY")
    print(f"{'=' * 60}")
    for table, count in results.items():
        status = "✓" if count >= 0 else "✗"
        print(f"  {status} {table}: {count} rows")


def run_all(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker) -> bool:
    print("\n" + "=" * 70)
    print("NVIDIA PHYSICALAI DATASET — MEDALLION INGESTION PIPELINE")
    print(f"  lidar_mode = {cfg.nvidia.lidar_mode}")
    print(f"  max_chunks = {cfg.nvidia.max_zip_chunks or 'ALL'}")
    print(f"  max_clips  = {cfg.nvidia.max_clips_per_chunk or 'ALL'}")
    print("=" * 70)

    total_start = time.time()

    # Bronze
    print("\n[PHASE 1/3] Bronze Layer Ingestion")
    print("-" * 50)
    t = time.time()
    bronze = run_bronze(cfg, tracker)
    print(f"Bronze completed in {time.time()-t:.2f}s")
    _print_results("BRONZE", bronze)

    if any(v < 0 for v in bronze.values()):
        print("[WARN] Some bronze tables failed — continuing anyway")

    # Silver
    print("\n[PHASE 2/3] Silver Layer Transformation")
    print("-" * 50)
    t = time.time()
    silver = run_silver(cfg, tracker)
    print(f"Silver completed in {time.time()-t:.2f}s")
    _print_results("SILVER", silver)

    # Gold
    print("\n[PHASE 3/3] Gold Layer Construction")
    print("-" * 50)
    t = time.time()
    gold = run_gold(cfg, tracker)
    print(f"Gold completed in {time.time()-t:.2f}s")
    _print_results("GOLD", gold)

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {total:.2f}s  (lidar_mode={cfg.nvidia.lidar_mode})")
    print(f"{'=' * 70}")
    return True


def run_zerocopy_fuse(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker,
                      fuse_root: str = "/mnt/nvidia-fuse") -> bool:
    """Full zero-copy medallion pipeline: FUSE register → Silver → Gold."""
    print("\n" + "=" * 70)
    print("NVIDIA PHYSICALAI — ZERO-COPY MEDALLION PIPELINE (FUSE)")
    print(f"  fuse_root  = {fuse_root}")
    print(f"  max_chunks = {cfg.nvidia.max_zip_chunks or 'ALL'}")
    print("=" * 70)

    total_start = time.time()

    # Bronze (zero-copy registration)
    print("\n[PHASE 1/3] Bronze Registration (FUSE — zero-copy)")
    print("-" * 50)
    t = time.time()
    bronze = run_register_fuse(cfg, tracker, fuse_root=fuse_root)
    print(f"Bronze registration completed in {time.time()-t:.2f}s")
    _print_results("BRONZE (FUSE)", bronze)

    # Silver
    print("\n[PHASE 2/3] Silver Layer Transformation")
    print("-" * 50)
    t = time.time()
    silver = run_silver(cfg, tracker)
    print(f"Silver completed in {time.time()-t:.2f}s")
    _print_results("SILVER", silver)

    # Gold
    print("\n[PHASE 3/3] Gold Layer Construction")
    print("-" * 50)
    t = time.time()
    gold = run_gold(cfg, tracker)
    print(f"Gold completed in {time.time()-t:.2f}s")
    _print_results("GOLD", gold)

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ZERO-COPY PIPELINE COMPLETE in {total:.2f}s")
    print(f"{'=' * 70}")
    return True


def run_zerocopy_s3(cfg: NvidiaPipelineConfig, tracker: BenchmarkTracker,
                    args=None) -> bool:
    """Full streaming medallion pipeline: upload → register → Silver → Gold."""
    print("\n" + "=" * 70)
    print("NVIDIA PHYSICALAI — STREAMING MEDALLION PIPELINE (S3)")
    print(f"  max_chunks = {cfg.nvidia.max_zip_chunks or 'ALL'}")
    print("=" * 70)

    total_start = time.time()

    # Upload to S3
    print("\n[PHASE 1/4] Byte-stream Upload to MinIO")
    print("-" * 50)
    t = time.time()
    run_upload_s3(cfg, args)
    print(f"Upload completed in {time.time()-t:.2f}s")

    # Bronze (register from S3)
    print("\n[PHASE 2/4] Bronze Registration (S3)")
    print("-" * 50)
    t = time.time()
    bronze = run_register_s3(cfg, tracker)
    print(f"Bronze registration completed in {time.time()-t:.2f}s")
    _print_results("BRONZE (S3)", bronze)

    # Silver
    print("\n[PHASE 3/4] Silver Layer Transformation")
    print("-" * 50)
    t = time.time()
    silver = run_silver(cfg, tracker)
    print(f"Silver completed in {time.time()-t:.2f}s")
    _print_results("SILVER", silver)

    # Gold
    print("\n[PHASE 4/4] Gold Layer Construction")
    print("-" * 50)
    t = time.time()
    gold = run_gold(cfg, tracker)
    print(f"Gold completed in {time.time()-t:.2f}s")
    _print_results("GOLD", gold)

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"STREAMING PIPELINE COMPLETE in {total:.2f}s")
    print(f"{'=' * 70}")
    return True


def run_benchmark(args):
    """
    Run the full pipeline twice — once with Draco blobs, once decoded —
    and save comparative benchmarks.
    """
    for mode in ["blob", "decoded"]:
        print(f"\n{'#' * 70}")
        print(f"#  BENCHMARK RUN: lidar_mode = {mode}")
        print(f"{'#' * 70}")
        cfg = _config_from_args(args)
        cfg.nvidia.lidar_mode = mode
        # Use distinct namespaces so runs don't collide
        cfg.nvidia.namespace_bronze = f"nvidia_bronze_{mode}"
        cfg.nvidia.namespace_silver = f"nvidia_silver_{mode}"
        cfg.nvidia.namespace_gold = f"nvidia_gold_{mode}"

        tracker = BenchmarkTracker(f"nvidia-{mode}")
        run_all(cfg, tracker)
        tracker.print_summary()
        tracker.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nvidia PhysicalAI dataset ingestion pipeline"
    )
    parser.add_argument(
        "phase",
        choices=[
            "bronze", "silver", "gold", "all", "benchmark",
            "register-fuse", "register-s3", "upload-s3",
            "zerocopy-fuse", "zerocopy-s3",
        ],
        help=(
            "Pipeline phase to run.  Original: bronze|silver|gold|all|benchmark.  "
            "Zero-copy: register-fuse|register-s3|upload-s3|zerocopy-fuse|zerocopy-s3."
        ),
    )
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Limit number of zip chunks per sensor (0=all)")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Limit clips per chunk (0=all)")
    parser.add_argument("--lidar-mode", choices=["blob", "decoded"],
                        default=None,
                        help="Override lidar storage mode")
    parser.add_argument("--silver-mode", choices=["inplace", "view"],
                        default=None,
                        help="Silver tier mode: inplace (enrich parquets on disk) or view (SQL views)")
    parser.add_argument("--gold-mode", choices=["materialized", "view"],
                        default=None,
                        help="Gold tier mode: materialized (tables) or view (zero storage)")
    parser.add_argument("--fuse-root", default="/mnt/nvidia-fuse",
                        help="FUSE mount root (for register-fuse / zerocopy-fuse)")
    parser.add_argument("--s3-prefix", default="nvidia_bronze",
                        help="S3 key prefix (for upload-s3 / register-s3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded (upload-s3 only)")
    args = parser.parse_args()

    if args.phase == "benchmark":
        run_benchmark(args)
        return

    cfg = _config_from_args(args)
    tracker = BenchmarkTracker(f"nvidia-{args.phase}-{cfg.nvidia.lidar_mode}")

    dispatch = {
        "bronze": lambda: run_bronze(cfg, tracker),
        "silver": lambda: run_silver(cfg, tracker),
        "gold": lambda: run_gold(cfg, tracker),
        "all": lambda: run_all(cfg, tracker),
        "register-fuse": lambda: run_register_fuse(cfg, tracker, fuse_root=args.fuse_root),
        "register-s3": lambda: run_register_s3(cfg, tracker, s3_prefix=args.s3_prefix),
        "upload-s3": lambda: run_upload_s3(cfg, args),
        "zerocopy-fuse": lambda: run_zerocopy_fuse(cfg, tracker, fuse_root=args.fuse_root),
        "zerocopy-s3": lambda: run_zerocopy_s3(cfg, tracker, args=args),
    }
    result = dispatch[args.phase]()

    if args.phase not in ("upload-s3",):
        tracker.print_summary()
        tracker.flush()

    if isinstance(result, dict):
        _print_results(args.phase.upper(), result)


if __name__ == "__main__":
    main()
