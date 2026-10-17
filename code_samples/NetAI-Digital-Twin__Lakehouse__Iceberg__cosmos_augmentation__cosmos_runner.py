#!/usr/bin/env python3
"""
CLI Runner for Cosmos augmentation pipeline.

Generates scene-level visual variations (weather, lighting, time-of-day)
from existing Gold-layer clips via Nvidia Cosmos, then stores the results
in a dedicated nvidia_cosmos Iceberg namespace.

Supports two backends:
  - api-catalog (default): Uses build.nvidia.com hosted API with an nvapi- key.
  - nim: Uses a self-hosted Cosmos NIM container.

Usage (inside spark-iceberg container):

    # Using NVIDIA API Catalog (default)
    python -m cosmos_augmentation.cosmos_runner generate \
        --api-key nvapi-XXXX --max-clips 2 --variations foggy,rainy

    # Using self-hosted NIM container
    python -m cosmos_augmentation.cosmos_runner generate \
        --backend nim --endpoint http://cosmos:8000 --max-clips 2

    # Check backend availability
    python -m cosmos_augmentation.cosmos_runner health

    # List eligible clips without generating anything
    python -m cosmos_augmentation.cosmos_runner extract-only --max-clips 5

    # Generate all default variations for all clips
    python -m cosmos_augmentation.cosmos_runner generate --api-key nvapi-XXXX
"""

import argparse
import sys
import time

from nvidia_ingestion.benchmark import BenchmarkTracker

from .config import CosmosPipelineConfig, CosmosConfig, build_spark_session, create_namespace
from .extract import ClipRecord, extract_clips
from .generate import CosmosClient, generate_variations, GeneratedVideo
from .ingest_results import CosmosResultIngester


def _config_from_args(args) -> CosmosPipelineConfig:
    cfg = CosmosPipelineConfig()
    if getattr(args, "max_clips", None):
        cfg.cosmos.max_clips = args.max_clips
    if getattr(args, "variations", None):
        cfg.cosmos.variations = args.variations.split(",")
    if getattr(args, "model", None):
        cfg.cosmos.model = args.model
    if getattr(args, "seed", None) is not None:
        cfg.cosmos.seed = args.seed
    if getattr(args, "endpoint", None):
        cfg.cosmos.endpoint = args.endpoint
    if getattr(args, "backend", None):
        cfg.cosmos.backend = args.backend
    if getattr(args, "api_key", None):
        cfg.cosmos.api_key = args.api_key
    return cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_health(cfg: CosmosPipelineConfig) -> None:
    """Check Cosmos backend availability."""
    client = CosmosClient(cfg.cosmos)
    ok = client.health_check()
    status = "READY" if ok else "UNREACHABLE"
    if cfg.cosmos.backend == "api-catalog":
        print(f"Cosmos API Catalog (key={'set' if cfg.cosmos.api_key else 'MISSING'}): {status}")
        print(f"  Infer URL: {cfg.cosmos.infer_url}")
    else:
        print(f"Cosmos NIM at {cfg.cosmos.endpoint}: {status}")
    if not ok:
        sys.exit(1)


def cmd_extract_only(cfg: CosmosPipelineConfig) -> None:
    """List eligible clips without generating anything."""
    spark = build_spark_session(cfg)
    create_namespace(spark, cfg)
    clips = extract_clips(spark, cfg)

    print(f"\n{'clip_id':<40} {'split':<8} {'country':<16} {'hour':<6}")
    print("-" * 70)
    for c in clips:
        print(f"{c.clip_id:<40} {c.split:<8} {c.country:<16} {c.hour_of_day:<6}")
    print(f"\nTotal: {len(clips)} clips")
    spark.stop()


def cmd_generate(cfg: CosmosPipelineConfig, tracker: BenchmarkTracker, dry_run: bool = False) -> None:
    """Full pipeline: extract clips -> call Cosmos -> upload -> write metadata."""
    print("\n" + "=" * 70)
    print("COSMOS AUGMENTATION PIPELINE")
    print(f"  backend    = {cfg.cosmos.backend}")
    print(f"  model      = {cfg.cosmos.model}")
    print(f"  variations = {cfg.cosmos.variations}")
    print(f"  max_clips  = {cfg.cosmos.max_clips or 'ALL'}")
    print(f"  infer_url  = {cfg.cosmos.infer_url}")
    print("=" * 70)

    total_start = time.time()

    # Phase 1: Extract clips from Gold
    print("\n[PHASE 1/3] Extract clips from Gold")
    print("-" * 50)
    t = time.time()
    spark = build_spark_session(cfg)
    create_namespace(spark, cfg)
    clips = extract_clips(spark, cfg)
    tracker.begin("extract_clips", "sensor_fusion_clip")
    tracker.end(rows_out=len(clips))
    print(f"  Extracted {len(clips)} clips in {time.time()-t:.2f}s")

    if not clips:
        print("  No clips found — nothing to generate.")
        spark.stop()
        return

    if dry_run:
        print("\n  [DRY RUN] Would generate variations for:")
        for c in clips:
            for v in cfg.cosmos.variations:
                print(f"    {c.clip_id[:12]}... -> {v}")
        spark.stop()
        return

    # Phase 2: Generate variations via Cosmos
    print("\n[PHASE 2/3] Generate variations via Cosmos NIM")
    print("-" * 50)
    t = time.time()
    client = CosmosClient(cfg.cosmos)

    if not client.health_check():
        if cfg.cosmos.backend == "api-catalog":
            print("  ERROR: COSMOS_API_KEY not set. Get one at https://build.nvidia.com")
        else:
            print(f"  ERROR: Cosmos NIM at {cfg.cosmos.endpoint} is not ready.")
            print("  Start the Cosmos container or set COSMOS_ENDPOINT.")
        spark.stop()
        sys.exit(1)

    all_videos: list[GeneratedVideo] = []
    for i, clip in enumerate(clips):
        print(f"\n  [{i+1}/{len(clips)}] Clip {clip.clip_id[:12]}...")
        # For text2world, no input video is needed.
        # For transfer/video2world, input_video_b64 would come from
        # camera frame assembly — left as None here (falls back to text2world).
        videos = generate_variations(
            client,
            clip,
            input_video_b64=None,
            variations=cfg.cosmos.variations,
            seed=cfg.cosmos.seed,
        )
        all_videos.extend(videos)

    tracker.begin("generate_variations", "cosmos_api")
    tracker.end(rows_out=len(all_videos))
    print(f"\n  Generated {len(all_videos)} videos in {time.time()-t:.2f}s")

    # Phase 3: Upload + write metadata
    print("\n[PHASE 3/3] Upload to MinIO and write Iceberg metadata")
    print("-" * 50)
    t = time.time()
    ingester = CosmosResultIngester(spark, cfg)
    n_meta = ingester.write_metadata(all_videos)
    n_lineage = ingester.write_lineage(all_videos)
    tracker.begin("ingest_results", "nvidia_cosmos")
    tracker.end(rows_out=n_meta + n_lineage)
    print(f"  Ingested {n_meta} metadata + {n_lineage} lineage rows in {time.time()-t:.2f}s")

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COSMOS PIPELINE COMPLETE in {total:.2f}s")
    print(f"  Videos generated: {len(all_videos)}")
    print(f"  Metadata rows:    {n_meta}")
    print(f"  Lineage rows:     {n_lineage}")
    print(f"{'=' * 70}")

    spark.stop()


def cmd_ingest_only(cfg: CosmosPipelineConfig) -> None:
    """Re-ingest metadata from videos already in MinIO (skip generation)."""
    print("ingest-only: not yet implemented — upload videos to MinIO under")
    print(f"  s3://{cfg.storage.bucket}/{cfg.cosmos.output_s3_prefix}/")
    print("  then run 'generate' to populate metadata tables.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cosmos augmentation pipeline — synthetic scene generation"
    )
    parser.add_argument(
        "command",
        choices=["health", "extract-only", "generate", "ingest-only"],
        help="Pipeline command to run.",
    )
    parser.add_argument("--backend", choices=["nim", "api-catalog"],
                        default=None,
                        help="Backend: 'nim' (self-hosted) or 'api-catalog' (build.nvidia.com, default)")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key",
                        help="NVIDIA API Catalog key (nvapi-...) for api-catalog backend")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Limit number of clips to process (0=all)")
    parser.add_argument("--variations", type=str, default=None,
                        help="Comma-separated variation names (default: config)")
    parser.add_argument("--model", choices=["transfer", "transfer2.5", "text2world", "video2world"],
                        default=None, help="Override Cosmos model")
    parser.add_argument("--seed", type=int, default=None,
                        help="Reproducibility seed")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Override Cosmos NIM endpoint URL (nim backend only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without calling Cosmos")
    args = parser.parse_args()

    cfg = _config_from_args(args)

    if args.command == "health":
        cmd_health(cfg)
    elif args.command == "extract-only":
        cmd_extract_only(cfg)
    elif args.command == "generate":
        tracker = BenchmarkTracker(f"cosmos-{cfg.cosmos.model}")
        cmd_generate(cfg, tracker, dry_run=args.dry_run)
        tracker.print_summary()
        tracker.flush()
    elif args.command == "ingest-only":
        cmd_ingest_only(cfg)


if __name__ == "__main__":
    main()
