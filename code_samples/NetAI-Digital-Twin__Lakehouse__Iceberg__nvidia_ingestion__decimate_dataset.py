#!/usr/bin/env python3
"""
Decimate the Nvidia PhysicalAI-AV dataset from ~90 TB down to ~10 TB
by selecting a stratified subset of chunks that maximises diversity
across country, season, time-of-day, and sensor platform.

Usage:
    # Dry-run (default): only prints what would be copied and estimated size
    python decimate_dataset.py

    # Execute the transfer
    python decimate_dataset.py --execute

    # Adjust target chunk count (default 340 ≈ 10.6 TB all-modality)
    python decimate_dataset.py --target-chunks 300 --execute
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_ROOT = Path(
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)
DST_ROOT = Path("/mnt/netai-e2e/nvidia-physicalai-av-subset")

# All sensor subdirectories that are chunked as zip files
SENSOR_DIRS = {
    "camera": [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ],
    "lidar": [
        "lidar_top_360fov",
    ],
    "radar": [
        "radar_corner_front_left_srr_0",
        "radar_corner_front_left_srr_3",
        "radar_corner_front_right_srr_0",
        "radar_corner_front_right_srr_3",
        "radar_corner_rear_left_srr_0",
        "radar_corner_rear_left_srr_3",
        "radar_corner_rear_right_srr_0",
        "radar_corner_rear_right_srr_3",
        "radar_front_center_imaging_lrr_1",
        "radar_front_center_mrr_2",
        "radar_front_center_srr_0",
        "radar_rear_left_mrr_2",
        "radar_rear_left_srr_0",
        "radar_rear_right_mrr_2",
        "radar_rear_right_srr_0",
        "radar_side_left_srr_0",
        "radar_side_left_srr_3",
        "radar_side_right_srr_0",
        "radar_side_right_srr_3",
    ],
    "labels": [
        "egomotion",
    ],
    "calibration": [
        "camera_intrinsics",
        "sensor_extrinsics",
        "vehicle_dimensions",
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("decimate")


# ---------------------------------------------------------------------------
# Stratified chunk selection
# ---------------------------------------------------------------------------
def select_chunks(target: int, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of selected chunks with metadata."""
    ci = pd.read_parquet(SRC_ROOT / "clip_index.parquet")
    dc = pd.read_parquet(SRC_ROOT / "metadata" / "data_collection.parquet")
    merged = ci.join(dc)

    merged["hour_bin"] = pd.cut(
        merged["hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    )
    merged["season"] = merged["month"].map(
        {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        }
    )

    chunk_meta = (
        merged.groupby("chunk")
        .agg(
            country=("country", lambda x: x.mode()[0]),
            season=("season", lambda x: x.mode()[0]),
            hour_bin=("hour_bin", lambda x: x.mode()[0]),
            platform=("platform_class", lambda x: x.mode()[0]),
            n_clips=("country", "size"),
            split=("split", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    rng = np.random.RandomState(seed)
    strata_keys = ["country", "season", "hour_bin", "platform"]
    groups = chunk_meta.groupby(strata_keys)
    strata_sizes = groups.size()
    n_strata = len(strata_sizes)

    if target < n_strata:
        log.warning(
            "Target %d < number of strata %d; selecting 1 per stratum.",
            target, n_strata,
        )
        target = n_strata

    remaining = target - n_strata
    proportions = strata_sizes / strata_sizes.sum()
    extra = (proportions * remaining).apply(np.floor).astype(int)
    shortfall = remaining - extra.sum()
    top = proportions.sort_values(ascending=False).index[: int(shortfall)]
    for s in top:
        extra[s] += 1
    alloc = extra + 1

    selected = []
    for keys, group in groups:
        n = alloc[keys]
        if n >= len(group):
            selected.append(group)
        else:
            selected.append(group.sample(n=n, random_state=rng))

    result = pd.concat(selected).sort_values("chunk")
    log.info(
        "Selected %d chunks (%d clips) across %d/%d countries, "
        "%d seasons, %d hour bins, %d platforms",
        len(result),
        result["n_clips"].sum(),
        result["country"].nunique(),
        chunk_meta["country"].nunique(),
        result["season"].nunique(),
        result["hour_bin"].nunique(),
        result["platform"].nunique(),
    )
    return result


# ---------------------------------------------------------------------------
# Build file manifest
# ---------------------------------------------------------------------------
def build_manifest(selected_chunks: pd.DataFrame) -> list[dict]:
    """Build a list of {src, dst, size} dicts for every file to transfer."""
    chunk_set = set(selected_chunks["chunk"].tolist())
    manifest = []

    for modality, sensors in SENSOR_DIRS.items():
        for sensor in sensors:
            src_dir = SRC_ROOT / modality / sensor
            if not src_dir.is_dir():
                log.warning("Sensor dir missing: %s", src_dir)
                continue

            for fname in sorted(os.listdir(src_dir)):
                # Parse chunk number from filename like sensor.chunk_0042.zip
                parts = fname.split(".")
                chunk_part = [p for p in parts if p.startswith("chunk_")]
                if not chunk_part:
                    continue
                chunk_num = int(chunk_part[0].split("_")[1])
                if chunk_num not in chunk_set:
                    continue

                src_path = src_dir / fname
                real_path = Path(os.path.realpath(src_path))
                size = real_path.stat().st_size

                dst_path = DST_ROOT / modality / sensor / fname
                manifest.append({
                    "src": str(real_path),
                    "dst": str(dst_path),
                    "size": size,
                    "modality": modality,
                    "sensor": sensor,
                    "chunk": chunk_num,
                })

    return manifest


# ---------------------------------------------------------------------------
# Copy logic
# ---------------------------------------------------------------------------
def copy_files(manifest: list[dict], dry_run: bool = True) -> None:
    """Copy files from manifest using rsync for fault tolerance."""
    total_bytes = sum(f["size"] for f in manifest)
    log.info(
        "Manifest: %d files, %.2f TB total",
        len(manifest), total_bytes / 1e12,
    )

    if dry_run:
        # Print summary by modality
        by_mod = {}
        for f in manifest:
            m = f["modality"]
            by_mod.setdefault(m, {"count": 0, "bytes": 0})
            by_mod[m]["count"] += 1
            by_mod[m]["bytes"] += f["size"]

        log.info("--- DRY RUN SUMMARY ---")
        for mod in sorted(by_mod):
            info = by_mod[mod]
            log.info(
                "  %-15s  %5d files  %8.2f TB",
                mod, info["count"], info["bytes"] / 1e12,
            )
        log.info("  %-15s  %5d files  %8.2f TB", "TOTAL", len(manifest), total_bytes / 1e12)
        log.info("Destination: %s", DST_ROOT)
        log.info("Pass --execute to start the transfer.")
        return

    # Save manifest for resumability
    manifest_path = DST_ROOT / "transfer_manifest.json"
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp)
    log.info("Manifest saved to %s", manifest_path)

    # Also save the selected chunks metadata
    copied = 0
    copied_bytes = 0
    errors = []
    t0 = time.time()

    for i, entry in enumerate(manifest):
        dst = Path(entry["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and dst.stat().st_size == entry["size"]:
            copied += 1
            copied_bytes += entry["size"]
            if (i + 1) % 500 == 0:
                log.info(
                    "[%d/%d] Skipped (already exists) %s",
                    i + 1, len(manifest), dst.name,
                )
            continue

        try:
            subprocess.run(
                [
                    "rsync", "-a", "--progress",
                    entry["src"], str(dst),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            copied += 1
            copied_bytes += entry["size"]
        except subprocess.CalledProcessError as exc:
            log.error("FAILED: %s -> %s: %s", entry["src"], dst, exc.stderr)
            errors.append(entry)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            pct = copied_bytes / total_bytes * 100
            rate = copied_bytes / elapsed / 1e9 if elapsed > 0 else 0
            eta_h = (total_bytes - copied_bytes) / (copied_bytes / elapsed) / 3600 if copied_bytes > 0 else 0
            log.info(
                "[%d/%d] %.1f%% done | %.1f GB/s | ETA %.1f hours",
                i + 1, len(manifest), pct, rate, eta_h,
            )

    elapsed = time.time() - t0
    log.info(
        "Transfer complete: %d/%d files (%.2f TB) in %.1f hours",
        copied, len(manifest), copied_bytes / 1e12, elapsed / 3600,
    )
    if errors:
        err_path = DST_ROOT / "transfer_errors.json"
        with open(err_path, "w") as fp:
            json.dump(errors, fp)
        log.error("%d errors saved to %s", len(errors), err_path)


# ---------------------------------------------------------------------------
# Copy metadata (small files, always copied in full)
# ---------------------------------------------------------------------------
def copy_metadata(dry_run: bool = True) -> None:
    """Copy non-chunked metadata files: clip_index, metadata/, README, LICENSE."""
    small_files = [
        "clip_index.parquet",
        "README.md",
        "LICENSE.pdf",
        "metadata/data_collection.parquet",
        "metadata/sensor_presence.parquet",
    ]
    if dry_run:
        log.info("Would also copy %d metadata files.", len(small_files))
        return

    for rel in small_files:
        src = SRC_ROOT / rel
        dst = DST_ROOT / rel
        if not src.exists():
            log.warning("Metadata file missing: %s", src)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        real = Path(os.path.realpath(src))
        subprocess.run(
            ["rsync", "-a", str(real), str(dst)],
            check=True,
        )
        log.info("Copied metadata: %s", rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Decimate Nvidia PhysicalAI-AV dataset to ~10 TB stratified subset"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually copy files (default is dry-run)",
    )
    parser.add_argument(
        "--target-chunks", type=int, default=340,
        help="Number of chunks to select (default: 340 ≈ 10.6 TB)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection",
    )
    parser.add_argument(
        "--save-selection", type=str, default=None,
        help="Save selected chunk list to CSV",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    log.info("Source: %s", SRC_ROOT)
    log.info("Destination: %s", DST_ROOT)
    log.info("Target chunks: %d | Seed: %d | Dry-run: %s",
             args.target_chunks, args.seed, dry_run)

    # Step 1: Select chunks
    selected = select_chunks(args.target_chunks, seed=args.seed)

    if args.save_selection:
        selected.to_csv(args.save_selection, index=False)
        log.info("Saved selection to %s", args.save_selection)

    # Step 2: Build file manifest
    log.info("Building file manifest (resolving symlinks, reading sizes)...")
    manifest = build_manifest(selected)

    # Step 3: Copy metadata
    copy_metadata(dry_run=dry_run)

    # Step 4: Copy sensor data
    copy_files(manifest, dry_run=dry_run)

    # Step 5: Save selection metadata alongside the data
    if not dry_run:
        sel_path = DST_ROOT / "selected_chunks.csv"
        selected.to_csv(sel_path, index=False)
        log.info("Saved chunk selection metadata to %s", sel_path)


if __name__ == "__main__":
    main()
