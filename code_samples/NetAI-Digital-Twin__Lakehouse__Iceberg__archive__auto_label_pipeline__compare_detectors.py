"""Compare PointPillars vs CenterPoint 3D detection on the same clips.

Usage:
    python -m auto_label_pipeline.compare_detectors [--num-clips N] [--max-spins M]
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

from .extract import SNAP_DEFAULT, CAMERA_SENSORS, extract_clips, select_clips
from .decode import decode_clip
from .detect_3d import Detector3D, Detection3D
from .detect_3d_centerpoint import DetectorCenterPoint
from .export import export_kitti, export_parquet
from .visualize import save_visualizations


def compare(
    source_path: str = SNAP_DEFAULT,
    output_dir: str = "/tmp/autolabel_output",
    workdir: str = "/tmp/autolabel_workdir",
    num_clips: int = 5,
    chunk: int = 0,
    max_spins: int = 10,
    score_threshold: float = 0.3,
):
    """Run both detectors on the same clips and compare results."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    # ── Extract clips ──
    print("=" * 70)
    print("EXTRACTING CLIPS")
    print("=" * 70)
    t0 = time.time()
    clips = select_clips(source_path, chunk=chunk, num_clips=num_clips)
    if not clips:
        print("ERROR: No clips found.")
        return
    extractions = extract_clips(clips, source_path, workdir, CAMERA_SENSORS)
    print(f"  Extracted {len(extractions)} clips in {time.time() - t0:.1f}s\n")

    # ── Load models ──
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    t0 = time.time()
    det_pp = Detector3D(score_threshold=score_threshold)
    t_pp_load = time.time() - t0
    print(f"  PointPillars loaded in {t_pp_load:.1f}s")

    t0 = time.time()
    det_cp = DetectorCenterPoint(score_threshold=score_threshold)
    t_cp_load = time.time() - t0
    print(f"  CenterPoint loaded in {t_cp_load:.1f}s\n")

    # ── Run both detectors per clip ──
    print("=" * 70)
    print("RUNNING COMPARISON")
    print("=" * 70)

    results = {}  # clip_id -> {pp: {...}, cp: {...}}

    for i, ext in enumerate(extractions):
        clip_id = ext.clip_id
        print(f"\n  [{i + 1}/{len(extractions)}] Clip: {clip_id}")

        # Decode
        decoded = decode_clip(ext, cameras=CAMERA_SENSORS, max_spins=max_spins)
        if not decoded.lidar_spins:
            print("    WARN: No LiDAR spins, skipping")
            continue

        n_spins = len(decoded.lidar_spins)
        n_points = sum(s.points.shape[0] for s in decoded.lidar_spins)
        print(f"    Decoded: {n_spins} spins, {n_points:,} total points")

        # ── PointPillars ──
        t0 = time.time()
        pp_by_spin = det_pp.detect_clip_spins(decoded.lidar_spins)
        t_pp = time.time() - t0
        pp_total = sum(len(d) for d in pp_by_spin.values())
        pp_classes = defaultdict(int)
        for dets in pp_by_spin.values():
            for d in dets:
                pp_classes[d.class_name] += 1

        print(f"    PointPillars: {pp_total} detections in {t_pp:.1f}s")
        if pp_classes:
            print(f"      Classes: {dict(sorted(pp_classes.items(), key=lambda x: -x[1]))}")

        # ── CenterPoint ──
        t0 = time.time()
        cp_by_spin = det_cp.detect_clip_spins(decoded.lidar_spins)
        t_cp = time.time() - t0
        cp_total = sum(len(d) for d in cp_by_spin.values())
        cp_classes = defaultdict(int)
        for dets in cp_by_spin.values():
            for d in dets:
                cp_classes[d.class_name] += 1

        print(f"    CenterPoint:  {cp_total} detections in {t_cp:.1f}s")
        if cp_classes:
            print(f"      Classes: {dict(sorted(cp_classes.items(), key=lambda x: -x[1]))}")

        results[clip_id] = {
            "n_spins": n_spins,
            "n_points": n_points,
            "pp": {"total": pp_total, "classes": dict(pp_classes), "time": t_pp, "by_spin": pp_by_spin},
            "cp": {"total": cp_total, "classes": dict(cp_classes), "time": t_cp, "by_spin": cp_by_spin},
        }

        # ── Export both sets ──
        lidar_ts = [s.timestamp for s in decoded.lidar_spins]

        pp_dir = os.path.join(output_dir, "comparison", "pointpillars")
        cp_dir = os.path.join(output_dir, "comparison", "centerpoint")
        os.makedirs(pp_dir, exist_ok=True)
        os.makedirs(cp_dir, exist_ok=True)

        export_kitti(pp_by_spin, pp_dir, clip_id)
        export_parquet(pp_by_spin, lidar_ts, pp_dir, clip_id)
        export_kitti(cp_by_spin, cp_dir, clip_id)
        export_parquet(cp_by_spin, lidar_ts, cp_dir, clip_id)

        # ── Side-by-side BEV visualization (selected spins) ──
        for spin_idx in range(0, n_spins, max(1, n_spins // 3)):
            spin = decoded.lidar_spins[spin_idx]
            pp_dets = pp_by_spin.get(spin.spin_index, [])
            cp_dets = cp_by_spin.get(spin.spin_index, [])

            save_visualizations(
                clip_id, spin.spin_index, spin.points, pp_dets,
                output_dir=os.path.join(output_dir, "comparison", "viz_pointpillars"),
            )
            save_visualizations(
                clip_id, spin.spin_index, spin.points, cp_dets,
                output_dir=os.path.join(output_dir, "comparison", "viz_centerpoint"),
            )

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n  {'Clip':<12} {'Spins':>6} {'PP Dets':>8} {'CP Dets':>8} {'PP Time':>8} {'CP Time':>8}")
    print("  " + "-" * 56)

    total_pp = total_cp = 0
    total_pp_t = total_cp_t = 0.0

    for clip_id, r in results.items():
        short_id = clip_id[:8]
        pp = r["pp"]
        cp = r["cp"]
        total_pp += pp["total"]
        total_cp += cp["total"]
        total_pp_t += pp["time"]
        total_cp_t += cp["time"]
        print(f"  {short_id:<12} {r['n_spins']:>6} {pp['total']:>8} {cp['total']:>8} "
              f"{pp['time']:>7.1f}s {cp['time']:>7.1f}s")

    print("  " + "-" * 56)
    print(f"  {'TOTAL':<12} {'':>6} {total_pp:>8} {total_cp:>8} "
          f"{total_pp_t:>7.1f}s {total_cp_t:>7.1f}s")

    # Class breakdown
    print(f"\n  Class breakdown:")
    all_classes = set()
    pp_class_totals = defaultdict(int)
    cp_class_totals = defaultdict(int)
    for r in results.values():
        for cls, cnt in r["pp"]["classes"].items():
            pp_class_totals[cls] += cnt
            all_classes.add(cls)
        for cls, cnt in r["cp"]["classes"].items():
            cp_class_totals[cls] += cnt
            all_classes.add(cls)

    print(f"  {'Class':<25} {'PointPillars':>12} {'CenterPoint':>12}")
    print("  " + "-" * 50)
    for cls in sorted(all_classes):
        print(f"  {cls:<25} {pp_class_totals.get(cls, 0):>12} {cp_class_totals.get(cls, 0):>12}")

    print(f"\n  Output: {output_dir}/comparison/")
    print(f"  Visualizations: viz_pointpillars/ and viz_centerpoint/")


def main():
    parser = argparse.ArgumentParser(description="Compare PointPillars vs CenterPoint")
    parser.add_argument("--source-path", default=SNAP_DEFAULT)
    parser.add_argument("--output-dir", default="/tmp/autolabel_output")
    parser.add_argument("--workdir", default="/tmp/autolabel_workdir")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--max-spins", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()

    max_spins = args.max_spins if args.max_spins > 0 else None

    compare(
        source_path=args.source_path,
        output_dir=args.output_dir,
        workdir=args.workdir,
        num_clips=args.num_clips,
        chunk=args.chunk,
        max_spins=max_spins,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
