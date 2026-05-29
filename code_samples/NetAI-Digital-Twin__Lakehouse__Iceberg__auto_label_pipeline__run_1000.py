"""Production run: ensemble 3D detection on 1000 clips.

Outputs annotations (KITTI + Parquet), BEV visualizations for sampled
spins, and a benchmark report.

Usage:
    python -m auto_label_pipeline.run_1000 [--output-dir DIR] [--max-spins N]
"""

import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .extract import SNAP_DEFAULT, CAMERA_SENSORS, extract_clips, select_clips
from .decode import decode_clip
from .detect_3d_ensemble import DetectorEnsemble
from .export import export_kitti, export_parquet, export_summary
from .visualize import save_visualizations


def run_1000(
    source_path: str = SNAP_DEFAULT,
    output_dir: str = "/tmp/autolabel_1000",
    workdir: str = "/tmp/autolabel_workdir",
    num_clips: int = 1000,
    max_spins: int = 10,
    score_threshold: float = 0.3,
    viz_every_n_clips: int = 10,
    viz_spins_per_clip: int = 3,
    cleanup_workdir: bool = True,
):
    """Run the ensemble pipeline on ~1000 clips.

    Args:
        output_dir: Root for all outputs (annotations, viz, benchmarks).
        workdir: Temp dir for extracted/decoded data (cleaned between clips).
        num_clips: Target number of clips.
        max_spins: Max LiDAR spins per clip.
        score_threshold: Detection confidence threshold.
        viz_every_n_clips: Save BEV visualizations every N clips.
        viz_spins_per_clip: Number of spins to visualize per selected clip.
        cleanup_workdir: Remove extracted data after each clip to save disk.
    """
    # Output structure
    annot_dir = os.path.join(output_dir, "annotations")
    viz_dir = os.path.join(output_dir, "visualizations")
    bench_dir = os.path.join(output_dir, "benchmarks")
    for d in [annot_dir, viz_dir, bench_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Discover clips across chunks ──
    print("=" * 70)
    print(f"AUTO-LABEL PIPELINE — {num_clips} CLIPS")
    print("=" * 70)

    # Collect clips from multiple chunks
    all_clips = []
    chunk = 0
    while len(all_clips) < num_clips:
        batch = select_clips(source_path, chunk=chunk, num_clips=100)
        if not batch:
            break
        all_clips.extend(batch)
        chunk += 1
    all_clips = all_clips[:num_clips]
    print(f"Selected {len(all_clips)} clips from {chunk} chunks\n")

    # ── Load ensemble detector ──
    print("Loading ensemble detector...")
    t0 = time.time()
    detector = DetectorEnsemble(score_threshold=score_threshold)
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s\n")

    # ── Per-clip benchmark CSV ──
    bench_csv_path = os.path.join(bench_dir, "per_clip_timing.csv")
    bench_csv = open(bench_csv_path, "w", newline="")
    bench_writer = csv.writer(bench_csv)
    bench_writer.writerow([
        "clip_index", "clip_id", "chunk", "n_spins", "n_points",
        "n_detections", "extract_s", "decode_s", "detect_s", "export_s",
        "total_s", "ms_per_spin",
    ])

    # ── Main loop ──
    all_clip_detections = {}
    global_class_counts = defaultdict(int)
    global_timings = {"extract": 0, "decode": 0, "detect": 0, "export": 0}
    total_spins = 0
    total_dets = 0
    failed_clips = []
    t_pipeline_start = time.time()

    # Process in batches of 10 (one chunk at a time)
    batch_size = 10
    for batch_start in range(0, len(all_clips), batch_size):
        batch_clips = all_clips[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(all_clips))

        # Extract batch
        t0 = time.time()
        try:
            extractions = extract_clips(batch_clips, source_path, workdir, CAMERA_SENSORS)
        except Exception as e:
            print(f"  WARN: Extraction failed for batch {batch_start}-{batch_end}: {e}")
            for c in batch_clips:
                failed_clips.append(str(c))
            continue
        t_extract_batch = time.time() - t0

        for j, ext in enumerate(extractions):
            clip_idx = batch_start + j
            clip_id = ext.clip_id
            t_clip_start = time.time()

            try:
                # Decode
                t0 = time.time()
                decoded = decode_clip(ext, cameras=CAMERA_SENSORS, max_spins=max_spins)
                t_decode = time.time() - t0

                if not decoded.lidar_spins:
                    print(f"  [{clip_idx+1}/{len(all_clips)}] {clip_id[:8]}: no LiDAR spins, skipping")
                    failed_clips.append(clip_id)
                    continue

                n_spins = len(decoded.lidar_spins)
                n_points = sum(s.points.shape[0] for s in decoded.lidar_spins)

                # Detect
                t0 = time.time()
                clip_dets = detector.detect_clip_spins(decoded.lidar_spins)
                t_detect = time.time() - t0

                n_dets = sum(len(d) for d in clip_dets.values())
                clip_classes = defaultdict(int)
                for dets in clip_dets.values():
                    for d in dets:
                        clip_classes[d.class_name] += 1
                        global_class_counts[d.class_name] += 1

                # Export
                t0 = time.time()
                lidar_ts = [s.timestamp for s in decoded.lidar_spins]
                export_kitti(clip_dets, annot_dir, clip_id)
                export_parquet(clip_dets, lidar_ts, annot_dir, clip_id)
                t_export = time.time() - t0

                # Visualize (sampled clips)
                if clip_idx % viz_every_n_clips == 0:
                    step = max(1, n_spins // viz_spins_per_clip)
                    for si in range(0, n_spins, step):
                        spin = decoded.lidar_spins[si]
                        dets = clip_dets.get(spin.spin_index, [])
                        save_visualizations(
                            clip_id, spin.spin_index, spin.points, dets,
                            output_dir=viz_dir,
                        )

                t_total = time.time() - t_clip_start
                t_extract = t_extract_batch / len(extractions)  # amortized

                # Record
                all_clip_detections[clip_id] = clip_dets
                total_spins += n_spins
                total_dets += n_dets
                global_timings["extract"] += t_extract
                global_timings["decode"] += t_decode
                global_timings["detect"] += t_detect
                global_timings["export"] += t_export

                bench_writer.writerow([
                    clip_idx, clip_id, f"chunk_{batch_start//100:04d}",
                    n_spins, n_points, n_dets,
                    f"{t_extract:.2f}", f"{t_decode:.2f}", f"{t_detect:.2f}",
                    f"{t_export:.2f}", f"{t_total:.2f}",
                    f"{t_detect/n_spins*1000:.0f}",
                ])

                ms_per_spin = t_detect / n_spins * 1000
                elapsed = time.time() - t_pipeline_start
                rate = (clip_idx + 1) / elapsed * 3600
                top_cls = sorted(clip_classes.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"{c}:{n}" for c, n in top_cls)
                print(f"  [{clip_idx+1:>4}/{len(all_clips)}] {clip_id[:8]}: "
                      f"{n_dets:>4} dets, {ms_per_spin:.0f}ms/spin, "
                      f"{top_str}  [{rate:.0f} clips/hr]")

            except Exception as e:
                print(f"  [{clip_idx+1}/{len(all_clips)}] {clip_id[:8]}: ERROR {e}")
                failed_clips.append(clip_id)

        # Cleanup extracted data for this batch to save disk
        if cleanup_workdir:
            for ext in extractions:
                clip_dir = os.path.join(workdir, ext.clip_id)
                if os.path.exists(clip_dir):
                    shutil.rmtree(clip_dir, ignore_errors=True)

    bench_csv.close()
    t_pipeline_total = time.time() - t_pipeline_start

    # ── Summary export ──
    summary_path = export_summary(all_clip_detections, annot_dir)

    # ── Benchmark report ──
    report = {
        "pipeline": "ensemble (CenterPoint TRT FP16 + PointPillars PyTorch GPU)",
        "platform": "NVIDIA GB10 (DGX Spark), aarch64, CUDA 13.0, 128GB unified",
        "total_clips_processed": len(all_clip_detections),
        "total_clips_failed": len(failed_clips),
        "total_spins": total_spins,
        "total_detections": total_dets,
        "total_wall_time_s": round(t_pipeline_total, 1),
        "model_load_time_s": round(t_load, 1),
        "timing_breakdown_s": {k: round(v, 1) for k, v in global_timings.items()},
        "avg_ms_per_spin_detect": round(global_timings["detect"] / max(total_spins, 1) * 1000, 0),
        "avg_s_per_clip_total": round(t_pipeline_total / max(len(all_clip_detections), 1), 2),
        "throughput_clips_per_hour": round(len(all_clip_detections) / max(t_pipeline_total, 1) * 3600, 0),
        "throughput_spins_per_sec": round(total_spins / max(global_timings["detect"], 1), 1),
        "class_distribution": dict(sorted(global_class_counts.items(), key=lambda x: -x[1])),
        "score_threshold": score_threshold,
        "max_spins_per_clip": max_spins,
        "failed_clips": failed_clips[:20],  # first 20 only
    }

    report_path = os.path.join(bench_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print final report ──
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Clips processed:    {len(all_clip_detections)}")
    print(f"  Clips failed:       {len(failed_clips)}")
    print(f"  Total spins:        {total_spins:,}")
    print(f"  Total detections:   {total_dets:,}")
    print(f"  Wall time:          {t_pipeline_total/60:.1f} minutes")
    print(f"\n  Timing breakdown:")
    for stage, dur in global_timings.items():
        pct = dur / max(t_pipeline_total, 1) * 100
        print(f"    {stage:>10}: {dur:>7.1f}s  ({pct:.0f}%)")
    print(f"    {'TOTAL':>10}: {t_pipeline_total:>7.1f}s")
    print(f"\n  Detection speed:    {global_timings['detect']/max(total_spins,1)*1000:.0f}ms/spin "
          f"({total_spins/max(global_timings['detect'],1):.1f} spins/sec)")
    print(f"  Pipeline speed:     {t_pipeline_total/max(len(all_clip_detections),1):.2f}s/clip "
          f"({len(all_clip_detections)/max(t_pipeline_total,1)*3600:.0f} clips/hr)")
    print(f"\n  Class distribution:")
    for cls, cnt in sorted(global_class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:<25} {cnt:>8}")
    print(f"\n  Output directory:   {output_dir}")
    print(f"    annotations/      KITTI labels + Parquet per clip")
    print(f"    visualizations/   BEV plots (every {viz_every_n_clips}th clip)")
    print(f"    benchmarks/       per_clip_timing.csv + benchmark_report.json")
    print(f"    summary:          {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble auto-labeling on 1000 clips"
    )
    parser.add_argument("--source-path", default=SNAP_DEFAULT)
    parser.add_argument("--output-dir", default="/tmp/autolabel_1000")
    parser.add_argument("--workdir", default="/tmp/autolabel_workdir")
    parser.add_argument("--num-clips", type=int, default=1000)
    parser.add_argument("--max-spins", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--viz-every", type=int, default=10,
                        help="Save visualizations every N clips")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep extracted data (uses more disk)")
    args = parser.parse_args()

    run_1000(
        source_path=args.source_path,
        output_dir=args.output_dir,
        workdir=args.workdir,
        num_clips=args.num_clips,
        max_spins=args.max_spins,
        score_threshold=args.score_threshold,
        viz_every_n_clips=args.viz_every,
        cleanup_workdir=not args.no_cleanup,
    )


if __name__ == "__main__":
    main()
