"""Stage 7: CLI orchestrator — runs the full auto-labeling pipeline end-to-end.

Uses OpenPCDet PointPillars for direct 3D detection from LiDAR (no frustum lifting).
Optionally runs YOLO 2D detection for camera-based annotations.
"""

import argparse
import os
import shutil
import time
from typing import Dict, List, Optional

from .extract import SNAP_DEFAULT, CAMERA_SENSORS, extract_clips, select_clips
from .calibration import load_camera_intrinsics, load_sensor_extrinsics
from .decode import decode_clip
from .detect_2d import Detector2D
from .detect_3d import Detector3D, suppress_duplicate_3d
from .export import export_kitti, export_parquet, export_summary
from .visualize import save_visualizations


def run(
    source_path: str = SNAP_DEFAULT,
    output_dir: str = "/tmp/autolabel_output",
    workdir: str = "/tmp/autolabel_workdir",
    num_clips: int = 5,
    chunk: int = 0,
    max_spins: int = 10,
    cameras: Optional[List[str]] = None,
    yolo_model: str = "yolo11x.pt",
    score_threshold_3d: float = 0.3,
    skip_viz: bool = False,
    skip_2d: bool = False,
    cleanup: bool = False,
):
    """Run the full auto-labeling pipeline.

    Args:
        source_path: Path to the dataset snapshot root.
        output_dir: Where to write final annotations.
        workdir: Temporary directory for extracted/decoded data.
        num_clips: Number of clips to process.
        chunk: Which chunk to select clips from.
        max_spins: Max LiDAR spins per clip (for faster testing).
        cameras: Camera sensors to use for 2D detection. None = front wide only.
        yolo_model: YOLO model name for 2D detection.
        score_threshold_3d: Confidence threshold for 3D detections.
        skip_viz: Skip visualization generation.
        skip_2d: Skip 2D camera detection (3D-only mode).
        cleanup: Remove workdir after completion.
    """
    timings = {}

    if cameras is None:
        cameras = CAMERA_SENSORS

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)

    # ── Stage 1: Select and extract clips ──
    print("=" * 60)
    print("STAGE 1: Clip extraction")
    print("=" * 60)
    t0 = time.time()

    clips = select_clips(source_path, chunk=chunk, num_clips=num_clips)
    if not clips:
        print("ERROR: No clips found. Check source_path and chunk number.")
        return

    extractions = extract_clips(clips, source_path, workdir, cameras)
    timings["extraction"] = time.time() - t0
    print(f"  Extracted {len(extractions)} clips in {timings['extraction']:.1f}s\n")

    # ── Load models ──
    print("=" * 60)
    print("STAGE 2: Loading models")
    print("=" * 60)
    t0 = time.time()

    # 3D detector (PointPillars via OpenPCDet)
    detector_3d = Detector3D(score_threshold=score_threshold_3d)

    # 2D detector (YOLO, optional)
    detector_2d = None
    if not skip_2d:
        detector_2d = Detector2D(model_name=yolo_model)

    timings["model_loading"] = time.time() - t0
    print(f"  Models loaded in {timings['model_loading']:.1f}s\n")

    # ── Stage 3+4: Decode + detect — per clip ──
    print("=" * 60)
    print("STAGE 3-4: Decode + 3D detect" + (" + 2D detect" if detector_2d else ""))
    print("=" * 60)
    t0 = time.time()

    all_clip_detections = {}

    for i, ext in enumerate(extractions):
        clip_id = ext.clip_id
        print(f"\n  [{i + 1}/{len(extractions)}] Processing clip: {clip_id}")

        # ── Decode ──
        t_dec = time.time()
        decoded = decode_clip(ext, cameras=cameras, max_spins=max_spins)
        print(f"    Decoded: {len(decoded.lidar_spins)} spins, "
              f"{sum(len(v) for v in decoded.camera_frames.values())} camera frames "
              f"({time.time() - t_dec:.1f}s)")

        if not decoded.lidar_spins:
            print("    WARN: No LiDAR spins decoded, skipping clip")
            continue

        # ── 3D detection (PointPillars on LiDAR) ──
        t_3d = time.time()
        clip_3d_by_spin = detector_3d.detect_clip_spins(decoded.lidar_spins)
        total_3d = sum(len(d) for d in clip_3d_by_spin.values())
        print(f"    3D detections: {total_3d} across {len(clip_3d_by_spin)} spins "
              f"({time.time() - t_3d:.1f}s)")

        # ── 2D detection (YOLO on cameras, optional) ──
        front_image = None
        front_dets = None

        if detector_2d:
            for spin_idx, spin in enumerate(decoded.lidar_spins):
                for sensor in ["camera_front_wide_120fov"]:
                    frames = decoded.camera_frames.get(sensor, [])
                    if spin_idx < len(frames):
                        frame = frames[spin_idx]
                        frame_dets = detector_2d.detect_frame(frame.image, sensor)
                        if spin_idx == 0:
                            front_image = frame.image
                            front_dets = frame_dets

        # ── Visualization (selected spins) ──
        if not skip_viz:
            for spin_idx in range(0, len(decoded.lidar_spins), max(1, len(decoded.lidar_spins) // 3)):
                spin = decoded.lidar_spins[spin_idx]
                dets = clip_3d_by_spin.get(spin.spin_index, [])

                # Get front camera frame for this spin
                cam_img = None
                cam_dets = None
                if detector_2d:
                    frames = decoded.camera_frames.get("camera_front_wide_120fov", [])
                    if spin_idx < len(frames):
                        cam_img = frames[spin_idx].image
                        cam_dets = detector_2d.detect_frame(cam_img, "camera_front_wide_120fov")

                save_visualizations(
                    clip_id, spin.spin_index, spin.points, dets,
                    camera_image=cam_img,
                    detections_2d=cam_dets,
                    output_dir=os.path.join(output_dir, "visualizations"),
                )

        # Collect class stats
        class_counts = {}
        for dets in clip_3d_by_spin.values():
            for d in dets:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        if class_counts:
            print(f"    Classes: {dict(sorted(class_counts.items(), key=lambda x: -x[1]))}")

        all_clip_detections[clip_id] = clip_3d_by_spin

        # ── Export ──
        lidar_ts = [s.timestamp for s in decoded.lidar_spins]
        export_kitti(clip_3d_by_spin, output_dir, clip_id)
        export_parquet(clip_3d_by_spin, lidar_ts, output_dir, clip_id)

    timings["detect_and_export"] = time.time() - t0
    print(f"\n  Total decode+detect+export: {timings['detect_and_export']:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("STAGE 5: Export summary")
    print("=" * 60)
    summary_path = export_summary(all_clip_detections, output_dir)
    print(f"  Summary: {summary_path}")

    # ── Report ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Clips processed:  {len(all_clip_detections)}")
    total_dets = sum(
        sum(len(d) for d in by_spin.values())
        for by_spin in all_clip_detections.values()
    )
    print(f"  Total 3D detections: {total_dets}")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Timing breakdown:")
    for stage, duration in timings.items():
        print(f"    {stage}: {duration:.1f}s")
    total_time = sum(timings.values())
    print(f"    TOTAL: {total_time:.1f}s")

    if num_clips > 0 and total_time > 0:
        per_clip = total_time / len(all_clip_detections) if all_clip_detections else 0
        print(f"\n  Estimated time for 100 clips: {per_clip * 100 / 60:.1f} minutes")
        print(f"  Estimated time for 1000 clips: {per_clip * 1000 / 3600:.1f} hours")

    if cleanup:
        print(f"\n  Cleaning up workdir: {workdir}")
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-labeling pipeline for NVIDIA PhysicalAI AV dataset"
    )
    parser.add_argument("--source-path", default=SNAP_DEFAULT,
                        help="Dataset snapshot root path")
    parser.add_argument("--output-dir", default="/tmp/autolabel_output",
                        help="Output directory for annotations")
    parser.add_argument("--workdir", default="/tmp/autolabel_workdir",
                        help="Temporary working directory")
    parser.add_argument("--num-clips", type=int, default=5,
                        help="Number of clips to process")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Chunk number to select clips from")
    parser.add_argument("--max-spins", type=int, default=10,
                        help="Max LiDAR spins per clip (0=all)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera sensors to process (default: all)")
    parser.add_argument("--yolo-model", default="yolo11x.pt",
                        help="YOLO model name for 2D detection")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="3D detection confidence threshold")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--skip-2d", action="store_true",
                        help="Skip 2D camera detection (3D-only mode)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove workdir after completion")

    args = parser.parse_args()

    max_spins = args.max_spins if args.max_spins > 0 else None

    run(
        source_path=args.source_path,
        output_dir=args.output_dir,
        workdir=args.workdir,
        num_clips=args.num_clips,
        chunk=args.chunk,
        max_spins=max_spins,
        cameras=args.cameras,
        yolo_model=args.yolo_model,
        score_threshold_3d=args.score_threshold,
        skip_viz=args.skip_viz,
        skip_2d=args.skip_2d,
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    main()
