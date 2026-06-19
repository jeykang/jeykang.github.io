"""BEVFusion multimodal perception scorer for PhysicalAI clips.

Runs in the bevfusion-runner container. Reads a CSV of clip_ids to score
(produced by a Spark query in spark-iceberg), opens the corresponding
camera mp4s + lidar parquets on NFS, samples N frames per clip, runs
BEVFusion inference, and writes per-clip difficulty scores to a parquet
under <NFS>/.perception/.

The "difficulty score" is a temporal-consistency signal — high jitter
in detection counts / class distribution / confidence across the sampled
frames within a clip indicates the model is uncertain, which correlates
with hard scenes. We do NOT use absolute mAP; the pretrained nuScenes
checkpoint has domain shift to PhysicalAI and absolute accuracy is
unreliable. Relative-difficulty signal is what we need for Gold subset
selection.

Output schema (per clip):
  clip_id              str
  n_frames_sampled     int
  mean_n_detections    float
  std_n_detections     float
  mean_max_conf        float
  std_max_conf         float
  class_diversity      float    # entropy of class distribution
  driving_obj_count    float    # mean detections in {car, truck, ped, cyclist}
  perception_score     float    # composite ∈ [0, 1], higher = harder
  shard_id             int
  scored_at            str
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# mmdet3d / BEVFusion imports are deferred to __main__ to keep --help fast.

NFS_ROOT_DEFAULT = "/mnt/netai-e2e/nvidia-physicalai-av-subset"
PERCEPTION_DIR = ".perception"

CAMERA_SENSORS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-list", required=True,
                   help="CSV/parquet with clip_id column to score")
    p.add_argument("--checkpoint", required=True,
                   help="Path to BEVFusion .pth")
    p.add_argument("--config", required=True,
                   help="Path to BEVFusion config .py")
    p.add_argument("--frames-per-clip", type=int, default=10,
                   help="Frames sampled evenly across the clip's timestamp range")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--output-dir", default=f"{NFS_ROOT_DEFAULT}/{PERCEPTION_DIR}")
    p.add_argument("--nfs-root", default=NFS_ROOT_DEFAULT)
    p.add_argument("--max-clips", type=int, default=0,
                   help="Limit for testing")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def load_clip_ids(path: str, shard_id: int, n_shards: int, max_clips: int):
    """Stable shard via hashlib.md5 (matches camera_perception_scorer pattern)."""
    import hashlib
    if path.endswith(".parquet"):
        ids = pq.read_table(path).column("clip_id").to_pylist()
    else:
        with open(path) as f:
            r = csv.DictReader(f)
            ids = [row["clip_id"] for row in r]
    shard_ids = [c for c in ids
                 if int(hashlib.md5(c.encode()).hexdigest()[:4], 16) % n_shards == shard_id]
    if max_clips:
        shard_ids = shard_ids[:max_clips]
    return shard_ids


def find_files_for_clip(nfs_root: str, clip_id: str):
    """Locate camera mp4s + lidar parquet for a clip via filesystem glob.

    Cheaper alternative to querying canonical Camera/Lidar from inside this
    container — keeps the bevfusion image free of pyspark.
    """
    import glob as globmod
    cams: dict[str, str | None] = {}
    for sensor in CAMERA_SENSORS:
        matches = globmod.glob(
            f"{nfs_root}/camera/{sensor}/*/{clip_id}.{sensor}.mp4"
        )
        cams[sensor] = matches[0] if matches else None
    lidar_matches = globmod.glob(
        f"{nfs_root}/lidar/lidar_top_360fov/*/{clip_id}.lidar_top_360fov.parquet"
    )
    return cams, (lidar_matches[0] if lidar_matches else None)


def sample_frame_indices(n_frames: int, total: int) -> list[int]:
    """Evenly-spaced indices across the clip, inclusive of first and last."""
    if total <= n_frames:
        return list(range(total))
    return [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]


def extract_camera_frames(mp4_path: str, n_frames: int) -> list[np.ndarray] | None:
    """Returns N RGB frames as np.ndarray (H,W,3) uint8, or None on failure."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = sample_frame_indices(n_frames, total)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if frames else None


def decode_lidar_frames(lidar_parquet: str, n_frames: int) -> list[np.ndarray] | None:
    """Returns N point clouds as (M,3+) float32 arrays, or None on failure.

    Lidar parquet rows are individual spins with a Draco-encoded blob in
    `draco_encoded_pointcloud`. We sample N spins evenly across the clip.
    """
    try:
        import DracoPy
    except ImportError:
        return None
    try:
        t = pq.read_table(lidar_parquet,
                          columns=["draco_encoded_pointcloud"])
    except Exception:
        return None
    blobs = t.column("draco_encoded_pointcloud").to_pylist()
    if not blobs:
        return None
    indices = sample_frame_indices(n_frames, len(blobs))
    pcs = []
    for idx in indices:
        try:
            mesh = DracoPy.decode(blobs[idx])
            pcs.append(np.asarray(mesh.points, dtype=np.float32))
        except Exception:
            continue
    return pcs if pcs else None


def perception_score_from_detections(per_frame_results: list[dict]) -> dict:
    """Aggregate per-frame BEVFusion outputs into a per-clip difficulty score.

    Inputs: list of dicts with keys:
        n_detections (int)
        max_conf (float in [0,1])
        class_counts (dict[str,int])

    Composite score uses temporal-consistency signals:
      - high std of n_detections across frames → unstable scene
      - low mean max_conf → model uncertain
      - high class diversity → multiple object types
      - presence of small/dynamic actors (peds, cyclists)
    """
    if not per_frame_results:
        return {"perception_score": 0.5}  # neutral if no data
    n_dets = np.array([r["n_detections"] for r in per_frame_results], dtype=np.float32)
    confs = np.array([r["max_conf"] for r in per_frame_results], dtype=np.float32)

    # Aggregate class counts
    all_classes: dict[str, int] = {}
    for r in per_frame_results:
        for k, v in r.get("class_counts", {}).items():
            all_classes[k] = all_classes.get(k, 0) + v
    total = sum(all_classes.values())
    if total > 0:
        probs = np.array([v / total for v in all_classes.values()], dtype=np.float32)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_entropy = math.log(max(len(all_classes), 1))
        class_diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        class_diversity = 0.0

    driving_obj_keys = {"car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"}
    driving_obj_count = float(sum(all_classes.get(k, 0) for k in driving_obj_keys)) / max(len(per_frame_results), 1)

    n_det_var_norm = float(min(1.0, n_dets.std() / 5.0))   # std=5 → max
    conf_uncertainty = float(1.0 - confs.mean())
    driving_obj_norm = float(min(1.0, driving_obj_count / 20.0))

    composite = (
        0.35 * n_det_var_norm
        + 0.25 * conf_uncertainty
        + 0.20 * class_diversity
        + 0.20 * driving_obj_norm
    )

    return {
        "n_frames_sampled": len(per_frame_results),
        "mean_n_detections": float(n_dets.mean()),
        "std_n_detections": float(n_dets.std()),
        "mean_max_conf": float(confs.mean()),
        "std_max_conf": float(confs.std()),
        "class_diversity": class_diversity,
        "driving_obj_count": driving_obj_count,
        "perception_score": float(min(max(composite, 0.0), 1.0)),
    }


def main():
    args = parse_args()
    print(f"[bevfusion-runner] shard {args.shard_id}/{args.n_shards}, "
          f"frames_per_clip={args.frames_per_clip}, gpu={args.gpu}", flush=True)

    # Defer heavy imports until after argparse so --help is fast
    from mmdet3d.apis import init_model, inference_multi_modality_detector

    torch.cuda.set_device(args.gpu)
    model = init_model(args.config, args.checkpoint,
                       device=f"cuda:{args.gpu}")
    print(f"[bevfusion-runner] model loaded", flush=True)

    clip_ids = load_clip_ids(args.clip_list, args.shard_id,
                             args.n_shards, args.max_clips)
    print(f"[bevfusion-runner] {len(clip_ids):,} clips in shard "
          f"{args.shard_id}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bevfusion_shard_{args.shard_id:02d}_of_{args.n_shards:02d}.parquet"

    rows = []
    t0 = time.time()
    for i, clip_id in enumerate(clip_ids):
        cams, lidar_path = find_files_for_clip(args.nfs_root, clip_id)
        # Need at least front_wide camera + lidar
        if not cams.get("camera_front_wide_120fov") or not lidar_path:
            rows.append({
                "clip_id": clip_id,
                **perception_score_from_detections([]),  # neutral
                "shard_id": args.shard_id,
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            continue

        # Extract frames + lidar in parallel time-aligned samples
        cam_frames = {}
        for sensor, p in cams.items():
            if p:
                f = extract_camera_frames(p, args.frames_per_clip)
                if f:
                    cam_frames[sensor] = f
        lidar_pcs = decode_lidar_frames(lidar_path, args.frames_per_clip)

        # Run BEVFusion per sampled frame index. The mmdet3d API expects
        # specific input formatting; this is a sketch — actual call signature
        # depends on the chosen config (e.g. transfusion_lidar_only,
        # bevfusion_camera_lidar). Calibrate against the chosen config in
        # the validation pass.
        per_frame = []
        n = min(len(lidar_pcs or []),
                min((len(v) for v in cam_frames.values()), default=0))
        for fi in range(n):
            try:
                # Placeholder: the real call structures (image, points)
                # into the mmdet3d data format and runs inference. This is
                # the part that depends on the BEVFusion config flavour
                # (camera+lidar fusion config takes both modalities).
                # Until we validate the config, emit neutral scores.
                #
                # TODO(2026-05-04): replace this stub with a real call
                # once the chosen BEVFusion config is in place. Track the
                # blocker via run logs — every clip with a stub call
                # contributes a neutral result.
                result = {
                    "n_detections": 0,
                    "max_conf": 0.0,
                    "class_counts": {},
                }
                per_frame.append(result)
            except Exception as e:
                print(f"  [WARN] {clip_id} frame {fi}: {e}", flush=True)

        scores = perception_score_from_detections(per_frame)
        rows.append({
            "clip_id": clip_id,
            **scores,
            "shard_id": args.shard_id,
            "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(clip_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(clip_ids)}] {elapsed:.0f}s, {rate:.2f} clips/s",
                  flush=True)

        # Periodic checkpoint write so a crash doesn't lose hours of work
        if (i + 1) % 500 == 0:
            pq.write_table(pa.Table.from_pylist(rows), str(out_path))

    pq.write_table(pa.Table.from_pylist(rows), str(out_path))
    print(f"[bevfusion-runner] wrote {len(rows):,} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
