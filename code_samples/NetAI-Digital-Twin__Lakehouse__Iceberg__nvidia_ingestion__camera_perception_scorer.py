"""Camera-only perception scorer for Gold edge-case scoring.

For each clip, samples N frames from the front_wide_120fov MP4, runs
YOLOv8 detection, and computes per-clip perception-complexity features.

Design notes
------------
* Runs inside the `av-perception:latest` Docker image (torch + ultralytics
  + opencv + decord pre-installed, with GPU access).
* Does NOT depend on Spark. Reads clip_index / data_collection as plain
  parquet via pyarrow; writes a per-shard parquet that can later be
  `add_files()`-registered into Iceberg.
* Parallelises across both available GPUs via `--gpu` shard selection.
* Output schema:

    clip_id             string
    n_frames_sampled    int
    mean_det_count      double
    std_det_count       double
    mean_conf           double
    max_conf            double
    class_diversity     int       # unique COCO classes detected
    driving_obj_count   double    # mean per-frame count of vehicle/person classes
    perception_score    double    # composite ∈ [0, 1]
    yolo_model          string
    source_mp4          string

CLI
---
    # Full run on GPU 0, writing shard 0 of 2:
    docker run --rm --gpus '"device=0"' \
      -v .../Iceberg:/work -w /work av-perception:latest \
      python -m nvidia_ingestion.camera_perception_scorer \
        --gpu 0 --shard 0 --num-shards 2

    # Small prototype (100 clips, GPU 0):
    docker run --rm --gpus '"device=0"' \
      -v .../Iceberg:/work -w /work av-perception:latest \
      python -m nvidia_ingestion.camera_perception_scorer \
        --gpu 0 --limit 100 --out-dir /work/tmp_perception
"""
from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# COCO classes commonly relevant on-road. Indices from the standard 80-class
# YOLOv8 COCO taxonomy.
DRIVING_COCO_IDS = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
    11,  # stop sign
}


# -----------------------------------------------------------------------------
# Dataset layout helpers
# -----------------------------------------------------------------------------

MP4_PATTERN = re.compile(r"([0-9a-f-]{36})\.camera_front_wide_120fov\.mp4$")


def build_clip_to_mp4_index(camera_root: str) -> Dict[str, str]:
    """Walk camera_front_wide_120fov chunk dirs to build {clip_id: mp4_path}."""
    index: Dict[str, str] = {}
    for chunk in sorted(os.listdir(camera_root)):
        cdir = os.path.join(camera_root, chunk)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            m = MP4_PATTERN.match(fname)
            if m:
                index[m.group(1)] = os.path.join(cdir, fname)
    return index


def load_clip_ids(clip_index_path: str) -> List[str]:
    """Return all valid clip_ids from clip_index.parquet."""
    tbl = pq.read_table(clip_index_path, columns=["clip_id", "clip_is_valid"])
    rows = tbl.to_pylist()
    return [r["clip_id"] for r in rows if r["clip_is_valid"]]


# -----------------------------------------------------------------------------
# Frame sampling
# -----------------------------------------------------------------------------

def sample_frames(mp4_path: str, n_frames: int = 6) -> Optional[np.ndarray]:
    """Sample n_frames evenly spaced frames from an MP4.

    Uses decord for fast random-access video decode. Returns a uint8
    array shaped (n_frames, H, W, 3) in RGB, or None on read failure.
    """
    import decord
    try:
        vr = decord.VideoReader(mp4_path, ctx=decord.cpu(0))
    except Exception:
        return None
    total = len(vr)
    if total < 2:
        return None
    idx = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
    try:
        frames = vr.get_batch(idx).asnumpy()  # (N, H, W, 3) RGB
    except Exception:
        return None
    return frames


# -----------------------------------------------------------------------------
# Detection + scoring
# -----------------------------------------------------------------------------

def score_clip(
    detector,
    clip_id: str,
    mp4_path: str,
    n_frames: int,
    conf_threshold: float,
    device: str,
) -> Optional[Dict]:
    """Run YOLOv8 on sampled frames; compute per-clip stats.

    Returns None if the clip could not be read.
    """
    frames = sample_frames(mp4_path, n_frames=n_frames)
    if frames is None or len(frames) == 0:
        return None

    # Batched YOLO inference. `detector.predict` accepts a list of ndarrays.
    results = detector.predict(
        list(frames),
        verbose=False,
        conf=conf_threshold,
        device=device,
    )

    per_frame_counts: List[int] = []
    per_frame_confs: List[float] = []
    driving_counts: List[int] = []
    class_union: set = set()
    max_conf = 0.0

    for r in results:
        boxes = r.boxes
        n = int(boxes.cls.shape[0]) if boxes is not None else 0
        per_frame_counts.append(n)
        if n == 0:
            per_frame_confs.append(0.0)
            driving_counts.append(0)
            continue
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        per_frame_confs.append(float(np.mean(confs)))
        class_union.update(classes.tolist())
        driving_counts.append(int(np.sum([c in DRIVING_COCO_IDS for c in classes])))
        max_conf = max(max_conf, float(np.max(confs)))

    counts = np.asarray(per_frame_counts, dtype=float)
    confs = np.asarray(per_frame_confs, dtype=float)
    drv = np.asarray(driving_counts, dtype=float)

    # Composite perception score ∈ [0, 1]:
    #   40% normalised mean driving-object count (cap at 15 → 1.0)
    #   25% std of detection count across frames (scene-dynamics proxy)
    #   20% class diversity (normalise 0..8+ driving classes → 0..1)
    #   15% low-confidence penalty (low avg confidence = ambiguous scene)
    mean_drv = float(np.mean(drv))
    drv_norm = min(1.0, mean_drv / 15.0)
    dyn_norm = min(1.0, float(np.std(counts)) / 10.0)
    drv_classes = len(class_union & DRIVING_COCO_IDS)
    div_norm = min(1.0, drv_classes / 8.0)
    amb_norm = 1.0 - min(1.0, float(np.mean(confs[confs > 0])) if np.any(confs > 0) else 0.0)
    perception = (
        0.40 * drv_norm
        + 0.25 * dyn_norm
        + 0.20 * div_norm
        + 0.15 * amb_norm
    )

    return {
        "clip_id": clip_id,
        "n_frames_sampled": int(len(frames)),
        "mean_det_count": float(np.mean(counts)),
        "std_det_count": float(np.std(counts)),
        "mean_conf": float(np.mean(confs)),
        "max_conf": float(max_conf),
        "class_diversity": int(len(class_union)),
        "driving_obj_count": float(np.mean(drv)),
        "perception_score": float(perception),
        "source_mp4": mp4_path,
    }


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-path",
                    default=os.environ.get(
                        "NVIDIA_SOURCE_PATH",
                        "/mnt/netai-e2e/nvidia-physicalai-av-subset",
                    ))
    ap.add_argument("--camera-sensor", default="camera_front_wide_120fov")
    ap.add_argument("--yolo-weights", default="yolov8n.pt",
                    help="Path or name of YOLO weights. 'yolov8n.pt' etc.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--conf-threshold", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many clips (prototype mode).")
    ap.add_argument("--shard", type=int, default=0,
                    help="Process clips where hash(clip_id) %% num_shards == shard.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir",
                    default="/mnt/netai-e2e/nvidia-physicalai-av-subset/.perception")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--skip-existing-glob", default="",
                    help="Glob of existing perception parquet(s) whose clip_ids "
                         "should be excluded (fill-in runs).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"perception_shard_{args.shard:02d}_of_{args.num_shards:02d}.parquet",
    )

    # Build index of clips with mp4s
    camera_root = os.path.join(args.source_path, "camera", args.camera_sensor)
    print(f"[{time.strftime('%H:%M:%S')}] indexing mp4s under {camera_root} ...", flush=True)
    mp4_index = build_clip_to_mp4_index(camera_root)
    print(f"  found {len(mp4_index):,} mp4s", flush=True)

    # Load target clip list
    clip_index_path = os.path.join(args.source_path, "clip_index.parquet")
    target_clips = load_clip_ids(clip_index_path)
    print(f"  clip_index: {len(target_clips):,} valid clips", flush=True)

    # Shard + limit. Use a deterministic hash (md5 of utf-8 bytes) so every
    # process shards identically — Python's builtin hash() is randomized
    # per-process via PYTHONHASHSEED and silently splits work unevenly.
    def _shard_pred(cid: str) -> bool:
        digest = hashlib.md5(cid.encode("utf-8")).digest()[:4]
        return int.from_bytes(digest, "big") % args.num_shards == args.shard

    # Optional: skip clips already scored in existing parquet(s) (fill-in runs)
    already_scored: set = set()
    if args.skip_existing_glob:
        for p in _glob.glob(args.skip_existing_glob):
            try:
                existing = pq.read_table(p, columns=["clip_id"]).column("clip_id").to_pylist()
                already_scored.update(existing)
            except Exception as e:
                print(f"  warn: could not read {p}: {e}", flush=True)
        print(f"  skip-existing: {len(already_scored):,} clip_ids already scored", flush=True)

    todo = [cid for cid in target_clips
            if cid in mp4_index and _shard_pred(cid) and cid not in already_scored]
    if args.limit:
        todo = todo[:args.limit]
    print(f"  shard {args.shard}/{args.num_shards}: {len(todo):,} clips to score"
          f"  → {out_path}", flush=True)
    if not todo:
        print("  nothing to do, exiting.", flush=True)
        return 0

    # Load YOLO
    import torch
    from ultralytics import YOLO

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.yolo_weights} on {device} ...",
          flush=True)
    model = YOLO(args.yolo_weights)
    model.to(device)
    # model.model_name is a read-only property on the underlying nn.Module,
    # so keep the label in a plain variable rather than on the object.
    model_label = os.path.basename(args.yolo_weights)

    # Score loop
    records: List[Dict] = []
    t0 = time.time()
    n_fail = 0
    for i, clip_id in enumerate(todo, 1):
        rec = score_clip(
            model,
            clip_id=clip_id,
            mp4_path=mp4_index[clip_id],
            n_frames=args.n_frames,
            conf_threshold=args.conf_threshold,
            device=device,
        )
        if rec is None:
            n_fail += 1
            continue
        rec["yolo_model"] = model_label
        records.append(rec)
        if i % args.progress_every == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(
                f"  [{time.strftime('%H:%M:%S')}] {i}/{len(todo)} "
                f"({rate:.2f} clip/s, eta {eta/60:.1f}m, fails={n_fail})",
                flush=True,
            )

    # Write parquet
    if not records:
        print("  no records produced.", flush=True)
        return 1
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, out_path, compression="zstd")
    print(f"[{time.strftime('%H:%M:%S')}] wrote {len(records)} rows → {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
