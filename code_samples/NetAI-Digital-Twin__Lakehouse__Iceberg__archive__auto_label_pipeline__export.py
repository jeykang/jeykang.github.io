"""Stage 5: Export annotations in KITTI and Parquet formats."""

import math
import os
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .detect_3d import Detection3D


def export_kitti(
    detections_by_spin: Dict[int, List[Detection3D]],
    output_dir: str,
    clip_id: str,
) -> str:
    """Export 3D detections in KITTI label format.

    One text file per spin/frame with columns:
    type truncated occluded alpha bbox_2d(x1,y1,x2,y2) dims(h,w,l) loc(x,y,z) ry score

    Returns:
        Path to the label directory.
    """
    label_dir = os.path.join(output_dir, clip_id, "kitti_labels")
    os.makedirs(label_dir, exist_ok=True)

    for spin_idx, dets in sorted(detections_by_spin.items()):
        label_path = os.path.join(label_dir, f"{spin_idx:06d}.txt")
        with open(label_path, "w") as f:
            for det in dets:
                d2 = det.detection_2d
                # KITTI format: type truncated occluded alpha
                #   bbox(left top right bottom) dimensions(h w l)
                #   location(x y z) rotation_y score
                if d2 is not None:
                    bbox_str = f"{d2.x1:.2f} {d2.y1:.2f} {d2.x2:.2f} {d2.y2:.2f}"
                else:
                    bbox_str = "0.00 0.00 0.00 0.00"
                line = (
                    f"{det.class_name} "
                    f"0.0 0 0.0 "
                    f"{bbox_str} "
                    f"{det.height:.2f} {det.width:.2f} {det.length:.2f} "
                    f"{det.x:.2f} {det.y:.2f} {det.z:.2f} "
                    f"{det.yaw:.4f} {det.confidence:.4f}"
                )
                f.write(line + "\n")

    return label_dir


def export_parquet(
    detections_by_spin: Dict[int, List[Detection3D]],
    lidar_timestamps: List[int],
    output_dir: str,
    clip_id: str,
) -> str:
    """Export 3D detections as a single Parquet file per clip.

    Returns:
        Path to the output parquet file.
    """
    os.makedirs(os.path.join(output_dir, clip_id), exist_ok=True)
    out_path = os.path.join(output_dir, clip_id, "annotations.parquet")

    rows = []
    for spin_idx, dets in sorted(detections_by_spin.items()):
        ts = lidar_timestamps[spin_idx] if spin_idx < len(lidar_timestamps) else 0
        for det in dets:
            rows.append({
                "clip_id": clip_id,
                "spin_index": spin_idx,
                "timestamp": ts,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "x": det.x,
                "y": det.y,
                "z": det.z,
                "length": det.length,
                "width": det.width,
                "height": det.height,
                "yaw": det.yaw,
                "bbox_2d_x1": det.detection_2d.x1 if det.detection_2d else 0.0,
                "bbox_2d_y1": det.detection_2d.y1 if det.detection_2d else 0.0,
                "bbox_2d_x2": det.detection_2d.x2 if det.detection_2d else 0.0,
                "bbox_2d_y2": det.detection_2d.y2 if det.detection_2d else 0.0,
                "camera_name": det.detection_2d.camera_name if det.detection_2d else "lidar_only",
                "num_lidar_points": det.num_lidar_points,
                "dimension_score": det.dimension_score,
            })

    if not rows:
        # Write empty parquet with schema
        schema = pa.schema([
            ("clip_id", pa.string()),
            ("spin_index", pa.int64()),
            ("timestamp", pa.int64()),
            ("class_name", pa.string()),
            ("confidence", pa.float64()),
            ("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64()),
            ("length", pa.float64()), ("width", pa.float64()), ("height", pa.float64()),
            ("yaw", pa.float64()),
            ("bbox_2d_x1", pa.float64()), ("bbox_2d_y1", pa.float64()),
            ("bbox_2d_x2", pa.float64()), ("bbox_2d_y2", pa.float64()),
            ("camera_name", pa.string()),
            ("num_lidar_points", pa.int64()),
            ("dimension_score", pa.float64()),
        ])
        pq.write_table(pa.table({f: [] for f in schema.names}, schema=schema), out_path)
    else:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)

    return out_path


def export_summary(
    all_clip_detections: Dict[str, Dict[int, List[Detection3D]]],
    output_dir: str,
) -> str:
    """Write a summary CSV across all processed clips.

    Returns:
        Path to the summary CSV.
    """
    summary_path = os.path.join(output_dir, "pipeline_summary.csv")
    with open(summary_path, "w") as f:
        f.write("clip_id,total_spins,total_3d_detections,classes_found,avg_confidence\n")
        for clip_id, by_spin in all_clip_detections.items():
            all_dets = [d for dets in by_spin.values() for d in dets]
            classes = set(d.class_name for d in all_dets)
            avg_conf = np.mean([d.confidence for d in all_dets]) if all_dets else 0.0
            f.write(
                f"{clip_id},{len(by_spin)},{len(all_dets)},"
                f"\"{';'.join(sorted(classes))}\",{avg_conf:.4f}\n"
            )
    return summary_path
