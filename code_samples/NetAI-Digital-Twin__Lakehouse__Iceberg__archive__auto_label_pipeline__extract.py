"""Stage 1: Extract a small subset of clips from the dataset ZIPs to local disk.

Reads clip_index.parquet to identify clips, then extracts camera MP4s,
LiDAR parquets, calibration, and egomotion from the chunked ZIPs —
all read-only on the source mount.
"""

import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq

SNAP_DEFAULT = (
    "/mnt/datax/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
    "/snapshots/0c8e5b7813562ab6e907e55db6ead3351922073f"
)

CAMERA_SENSORS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]


@dataclass
class ExtractionResult:
    """Paths produced by extracting one clip."""

    clip_id: str
    camera_mp4s: dict  # sensor_name -> path
    camera_timestamps: dict  # sensor_name -> path
    lidar_parquet: Optional[str] = None
    egomotion_parquet: Optional[str] = None
    calibration_intrinsics: Optional[str] = None
    calibration_extrinsics: Optional[str] = None
    vehicle_dimensions: Optional[str] = None


def select_clips(
    source_path: str = SNAP_DEFAULT,
    chunk: int = 0,
    num_clips: int = 10,
    split: str = "train",
) -> List[dict]:
    """Select clips from clip_index.parquet for a given chunk."""
    idx_path = os.path.join(source_path, "clip_index.parquet")
    table = pq.read_table(idx_path)
    df = table.to_pandas().reset_index()  # clip_id is the pandas index
    subset = df[(df["chunk"] == chunk) & (df["split"] == split) & (df["clip_is_valid"])]
    clips = subset.head(num_clips).to_dict("records")
    print(f"Selected {len(clips)} clips from chunk {chunk} (split={split})")
    return clips


def _extract_matching_files(zip_path: str, clip_id: str, dest_dir: str) -> List[str]:
    """Extract files matching a clip_id from a ZIP to dest_dir. Returns paths."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.startswith(clip_id):
                dest = os.path.join(dest_dir, os.path.basename(name))
                with z.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                extracted.append(dest)
    return extracted


def _find_zip_for_chunk(sensor_dir: str, sensor_name: str, chunk: int) -> Optional[str]:
    """Locate the chunk ZIP for a sensor, resolving symlinks."""
    zip_name = f"{sensor_name}.chunk_{chunk:04d}.zip"
    zip_path = os.path.join(sensor_dir, zip_name)
    if os.path.exists(zip_path):
        return os.path.realpath(zip_path)
    return None


def _find_parquet_for_chunk(cal_dir: str, cal_name: str, chunk: int) -> Optional[str]:
    """Locate a calibration parquet chunk file."""
    pq_name = f"{cal_name}.chunk_{chunk:04d}.parquet"
    pq_path = os.path.join(cal_dir, pq_name)
    if os.path.exists(pq_path):
        return os.path.realpath(pq_path)
    return None


def extract_clip(
    clip_id: str,
    chunk: int,
    source_path: str = SNAP_DEFAULT,
    output_root: str = "/tmp/autolabel_workdir",
    cameras: Optional[List[str]] = None,
) -> ExtractionResult:
    """Extract all sensor data for a single clip to local disk."""
    if cameras is None:
        cameras = CAMERA_SENSORS

    clip_dir = os.path.join(output_root, clip_id)
    cam_dir = os.path.join(clip_dir, "camera")
    lid_dir = os.path.join(clip_dir, "lidar")
    ego_dir = os.path.join(clip_dir, "ego")
    cal_dir = os.path.join(clip_dir, "calibration")
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(lid_dir, exist_ok=True)
    os.makedirs(ego_dir, exist_ok=True)
    os.makedirs(cal_dir, exist_ok=True)

    result = ExtractionResult(clip_id=clip_id, camera_mp4s={}, camera_timestamps={})

    # --- Camera ---
    for sensor in cameras:
        sensor_dir = os.path.join(source_path, "camera", sensor)
        zip_path = _find_zip_for_chunk(sensor_dir, sensor, chunk)
        if zip_path is None:
            continue
        files = _extract_matching_files(zip_path, clip_id, cam_dir)
        for f in files:
            if f.endswith(".mp4"):
                result.camera_mp4s[sensor] = f
            elif f.endswith(".timestamps.parquet"):
                result.camera_timestamps[sensor] = f

    # --- LiDAR ---
    lid_sensor_dir = os.path.join(source_path, "lidar", "lidar_top_360fov")
    lid_zip = _find_zip_for_chunk(lid_sensor_dir, "lidar_top_360fov", chunk)
    if lid_zip:
        files = _extract_matching_files(lid_zip, clip_id, lid_dir)
        for f in files:
            if f.endswith(".parquet"):
                result.lidar_parquet = f

    # --- Egomotion ---
    ego_sensor_dir = os.path.join(source_path, "labels", "egomotion")
    ego_zip = _find_zip_for_chunk(ego_sensor_dir, "egomotion", chunk)
    if ego_zip:
        files = _extract_matching_files(ego_zip, clip_id, ego_dir)
        for f in files:
            if f.endswith(".parquet"):
                result.egomotion_parquet = f

    # --- Calibration (parquet files, not zips) ---
    for cal_name, attr in [
        ("camera_intrinsics", "calibration_intrinsics"),
        ("sensor_extrinsics", "calibration_extrinsics"),
        ("vehicle_dimensions", "vehicle_dimensions"),
    ]:
        cal_sensor_dir = os.path.join(source_path, "calibration", cal_name)
        pq_path = _find_parquet_for_chunk(cal_sensor_dir, cal_name, chunk)
        if pq_path:
            # Read only rows for this clip_id from the chunk parquet
            table = pq.read_table(pq_path)
            df = table.to_pandas().reset_index()
            clip_rows = df[df["clip_id"] == clip_id]
            if not clip_rows.empty:
                import pyarrow as pa

                out_path = os.path.join(cal_dir, f"{cal_name}.parquet")
                pq.write_table(pa.Table.from_pandas(clip_rows), out_path)
                setattr(result, attr, out_path)

    return result


def extract_clips(
    clips: List[dict],
    source_path: str = SNAP_DEFAULT,
    output_root: str = "/tmp/autolabel_workdir",
    cameras: Optional[List[str]] = None,
) -> List[ExtractionResult]:
    """Extract multiple clips. Each dict must have 'clip_id' and 'chunk'."""
    results = []
    for i, clip in enumerate(clips):
        print(f"  Extracting clip {i + 1}/{len(clips)}: {clip['clip_id']}")
        r = extract_clip(
            clip_id=clip["clip_id"],
            chunk=clip["chunk"],
            source_path=source_path,
            output_root=output_root,
            cameras=cameras,
        )
        results.append(r)
    return results
