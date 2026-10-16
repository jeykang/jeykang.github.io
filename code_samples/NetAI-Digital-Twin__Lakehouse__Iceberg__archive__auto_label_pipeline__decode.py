"""Stage 2: Decode camera frames and LiDAR point clouds from extracted files.

Camera: MP4 → numpy frames at LiDAR-synchronized timestamps.
LiDAR:  Draco-encoded parquet → per-spin numpy point clouds.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import DracoPy
import numpy as np
import pyarrow.parquet as pq


@dataclass
class LidarSpin:
    """A single LiDAR 360-degree sweep."""

    spin_index: int
    timestamp: int  # microseconds
    points: np.ndarray  # (N, 3+) xyz and optional attributes


@dataclass
class CameraFrame:
    """A single camera image matched to a LiDAR spin."""

    sensor_name: str
    frame_index: int
    timestamp: int  # microseconds
    image: np.ndarray  # (H, W, 3) BGR


@dataclass
class DecodedClip:
    """All decoded sensor data for one clip."""

    clip_id: str
    lidar_spins: List[LidarSpin] = field(default_factory=list)
    camera_frames: Dict[str, List[CameraFrame]] = field(default_factory=dict)
    egomotion: Optional[np.ndarray] = None  # structured array


def decode_lidar(parquet_path: str, max_spins: Optional[int] = None) -> List[LidarSpin]:
    """Decode Draco-encoded LiDAR point clouds from a clip parquet.

    Args:
        parquet_path: Path to the clip's lidar parquet file.
        max_spins: If set, decode only the first N spins.

    Returns:
        List of LidarSpin objects ordered by spin_index.
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas().reset_index().sort_values("spin_index")

    if max_spins is not None:
        df = df.head(max_spins)

    spins = []
    for _, row in df.iterrows():
        draco_bytes = row["draco_encoded_pointcloud"]
        mesh = DracoPy.decode(draco_bytes)
        points = np.array(mesh.points, dtype=np.float32)
        spins.append(LidarSpin(
            spin_index=int(row["spin_index"]),
            timestamp=int(row["reference_timestamp"]),
            points=points,
        ))
    return spins


def _load_camera_timestamps(ts_parquet_path: str) -> np.ndarray:
    """Load frame timestamps from a camera timestamps parquet.

    Returns:
        Structured array with fields 'frame_index' and 'timestamp'.
    """
    table = pq.read_table(ts_parquet_path)
    df = table.to_pandas().reset_index().sort_values("frame_index").reset_index(drop=True)
    return df[["frame_index", "timestamp"]].to_numpy()


def _find_nearest_frame(cam_timestamps: np.ndarray, lidar_ts: int) -> Tuple[int, int]:
    """Find the camera frame index nearest to a LiDAR timestamp.

    Args:
        cam_timestamps: (N, 2) array of [frame_index, timestamp].
        lidar_ts: LiDAR reference timestamp in microseconds.

    Returns:
        (frame_index, timestamp) of the nearest camera frame.
    """
    diffs = np.abs(cam_timestamps[:, 1] - lidar_ts)
    idx = np.argmin(diffs)
    return int(cam_timestamps[idx, 0]), int(cam_timestamps[idx, 1])


def decode_camera_at_lidar_times(
    mp4_path: str,
    ts_parquet_path: str,
    sensor_name: str,
    lidar_timestamps: List[int],
) -> List[CameraFrame]:
    """Extract camera frames at timestamps closest to LiDAR spins.

    Args:
        mp4_path: Path to the camera MP4 file.
        ts_parquet_path: Path to the camera timestamps parquet.
        sensor_name: Camera sensor name (e.g. 'camera_front_wide_120fov').
        lidar_timestamps: List of LiDAR reference timestamps to match.

    Returns:
        List of CameraFrame objects, one per LiDAR timestamp.
    """
    cam_ts = _load_camera_timestamps(ts_parquet_path)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find which frame indices we need
    needed = []
    for lid_ts in lidar_timestamps:
        fi, ts = _find_nearest_frame(cam_ts, lid_ts)
        needed.append((fi, ts))

    # Sort by frame index for sequential read efficiency
    indexed_needed = sorted(enumerate(needed), key=lambda x: x[1][0])

    frames = [None] * len(needed)
    current_pos = -1

    for orig_idx, (fi, ts) in indexed_needed:
        if fi >= total_frames:
            continue
        if fi != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, img = cap.read()
        if ret:
            frames[orig_idx] = CameraFrame(
                sensor_name=sensor_name,
                frame_index=fi,
                timestamp=ts,
                image=img,
            )
        current_pos = fi + 1

    cap.release()
    return [f for f in frames if f is not None]


def decode_clip(
    extraction_result,
    cameras: Optional[List[str]] = None,
    max_spins: Optional[int] = None,
) -> DecodedClip:
    """Decode all sensor data for an extracted clip.

    Args:
        extraction_result: ExtractionResult from extract.py.
        cameras: List of camera sensors to decode. None = all available.
        max_spins: Limit LiDAR spins decoded (for faster testing).

    Returns:
        DecodedClip with decoded LiDAR and camera data.
    """
    result = DecodedClip(clip_id=extraction_result.clip_id)

    # Decode LiDAR
    if extraction_result.lidar_parquet:
        print(f"    Decoding LiDAR ({max_spins or 'all'} spins)...")
        result.lidar_spins = decode_lidar(
            extraction_result.lidar_parquet, max_spins=max_spins
        )

    lidar_timestamps = [s.timestamp for s in result.lidar_spins]

    # Decode cameras at LiDAR times
    available_cameras = cameras or list(extraction_result.camera_mp4s.keys())
    for sensor in available_cameras:
        mp4 = extraction_result.camera_mp4s.get(sensor)
        ts_pq = extraction_result.camera_timestamps.get(sensor)
        if mp4 and ts_pq and lidar_timestamps:
            print(f"    Decoding {sensor} frames...")
            frames = decode_camera_at_lidar_times(
                mp4, ts_pq, sensor, lidar_timestamps
            )
            result.camera_frames[sensor] = frames

    # Load egomotion
    if extraction_result.egomotion_parquet:
        ego_df = pq.read_table(extraction_result.egomotion_parquet).to_pandas()
        result.egomotion = ego_df.to_numpy()

    return result
