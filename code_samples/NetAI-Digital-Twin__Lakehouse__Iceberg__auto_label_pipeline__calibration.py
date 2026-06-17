"""Camera calibration utilities for the NVIDIA PhysicalAI AV dataset.

Implements the f-theta (equidistant) camera model used by the dataset
and sensor extrinsic transforms between camera/LiDAR coordinate frames.
"""

import numpy as np
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class FThetaCamera:
    """F-theta (equidistant) camera intrinsics with polynomial distortion."""

    width: int
    height: int
    cx: float
    cy: float
    bw_poly: np.ndarray  # backward polynomial: angle -> radius (5 coefficients)
    fw_poly: np.ndarray  # forward polynomial: radius -> angle (5 coefficients)

    def project_to_pixel(self, points_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points in camera frame to pixel coordinates.

        Args:
            points_cam: (N, 3) array in camera coordinates (x-right, y-down, z-forward).

        Returns:
            pixels: (N, 2) array of [u, v] pixel coordinates.
            valid: (N,) boolean mask for points in front of camera.
        """
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        valid = z > 0.1

        # Angle from optical axis
        r_3d = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(r_3d, z)

        # F-theta backward model: theta -> pixel radius
        r_px = np.zeros_like(theta)
        for i, c in enumerate(self.bw_poly):
            r_px += c * theta ** (i + 1)

        # Direction in image plane
        phi = np.arctan2(y, x)
        u = self.cx + r_px * np.cos(phi)
        v = self.cy + r_px * np.sin(phi)

        # Check bounds
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

        return np.column_stack([u, v]), valid

    def pixel_to_ray(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to unit ray directions in camera frame.

        Args:
            pixels: (N, 2) array of [u, v] pixel coordinates.

        Returns:
            rays: (N, 3) unit ray directions in camera frame.
        """
        du = pixels[:, 0] - self.cx
        dv = pixels[:, 1] - self.cy
        r_px = np.sqrt(du ** 2 + dv ** 2)
        phi = np.arctan2(dv, du)

        # Forward model: pixel radius -> angle from optical axis
        theta = np.zeros_like(r_px)
        for i, c in enumerate(self.fw_poly):
            theta += c * r_px ** (i + 1)

        # Convert to 3D ray
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return np.column_stack([x, y, z])


@dataclass
class SensorExtrinsic:
    """Rigid transform from sensor frame to vehicle (rig) frame."""

    rotation: np.ndarray  # (3, 3) rotation matrix
    translation: np.ndarray  # (3,) translation vector

    def sensor_to_rig(self, points: np.ndarray) -> np.ndarray:
        """Transform points from sensor frame to rig (vehicle) frame."""
        return (self.rotation @ points.T).T + self.translation

    def rig_to_sensor(self, points: np.ndarray) -> np.ndarray:
        """Transform points from rig frame to sensor frame."""
        return (self.rotation.T @ (points - self.translation).T).T


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    r = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])
    return r


def load_camera_intrinsics(parquet_path: str) -> Dict[str, FThetaCamera]:
    """Load camera intrinsics from a calibration parquet file.

    Returns:
        Dict mapping camera_name -> FThetaCamera.
    """
    df = pq.read_table(parquet_path).to_pandas().reset_index()
    cameras = {}
    for _, row in df.iterrows():
        bw = np.array([row[f"bw_poly_{i}"] for i in range(5)])
        fw = np.array([row[f"fw_poly_{i}"] for i in range(5)])
        cameras[row["camera_name"]] = FThetaCamera(
            width=int(row["width"]),
            height=int(row["height"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            bw_poly=bw,
            fw_poly=fw,
        )
    return cameras


def load_sensor_extrinsics(parquet_path: str) -> Dict[str, SensorExtrinsic]:
    """Load sensor extrinsics from a calibration parquet file.

    Returns:
        Dict mapping sensor_name -> SensorExtrinsic.
    """
    df = pq.read_table(parquet_path).to_pandas().reset_index()
    extrinsics = {}
    for _, row in df.iterrows():
        R = _quat_to_rotation_matrix(
            float(row["qx"]), float(row["qy"]),
            float(row["qz"]), float(row["qw"]),
        )
        t = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        extrinsics[row["sensor_name"]] = SensorExtrinsic(rotation=R, translation=t)
    return extrinsics


def lidar_to_camera(
    points_lidar: np.ndarray,
    lidar_ext: SensorExtrinsic,
    camera_ext: SensorExtrinsic,
) -> np.ndarray:
    """Transform LiDAR points to camera frame via the rig (vehicle) frame."""
    points_rig = lidar_ext.sensor_to_rig(points_lidar)
    points_cam = camera_ext.rig_to_sensor(points_rig)
    return points_cam


def camera_to_lidar(
    points_cam: np.ndarray,
    lidar_ext: SensorExtrinsic,
    camera_ext: SensorExtrinsic,
) -> np.ndarray:
    """Transform camera-frame points to LiDAR frame via the rig frame."""
    points_rig = camera_ext.sensor_to_rig(points_cam)
    points_lidar = lidar_ext.rig_to_sensor(points_rig)
    return points_lidar
