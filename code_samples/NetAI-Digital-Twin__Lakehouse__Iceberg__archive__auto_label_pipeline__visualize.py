"""Stage 6: Visualization — BEV plots and camera overlays."""

import math
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detect_2d import Detection2D
from .detect_3d import Detection3D

# Color map for classes (BGR)
CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 200, 100),
    "bus": (0, 150, 200),
    "person": (0, 0, 255),
    "bicycle": (255, 100, 0),
    "motorcycle": (255, 0, 150),
    "traffic_light": (0, 255, 255),
    "stop_sign": (128, 0, 255),
}


def draw_2d_detections(
    image: np.ndarray,
    detections: List[Detection2D],
) -> np.ndarray:
    """Draw 2D bounding boxes on a camera image."""
    vis = image.copy()
    for det in detections:
        color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def draw_bev(
    lidar_points: np.ndarray,
    detections_3d: List[Detection3D],
    x_range: tuple = (-50, 50),
    y_range: tuple = (-50, 50),
    resolution: float = 0.1,
) -> np.ndarray:
    """Render a bird's-eye view with LiDAR points and 3D boxes.

    Args:
        lidar_points: (N, 3) point cloud.
        detections_3d: 3D detections to draw.
        x_range: (min, max) in meters for x-axis.
        y_range: (min, max) in meters for y-axis.
        resolution: meters per pixel.

    Returns:
        (H, W, 3) BGR image.
    """
    w = int((x_range[1] - x_range[0]) / resolution)
    h = int((y_range[1] - y_range[0]) / resolution)
    bev = np.zeros((h, w, 3), dtype=np.uint8)

    def to_px(x, y):
        px = int((x - x_range[0]) / resolution)
        py = int((y_range[1] - y) / resolution)  # flip y
        return px, py

    # Draw LiDAR points
    mask = (
        (lidar_points[:, 0] >= x_range[0]) & (lidar_points[:, 0] < x_range[1])
        & (lidar_points[:, 1] >= y_range[0]) & (lidar_points[:, 1] < y_range[1])
    )
    pts = lidar_points[mask]
    for p in pts[::3]:  # subsample for speed
        px, py = to_px(p[0], p[1])
        if 0 <= px < w and 0 <= py < h:
            bev[py, px] = (80, 80, 80)

    # Draw ego vehicle
    ego_px, ego_py = to_px(0, 0)
    cv2.circle(bev, (ego_px, ego_py), 5, (255, 255, 255), -1)
    cv2.putText(bev, "EGO", (ego_px + 8, ego_py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw 3D boxes
    for det in detections_3d:
        color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
        corners = _box_corners_bev(det)
        pts_px = [to_px(c[0], c[1]) for c in corners]
        pts_arr = np.array(pts_px, dtype=np.int32)
        cv2.polylines(bev, [pts_arr], isClosed=True, color=color, thickness=2)

        # Label
        cx_px, cy_px = to_px(det.x, det.y)
        label = f"{det.class_name[:3]} {det.confidence:.1f}"
        cv2.putText(bev, label, (cx_px - 15, cy_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Scale bar and axis labels
    cv2.putText(bev, "10m", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    bar_len = int(10.0 / resolution)
    cv2.line(bev, (10, h - 20), (10 + bar_len, h - 20), (180, 180, 180), 2)

    return bev


def _box_corners_bev(det: Detection3D) -> List[tuple]:
    """Get 4 BEV corners of a 3D box."""
    cos_y = math.cos(det.yaw)
    sin_y = math.sin(det.yaw)
    hl, hw = det.length / 2, det.width / 2

    corners_local = [
        (hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)
    ]
    corners_world = []
    for dx, dy in corners_local:
        rx = cos_y * dx - sin_y * dy + det.x
        ry = sin_y * dx + cos_y * dy + det.y
        corners_world.append((rx, ry))
    return corners_world


def save_visualizations(
    clip_id: str,
    spin_idx: int,
    lidar_points: np.ndarray,
    detections_3d: List[Detection3D],
    camera_image: Optional[np.ndarray] = None,
    detections_2d: Optional[List[Detection2D]] = None,
    output_dir: str = "/tmp/autolabel_workdir/visualizations",
) -> Dict[str, str]:
    """Save BEV and camera overlay visualizations.

    Returns:
        Dict of visualization type -> file path.
    """
    clip_viz_dir = os.path.join(output_dir, clip_id)
    os.makedirs(clip_viz_dir, exist_ok=True)
    paths = {}

    # BEV
    bev = draw_bev(lidar_points, detections_3d)
    bev_path = os.path.join(clip_viz_dir, f"bev_{spin_idx:04d}.png")
    cv2.imwrite(bev_path, bev)
    paths["bev"] = bev_path

    # Camera overlay
    if camera_image is not None and detections_2d is not None:
        cam_vis = draw_2d_detections(camera_image, detections_2d)
        cam_path = os.path.join(clip_viz_dir, f"cam_{spin_idx:04d}.png")
        cv2.imwrite(cam_path, cam_vis)
        paths["camera"] = cam_path

    return paths
