"""GPU-accelerated voxelization using PyTorch.

Replaces the Python-loop voxelizers in detect_3d.py and
detect_3d_centerpoint.py.  All heavy work runs on CUDA via torch ops —
no per-point Python loops.
"""

from typing import Tuple

import numpy as np
import torch


def _voxelize_gpu(
    points_np: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    max_points_per_voxel: int,
    max_num_voxels: int,
    n_features: int,
    mode: str = "3d",
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core GPU voxelization shared by both detectors.

    Args:
        points_np: (N, C) point cloud, at least xyz.
        voxel_size: (3,) voxel dimensions.
        point_cloud_range: (6,) [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_voxel: Max points retained per voxel.
        max_num_voxels: Max total voxels.
        n_features: Number of point features to keep (3, 4, or 5).
        mode: "3d" for z,y,x coords (OpenPCDet PointPillars) or
              "pillar" for x,y coords (CenterPoint).
        device: CUDA device.

    Returns:
        voxels: (M, max_points, n_features) padded point features.
        coords: (M, 3) for 3d mode [z,y,x] or (M, 2) for pillar mode [x,y].
        num_points: (M,) actual point count per voxel.
    """
    # Pad/trim to n_features
    if points_np.shape[1] < n_features:
        points_np = np.hstack([
            points_np.astype(np.float32),
            np.zeros((len(points_np), n_features - points_np.shape[1]), np.float32),
        ])
    else:
        points_np = points_np[:, :n_features].astype(np.float32)

    pts = torch.from_numpy(points_np).to(device)
    pcr = torch.tensor(point_cloud_range, device=device, dtype=torch.float32)
    vs = torch.tensor(voxel_size, device=device, dtype=torch.float32)

    # Range filter
    mask = (
        (pts[:, 0] >= pcr[0]) & (pts[:, 0] < pcr[3])
        & (pts[:, 1] >= pcr[1]) & (pts[:, 1] < pcr[4])
        & (pts[:, 2] >= pcr[2]) & (pts[:, 2] < pcr[5])
    )
    pts = pts[mask]

    if len(pts) == 0:
        if mode == "pillar":
            return (np.zeros((0, max_points_per_voxel, n_features), np.float32),
                    np.zeros((0, 2), np.int32), np.zeros(0, np.int32))
        else:
            return (np.zeros((0, max_points_per_voxel, n_features), np.float32),
                    np.zeros((0, 3), np.int32), np.zeros(0, np.int32))

    # Grid indices
    gi = torch.floor((pts[:, :3] - pcr[:3]) / vs).to(torch.int64)
    grid_size = torch.ceil((pcr[3:6] - pcr[:3]) / vs).to(torch.int64)

    # Clamp
    gi[:, 0].clamp_(0, grid_size[0].item() - 1)
    gi[:, 1].clamp_(0, grid_size[1].item() - 1)
    gi[:, 2].clamp_(0, grid_size[2].item() - 1)

    # Flatten to single key
    keys = gi[:, 2] * grid_size[1] * grid_size[0] + gi[:, 1] * grid_size[0] + gi[:, 0]

    unique_keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)

    # Keep most-populated voxels if over limit
    if len(unique_keys) > max_num_voxels:
        _, top = torch.topk(counts, max_num_voxels)
        keep_mask = torch.isin(keys, unique_keys[top])
        pts = pts[keep_mask]
        keys = keys[keep_mask]
        unique_keys, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)

    n_vox = len(unique_keys)

    # Sort by voxel for contiguous memory access
    sort_idx = torch.argsort(inverse)
    sorted_pts = pts[sort_idx]
    sorted_inv = inverse[sort_idx]

    # Compute within-voxel point indices
    offsets = torch.zeros(n_vox + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, 0)
    within_idx = torch.arange(len(sorted_pts), device=device) - offsets[sorted_inv]

    # Keep first max_points_per_voxel per voxel
    keep = within_idx < max_points_per_voxel
    kept_pts = sorted_pts[keep]
    kept_inv = sorted_inv[keep]
    kept_within = within_idx[keep]

    # Scatter into voxel tensor
    voxels = torch.zeros((n_vox, max_points_per_voxel, n_features),
                         dtype=torch.float32, device=device)
    voxels[kept_inv, kept_within] = kept_pts

    num_points = torch.minimum(counts,
                               torch.tensor(max_points_per_voxel, device=device))

    # Build coordinates
    gsx = grid_size[0].item()
    gsy = grid_size[1].item()
    if mode == "pillar":
        # CenterPoint: (M, 2) as [x_idx, y_idx]
        x_idx = unique_keys % gsx
        y_idx = (unique_keys // gsx) % gsy
        coords = torch.stack([x_idx, y_idx], dim=1)
    else:
        # OpenPCDet: (M, 3) as [z_idx, y_idx, x_idx]
        x_idx = unique_keys % gsx
        y_idx = (unique_keys // gsx) % gsy
        z_idx = unique_keys // (gsy * gsx)
        coords = torch.stack([z_idx, y_idx, x_idx], dim=1)

    return (voxels.cpu().numpy(),
            coords.cpu().to(torch.int32).numpy(),
            num_points.cpu().to(torch.int32).numpy())


def voxelize_pointpillars(
    points: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    max_points_per_voxel: int = 20,
    max_num_voxels: int = 30000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU voxelization for PointPillars (OpenPCDet format).

    Returns:
        voxels: (M, max_points, 5) with features [x, y, z, intensity, timestamp].
        coords: (M, 3) as [z_idx, y_idx, x_idx].
        num_points: (M,) per-voxel point count.
    """
    return _voxelize_gpu(
        points, voxel_size, point_cloud_range,
        max_points_per_voxel, max_num_voxels,
        n_features=5, mode="3d",
    )


def voxelize_centerpoint(
    points: np.ndarray,
    voxel_size: np.ndarray,
    point_cloud_range: np.ndarray,
    grid_size: np.ndarray,
    max_points_per_voxel: int = 20,
    max_num_voxels: int = 40000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU voxelization + 9-feature encoding for CenterPoint.

    Returns:
        features: (M, max_points, 9) PillarVFE features.
        coords: (M, 2) as [x_idx, y_idx].
        num_points: (M,) per-voxel point count.
    """
    device = "cuda"

    raw_voxels, coords, num_points = _voxelize_gpu(
        points, voxel_size, point_cloud_range,
        max_points_per_voxel, max_num_voxels,
        n_features=4, mode="pillar",
    )

    if len(raw_voxels) == 0:
        return (np.zeros((0, max_points_per_voxel, 9), np.float32),
                coords, num_points)

    # Compute 9 PillarVFE features on GPU
    n_vox = len(raw_voxels)
    rv = torch.from_numpy(raw_voxels).to(device)  # (M, max_pts, 4)
    np_t = torch.from_numpy(num_points).to(device).float().clamp(min=1)  # (M,)

    features = torch.zeros((n_vox, max_points_per_voxel, 9),
                           dtype=torch.float32, device=device)

    # Features 0-3: x, y, z, intensity
    features[:, :, :4] = rv

    # Features 4-6: offset from cluster center (mean of points in pillar)
    sums = rv[:, :, :3].sum(dim=1)  # (M, 3)
    means = sums / np_t.unsqueeze(1)  # (M, 3)
    features[:, :, 4] = rv[:, :, 0] - means[:, 0:1]
    features[:, :, 5] = rv[:, :, 1] - means[:, 1:2]
    features[:, :, 6] = rv[:, :, 2] - means[:, 2:3]

    # Features 7-8: offset from pillar geometric center
    coords_t = torch.from_numpy(coords).to(device).float()
    pillar_cx = coords_t[:, 0] * voxel_size[0] + point_cloud_range[0] + voxel_size[0] / 2
    pillar_cy = coords_t[:, 1] * voxel_size[1] + point_cloud_range[1] + voxel_size[1] / 2
    features[:, :, 7] = rv[:, :, 0] - pillar_cx.unsqueeze(1)
    features[:, :, 8] = rv[:, :, 1] - pillar_cy.unsqueeze(1)

    # Zero out padded points
    pad_mask = (rv[:, :, :3].abs().sum(dim=2) == 0)
    features[pad_mask] = 0.0

    return features.cpu().numpy(), coords, num_points
