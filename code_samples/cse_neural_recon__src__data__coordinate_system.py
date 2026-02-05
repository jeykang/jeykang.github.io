"""
Unified Coordinate System for Multi-Environment Training.

This module provides coordinate transformations that allow training
on data from multiple environments with different world origins.

Key concepts:
1. Each environment has its own world coordinate system
2. We compute per-environment bounds from depth-projected points
3. Points are normalized to a canonical [-1, 1] or [0, 1] range
4. The model learns scale-invariant geometry

This enables:
- Training on warehouse, hospital, office data together
- Deployment to completely new environments
- Consistent SDF values regardless of environment size
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from PIL import Image


@dataclass
class EnvironmentBounds:
    """Stores coordinate bounds for an environment."""
    name: str
    min_bounds: np.ndarray  # (3,) - min x, y, z in world coords
    max_bounds: np.ndarray  # (3,) - max x, y, z in world coords
    center: np.ndarray      # (3,) - center point
    extent: np.ndarray      # (3,) - size in each dimension
    scale: float            # max extent (for uniform scaling)
    
    def to_normalized(self, points: np.ndarray) -> np.ndarray:
        """Transform world points to normalized [-1, 1] coordinates."""
        # Center and scale uniformly
        normalized = (points - self.center) / (self.scale / 2)
        return normalized
    
    def from_normalized(self, points: np.ndarray) -> np.ndarray:
        """Transform normalized [-1, 1] coordinates back to world."""
        world = points * (self.scale / 2) + self.center
        return world
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'min_bounds': self.min_bounds.tolist(),
            'max_bounds': self.max_bounds.tolist(),
            'center': self.center.tolist(),
            'extent': self.extent.tolist(),
            'scale': float(self.scale),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'EnvironmentBounds':
        """Deserialize from dictionary."""
        return cls(
            name=d['name'],
            min_bounds=np.array(d['min_bounds']),
            max_bounds=np.array(d['max_bounds']),
            center=np.array(d['center']),
            extent=np.array(d['extent']),
            scale=d['scale'],
        )


class UnifiedCoordinateSystem:
    """
    Manages coordinate transformations across multiple environments.
    
    For multi-environment training, each environment gets normalized to
    a common [-1, 1] space. The model learns geometry in this canonical
    space, making it environment-agnostic.
    
    For deployment to new environments:
    1. Compute bounds from initial depth observations
    2. Create EnvironmentBounds with those bounds
    3. Normalize query points using those bounds
    4. Model predictions are in normalized space
    """
    
    def __init__(self, normalize_mode: str = 'per_environment'):
        """
        Args:
            normalize_mode: How to normalize coordinates
                - 'per_environment': Each env normalized independently to [-1,1]
                - 'global': All envs share a global coordinate frame
                - 'metric': Keep metric coordinates, just center
        """
        self.normalize_mode = normalize_mode
        self.environments: Dict[str, EnvironmentBounds] = {}
        self.global_bounds: Optional[EnvironmentBounds] = None
        
    def compute_environment_bounds(
        self,
        env_name: str,
        sequence_dirs: List[str],
        num_sample_frames: int = 100,
        depth_scale: float = 0.001,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
    ) -> EnvironmentBounds:
        """
        Compute bounds for an environment by sampling depth frames.
        
        Args:
            env_name: Name for this environment
            sequence_dirs: List of sequence directories to sample from
            num_sample_frames: Number of frames to sample per sequence
            depth_scale: Scale factor for depth images (mm to m)
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            
        Returns:
            EnvironmentBounds for this environment
        """
        all_points = []
        
        for seq_dir in sequence_dirs:
            points = self._sample_points_from_sequence(
                seq_dir, num_sample_frames, depth_scale, min_depth, max_depth
            )
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            raise ValueError(f"No valid points found for environment {env_name}")
        
        all_points = np.vstack(all_points)
        
        # Compute bounds with small margin
        margin = 0.5  # 50cm margin
        min_bounds = all_points.min(axis=0) - margin
        max_bounds = all_points.max(axis=0) + margin
        center = (min_bounds + max_bounds) / 2
        extent = max_bounds - min_bounds
        scale = extent.max()  # Uniform scaling based on largest dimension
        
        bounds = EnvironmentBounds(
            name=env_name,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            center=center,
            extent=extent,
            scale=scale,
        )
        
        self.environments[env_name] = bounds
        
        print(f"Environment '{env_name}' bounds computed:")
        print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Extent: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
        print(f"  Scale: {scale:.2f} m")
        
        return bounds
    
    def _sample_points_from_sequence(
        self,
        seq_dir: str,
        num_frames: int,
        depth_scale: float,
        min_depth: float,
        max_depth: float,
    ) -> np.ndarray:
        """Sample 3D points from a sequence by unprojecting depth."""
        # Load intrinsics
        intrinsics_path = os.path.join(seq_dir, 'camera_info_left_intrinsics.json')
        if not os.path.exists(intrinsics_path):
            return np.array([]).reshape(0, 3)
            
        with open(intrinsics_path) as f:
            meta = json.load(f)
        K = np.array(meta['K']).reshape(3, 3)
        
        # Load poses
        pose_file = os.path.join(seq_dir, 'ground_truth.txt')
        poses = []
        with open(pose_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = [float(x) for x in line.strip().split()]
                if len(parts) >= 8:
                    mat = np.eye(4)
                    mat[:3, 3] = parts[1:4]
                    mat[:3, :3] = R.from_quat(parts[4:8]).as_matrix()
                    poses.append((parts[0], mat))
        
        # Sample depth frames
        depth_dir = os.path.join(seq_dir, 'depth_left')
        if not os.path.exists(depth_dir):
            return np.array([]).reshape(0, 3)
            
        depth_files = sorted(os.listdir(depth_dir))
        step = max(1, len(depth_files) // num_frames)
        sample_files = depth_files[::step][:num_frames]
        
        all_points = []
        for df in sample_files:
            ts = float(df.replace('.png', '')) / 1e9
            
            # Find nearest pose
            pose_idx = np.argmin([abs(p[0] - ts) for p in poses])
            pose = poses[pose_idx][1]
            
            # Load and filter depth
            depth_path = os.path.join(depth_dir, df)
            depth = np.array(Image.open(depth_path)).astype(np.float32) * depth_scale
            valid = (depth > min_depth) & (depth < max_depth)
            
            if valid.sum() == 0:
                continue
            
            # Unproject to camera frame
            v, u = np.where(valid)
            z = depth[valid]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            
            # Transform to world frame
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
            pts_world = (pose @ pts_cam.T).T[:, :3]
            
            # Subsample to reduce memory
            all_points.append(pts_world[::50])
        
        if all_points:
            return np.vstack(all_points)
        return np.array([]).reshape(0, 3)
    
    def compute_global_bounds(self):
        """Compute global bounds encompassing all environments."""
        if not self.environments:
            raise ValueError("No environments registered")
        
        all_min = np.stack([e.min_bounds for e in self.environments.values()])
        all_max = np.stack([e.max_bounds for e in self.environments.values()])
        
        global_min = all_min.min(axis=0)
        global_max = all_max.max(axis=0)
        center = (global_min + global_max) / 2
        extent = global_max - global_min
        scale = extent.max()
        
        self.global_bounds = EnvironmentBounds(
            name='global',
            min_bounds=global_min,
            max_bounds=global_max,
            center=center,
            extent=extent,
            scale=scale,
        )
        
        print(f"Global bounds computed:")
        print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Extent: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
        print(f"  Scale: {scale:.2f} m")
        
        return self.global_bounds
    
    def normalize_points(
        self,
        points: np.ndarray,
        env_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalize points to canonical coordinates.
        
        Args:
            points: (N, 3) world coordinates
            env_name: Environment name (required for per_environment mode)
            
        Returns:
            (N, 3) normalized coordinates in [-1, 1]
        """
        if self.normalize_mode == 'per_environment':
            if env_name is None:
                raise ValueError("env_name required for per_environment normalization")
            bounds = self.environments[env_name]
        elif self.normalize_mode == 'global':
            if self.global_bounds is None:
                self.compute_global_bounds()
            bounds = self.global_bounds
        elif self.normalize_mode == 'metric':
            # Just center, keep metric scale
            if self.global_bounds is None:
                self.compute_global_bounds()
            return points - self.global_bounds.center
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        
        return bounds.to_normalized(points)
    
    def denormalize_points(
        self,
        points: np.ndarray,
        env_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convert normalized coordinates back to world coordinates.
        
        Args:
            points: (N, 3) normalized coordinates
            env_name: Environment name (required for per_environment mode)
            
        Returns:
            (N, 3) world coordinates
        """
        if self.normalize_mode == 'per_environment':
            if env_name is None:
                raise ValueError("env_name required for per_environment normalization")
            bounds = self.environments[env_name]
        elif self.normalize_mode == 'global':
            bounds = self.global_bounds
        elif self.normalize_mode == 'metric':
            return points + self.global_bounds.center
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")
        
        return bounds.from_normalized(points)
    
    def save(self, path: str):
        """Save coordinate system configuration."""
        data = {
            'normalize_mode': self.normalize_mode,
            'environments': {k: v.to_dict() for k, v in self.environments.items()},
            'global_bounds': self.global_bounds.to_dict() if self.global_bounds else None,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved coordinate system to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'UnifiedCoordinateSystem':
        """Load coordinate system configuration."""
        with open(path) as f:
            data = json.load(f)
        
        cs = cls(normalize_mode=data['normalize_mode'])
        cs.environments = {
            k: EnvironmentBounds.from_dict(v) 
            for k, v in data['environments'].items()
        }
        if data['global_bounds']:
            cs.global_bounds = EnvironmentBounds.from_dict(data['global_bounds'])
        
        print(f"Loaded coordinate system from {path}")
        return cs
    
    def create_for_new_environment(
        self,
        env_name: str,
        depth_images: List[np.ndarray],
        poses: List[np.ndarray],
        intrinsics: np.ndarray,
        depth_scale: float = 0.001,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
    ) -> EnvironmentBounds:
        """
        Create bounds for a new (deployment) environment from observations.
        
        This is used when deploying the trained model to a completely new
        environment that wasn't in the training set.
        
        Args:
            env_name: Name for the new environment
            depth_images: List of depth images
            poses: List of 4x4 camera poses
            intrinsics: 3x3 camera intrinsic matrix
            depth_scale: Scale factor for depth
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            
        Returns:
            EnvironmentBounds for the new environment
        """
        all_points = []
        K = intrinsics
        
        for depth, pose in zip(depth_images, poses):
            depth_m = depth.astype(np.float32) * depth_scale
            valid = (depth_m > min_depth) & (depth_m < max_depth)
            
            if valid.sum() == 0:
                continue
            
            v, u = np.where(valid)
            z = depth_m[valid]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
            pts_world = (pose @ pts_cam.T).T[:, :3]
            all_points.append(pts_world[::100])
        
        all_points = np.vstack(all_points)
        
        margin = 0.5
        min_bounds = all_points.min(axis=0) - margin
        max_bounds = all_points.max(axis=0) + margin
        center = (min_bounds + max_bounds) / 2
        extent = max_bounds - min_bounds
        scale = extent.max()
        
        bounds = EnvironmentBounds(
            name=env_name,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            center=center,
            extent=extent,
            scale=scale,
        )
        
        self.environments[env_name] = bounds
        return bounds


def compute_all_environment_bounds(
    base_data_dir: str = 'data',
    environments: Optional[List[str]] = None,
) -> UnifiedCoordinateSystem:
    """
    Compute bounds for all environments in the dataset.
    
    Args:
        base_data_dir: Base directory containing *_extracted folders
        environments: List of environment names, or None for all
        
    Returns:
        UnifiedCoordinateSystem with all environments registered
    """
    if environments is None:
        environments = ['warehouse', 'hospital', 'office']
    
    cs = UnifiedCoordinateSystem(normalize_mode='per_environment')
    
    for env in environments:
        env_dir = os.path.join(base_data_dir, f'{env}_extracted')
        if not os.path.exists(env_dir):
            print(f"Warning: {env_dir} not found, skipping")
            continue
        
        # Find all sequence directories
        seq_dirs = sorted(glob.glob(os.path.join(env_dir, '*_robot*')))
        seq_dirs = [d for d in seq_dirs if os.path.isdir(d)]
        
        if not seq_dirs:
            print(f"Warning: No sequences found in {env_dir}")
            continue
        
        print(f"\nComputing bounds for {env} ({len(seq_dirs)} sequences)...")
        cs.compute_environment_bounds(env, seq_dirs)
    
    # Compute global bounds
    cs.compute_global_bounds()
    
    return cs


if __name__ == '__main__':
    # Test coordinate system computation
    cs = compute_all_environment_bounds()
    cs.save('output/coordinate_system.json')
    
    # Test point normalization
    print("\nTest normalization:")
    test_point = np.array([[0, 0, 0], [10, 10, 5]])
    for env_name in cs.environments:
        normalized = cs.normalize_points(test_point, env_name)
        recovered = cs.denormalize_points(normalized, env_name)
        print(f"  {env_name}: {test_point[0]} -> {normalized[0]} -> {recovered[0]}")
