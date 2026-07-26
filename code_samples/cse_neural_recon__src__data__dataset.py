"""
Enhanced CSE Dataset Loader with Multi-Camera Support.

This module provides dataset classes for loading the CSE 
(Collaborative SLAM in Service Environments) dataset.
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Union
import cv2


def create_multi_sequence_dataset(
    base_dir: str,
    sequences: Optional[List[str]] = None,
    pattern: str = "static_*",
    **kwargs
) -> Dataset:
    """
    Create a combined dataset from multiple sequences.
    
    This allows training on data from multiple robot trajectories
    to get better scene coverage.
    
    Args:
        base_dir: Base directory containing sequence folders
        sequences: List of sequence names, or None to use pattern
        pattern: Glob pattern to find sequences if sequences is None
        **kwargs: Arguments passed to CSEDataset
        
    Returns:
        Combined dataset (ConcatDataset of CSEDatasets)
    """
    if sequences is None:
        # Find sequences matching pattern
        import glob
        seq_dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
        sequences = [os.path.basename(d) for d in seq_dirs if os.path.isdir(d)]
    
    if not sequences:
        raise ValueError(f"No sequences found in {base_dir} matching {pattern}")
    
    datasets = []
    total_frames = 0
    
    for seq_name in sequences:
        seq_path = os.path.join(base_dir, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Warning: Sequence {seq_name} not found, skipping")
            continue
            
        try:
            ds = CSEDataset(run_dir=seq_path, **kwargs)
            datasets.append(ds)
            total_frames += len(ds)
            print(f"  Loaded {seq_name}: {len(ds)} frames")
        except Exception as e:
            print(f"Warning: Failed to load {seq_name}: {e}")
    
    if not datasets:
        raise ValueError(f"No valid sequences found in {base_dir}")
    
    combined = ConcatDataset(datasets)
    print(f"Combined dataset: {total_frames} frames from {len(datasets)} sequences")
    
    return combined


def create_multi_environment_dataset(
    environments: List[Dict],
    compute_unified_coords: bool = True,
    coord_system_path: Optional[str] = None,
    **kwargs
) -> Dataset:
    """
    Create a combined dataset from multiple environments.
    
    Each environment can have multiple sequences (static and dynamic).
    This provides maximum data diversity for robust model training.
    
    When compute_unified_coords=True, computes per-environment bounds
    so that all points can be normalized to [-1, 1] for training.
    This allows the model to learn scale-invariant geometry.
    
    Args:
        environments: List of dicts with 'base_dir' and optional 'sequences' or 'pattern'
            Example: [
                {'base_dir': 'data/warehouse_extracted', 'sequences': ['static_warehouse_robot1']},
                {'base_dir': 'data/hospital_extracted', 'pattern': 'static_*'},
            ]
        compute_unified_coords: Whether to compute unified coordinate system
        coord_system_path: Path to save/load coordinate system JSON
        **kwargs: Arguments passed to CSEDataset
        
    Returns:
        Combined dataset (ConcatDataset across all environments and sequences)
        The dataset has a .coordinate_system attribute if compute_unified_coords=True
    """
    from .coordinate_system import UnifiedCoordinateSystem
    
    all_datasets = []
    env_to_datasets = {}  # Track which datasets belong to which environment
    total_frames = 0
    
    print("=" * 60)
    print("MULTI-ENVIRONMENT DATASET LOADING")
    print("=" * 60)
    
    for env in environments:
        base_dir = env['base_dir']
        sequences = env.get('sequences', None)
        pattern = env.get('pattern', '*')
        
        # Extract environment name from base_dir
        env_name = os.path.basename(base_dir).replace('_extracted', '')
        
        print(f"\nEnvironment: {base_dir} ({env_name})")
        
        if sequences is None:
            # Find sequences matching pattern
            seq_dirs = sorted(glob.glob(os.path.join(base_dir, pattern)))
            sequences = [os.path.basename(d) for d in seq_dirs if os.path.isdir(d)]
        
        if not sequences:
            print(f"  Warning: No sequences found matching pattern '{pattern}'")
            continue
        
        env_frames = 0
        env_datasets = []
        
        for seq_name in sequences:
            seq_path = os.path.join(base_dir, seq_name)
            if not os.path.isdir(seq_path):
                print(f"  Warning: Sequence {seq_name} not found, skipping")
                continue
                
            try:
                ds = CSEDataset(run_dir=seq_path, **kwargs)
                # Store environment name in dataset for coordinate lookup
                ds.environment_name = env_name
                all_datasets.append(ds)
                env_datasets.append(ds)
                env_frames += len(ds)
                total_frames += len(ds)
                print(f"  ✓ {seq_name}: {len(ds)} frames")
            except Exception as e:
                print(f"  ✗ {seq_name}: Failed - {e}")
        
        env_to_datasets[env_name] = env_datasets
        print(f"  Subtotal: {env_frames} frames from {len(sequences)} sequences")
    
    if not all_datasets:
        raise ValueError("No valid datasets found across any environment")
    
    # Compute unified coordinate system if requested
    coord_system = None
    if compute_unified_coords:
        print("\nComputing unified coordinate system...")
        coord_system = UnifiedCoordinateSystem(normalize_mode='per_environment')
        
        # Try to load existing coord system
        if coord_system_path and os.path.exists(coord_system_path):
            coord_system = UnifiedCoordinateSystem.load(coord_system_path)
        else:
            # Compute bounds for each environment
            for env_name, datasets in env_to_datasets.items():
                seq_dirs = [ds.run_dir for ds in datasets]
                coord_system.compute_environment_bounds(env_name, seq_dirs[:3])  # Sample first 3
            
            coord_system.compute_global_bounds()
            
            # Save for later use
            if coord_system_path:
                os.makedirs(os.path.dirname(coord_system_path), exist_ok=True)
                coord_system.save(coord_system_path)
    
    combined = ConcatDataset(all_datasets)
    
    # Attach coordinate system and environment mapping to the combined dataset
    combined.coordinate_system = coord_system
    combined.env_to_datasets = env_to_datasets
    combined.all_datasets = all_datasets
    
    print("=" * 60)
    print(f"TOTAL: {total_frames:,} frames from {len(all_datasets)} sequences")
    if coord_system:
        print(f"Coordinate system: {len(coord_system.environments)} environments")
    print("=" * 60)
    
    return combined


class CSEDataset(Dataset):
    """
    Enhanced CSE Dataset loader for single camera sequences.
    
    Features:
    - Configurable image resolution with proper intrinsic scaling
    - Flexible timestamp synchronization with interpolation
    - On-the-fly depth filtering and preprocessing
    - Memory-efficient frame caching option
    
    Args:
        run_dir: Path to sequence directory (e.g., 'static_warehouse_robot1')
        side: Camera side ('left' or 'right')
        img_wh: Target image resolution (width, height)
        sync_tolerance: Maximum time difference for frame sync (seconds)
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)
        depth_scale: Scale factor for depth images
        interpolate_poses: Whether to interpolate poses to exact timestamps
        cache_frames: Whether to cache loaded frames in memory
        transform: Optional transform to apply to samples
    """
    
    def __init__(
        self,
        run_dir: str,
        side: str = 'left',
        img_wh: Tuple[int, int] = (640, 360),
        sync_tolerance: float = 0.05,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
        depth_scale: float = 1.0,
        interpolate_poses: bool = True,
        cache_frames: bool = False,
        transform: Optional[callable] = None
    ):
        self.run_dir = run_dir
        self.side = side
        self.img_wh = img_wh
        self.sync_tolerance = sync_tolerance
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.interpolate_poses = interpolate_poses
        self.cache_frames = cache_frames
        self.transform = transform
        
        self._cache = {} if cache_frames else None
        
        # Load camera intrinsics
        self._load_intrinsics()
        
        # Load ground truth poses
        self._load_poses()
        
        # Load and synchronize frames
        self._load_frames()
        
        print(f"CSEDataset initialized: {len(self.frames)} frames from {run_dir}")
        
    def _load_intrinsics(self):
        """Load and scale camera intrinsics."""
        intrinsics_path = os.path.join(
            self.run_dir, f"camera_info_{self.side}_intrinsics.json"
        )
        
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
            
        with open(intrinsics_path, 'r') as f:
            meta = json.load(f)
            
        self.K_orig = np.array(meta['K']).reshape(3, 3)
        self.W_orig = meta['width']
        self.H_orig = meta['height']
        self.distortion = np.array(meta.get('D', []))
        self.camera_model = meta.get('model', 'pinhole')
        
        # Scale intrinsics to target resolution
        scale_x = self.img_wh[0] / self.W_orig
        scale_y = self.img_wh[1] / self.H_orig
        
        self.K = self.K_orig.copy()
        self.K[0, :] *= scale_x  # fx, cx
        self.K[1, :] *= scale_y  # fy, cy
        
    def _load_poses(self):
        """Load ground truth poses in TUM format."""
        pose_path = os.path.join(self.run_dir, "ground_truth.txt")
        
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing poses: {pose_path}")
            
        timestamps = []
        positions = []
        quaternions = []
        
        with open(pose_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = [float(x) for x in line.strip().split()]
                if len(parts) < 8:
                    continue
                    
                # Format: timestamp tx ty tz qx qy qz qw
                timestamps.append(parts[0])
                positions.append(parts[1:4])
                quaternions.append(parts[4:8])
                
        self.pose_timestamps = np.array(timestamps)
        self.pose_positions = np.array(positions)
        self.pose_quaternions = np.array(quaternions)
        
        # Create pose matrices
        self.poses = {}
        for i, ts in enumerate(self.pose_timestamps):
            mat = np.eye(4)
            mat[:3, 3] = self.pose_positions[i]
            mat[:3, :3] = R.from_quat(self.pose_quaternions[i]).as_matrix()
            self.poses[ts] = mat
            
        # Setup interpolators for pose interpolation
        if self.interpolate_poses:
            self._setup_pose_interpolation()
            
    def _setup_pose_interpolation(self):
        """Setup interpolators for smooth pose interpolation."""
        # Position interpolation (linear)
        self.pos_interp = interp1d(
            self.pose_timestamps,
            self.pose_positions,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # For quaternion interpolation, we'll use SLERP
        # Store for manual interpolation
        self._quat_times = self.pose_timestamps
        self._quats = self.pose_quaternions
        
    def _interpolate_pose(self, timestamp: float) -> np.ndarray:
        """Interpolate pose at given timestamp using SLERP."""
        # Find bracketing poses
        idx = np.searchsorted(self._quat_times, timestamp)
        
        if idx == 0:
            idx = 1
        elif idx >= len(self._quat_times):
            idx = len(self._quat_times) - 1
            
        t0, t1 = self._quat_times[idx-1], self._quat_times[idx]
        alpha = (timestamp - t0) / (t1 - t0 + 1e-10)
        alpha = np.clip(alpha, 0, 1)
        
        # SLERP for rotation
        q0 = R.from_quat(self._quats[idx-1])
        q1 = R.from_quat(self._quats[idx])
        
        # Use scipy's SLERP
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], R.concatenate([q0, q1]))
        rot = slerp(alpha)
        
        # Linear interpolation for position
        pos = self.pos_interp(timestamp)
        
        # Build pose matrix
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, :3] = rot.as_matrix()
        
        return mat
        
    def _load_frames(self):
        """Load and synchronize RGB and depth frames."""
        # Get RGB files
        rgb_folder = os.path.join(self.run_dir, f"rgb_{self.side}_compressed")
        depth_folder = os.path.join(self.run_dir, f"depth_{self.side}")
        
        self.rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        self.depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
        
        if not self.rgb_files:
            raise FileNotFoundError(f"No RGB images found in {rgb_folder}")
            
        # Build depth timestamp lookup
        self.depth_map = {}
        for d_path in self.depth_files:
            # Filename is nanoseconds (e.g., 1933333434.png)
            ts = float(os.path.basename(d_path).replace('.png', '')) / 1e9
            self.depth_map[ts] = d_path
        self.depth_timestamps = np.array(sorted(self.depth_map.keys()))
        
        # Synchronize frames
        self.frames = []
        
        for rgb_path in self.rgb_files:
            ts_rgb = float(os.path.basename(rgb_path).replace('.png', '')) / 1e9
            
            # Find nearest pose
            idx_p = np.searchsorted(self.pose_timestamps, ts_rgb)
            idx_p = np.clip(idx_p, 0, len(self.pose_timestamps) - 1)
            ts_pose = self.pose_timestamps[idx_p]
            
            # Find nearest depth
            if len(self.depth_timestamps) > 0:
                idx_d = np.searchsorted(self.depth_timestamps, ts_rgb)
                idx_d = np.clip(idx_d, 0, len(self.depth_timestamps) - 1)
                ts_depth = self.depth_timestamps[idx_d]
                depth_diff = abs(ts_rgb - ts_depth)
            else:
                depth_diff = float('inf')
                ts_depth = None
                
            pose_diff = abs(ts_rgb - ts_pose)
            
            # Check synchronization tolerance
            if pose_diff < self.sync_tolerance:
                frame_data = {
                    'rgb': rgb_path,
                    'timestamp': ts_rgb,
                }
                
                # Add depth if available and synchronized
                if ts_depth is not None and depth_diff < self.sync_tolerance:
                    frame_data['depth'] = self.depth_map[ts_depth]
                    
                # Add pose (interpolated or nearest)
                if self.interpolate_poses:
                    frame_data['pose'] = self._interpolate_pose(ts_rgb)
                else:
                    frame_data['pose'] = self.poses[ts_pose]
                    
                self.frames.append(frame_data)
                
    def __len__(self) -> int:
        return len(self.frames)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_frames and idx in self._cache:
            sample = self._cache[idx]
        else:
            sample = self._load_sample(idx)
            if self.cache_frames:
                self._cache[idx] = sample
                
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample."""
        frame = self.frames[idx]
        
        # Load RGB
        rgb = Image.open(frame['rgb']).convert('RGB')
        rgb = rgb.resize(self.img_wh, Image.BILINEAR)
        rgb = torch.from_numpy(np.array(rgb)).float() / 255.0
        rgb = rgb.permute(2, 0, 1)  # HWC -> CHW
        
        sample = {
            'rgb': rgb,
            'pose': torch.from_numpy(frame['pose']).float(),
            'K': torch.from_numpy(self.K).float(),
            'timestamp': torch.tensor(frame['timestamp']),
            'idx': torch.tensor(idx),
        }
        
        # Load depth if available
        if 'depth' in frame:
            depth = Image.open(frame['depth'])
            depth = depth.resize(self.img_wh, Image.NEAREST)
            # Convert to int32 first (uint16 not supported by all PyTorch versions)
            depth_arr = np.array(depth).astype(np.int32)
            depth = torch.from_numpy(depth_arr).float()
            
            # Apply depth scale
            depth = depth * self.depth_scale
            
            # Create valid mask
            valid_mask = (depth > self.min_depth) & (depth < self.max_depth)
            
            sample['depth'] = depth
            sample['valid_mask'] = valid_mask
            
        return sample
        
    def get_all_poses(self) -> np.ndarray:
        """Get all poses as (N, 4, 4) array."""
        return np.stack([f['pose'] for f in self.frames], axis=0)
        
    def get_bounds(self, margin: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scene bounds from camera positions."""
        positions = np.stack([f['pose'][:3, 3] for f in self.frames], axis=0)
        
        min_bounds = positions.min(axis=0) - margin
        max_bounds = positions.max(axis=0) + margin
        
        # Ensure reasonable z bounds (assuming z is up)
        min_bounds[2] = max(min_bounds[2], -1.0)
        max_bounds[2] = min(max_bounds[2], 5.0)
        
        return min_bounds, max_bounds


class CSEMultiCameraDataset(Dataset):
    """
    Multi-camera CSE Dataset loader for 6-camera rigs.
    
    Loads synchronized frames from multiple cameras with proper
    extrinsic transformations.
    
    Args:
        run_dir: Path to sequence directory
        cameras: List of camera configurations (side, extrinsic, etc.)
        img_wh: Target image resolution
        sync_tolerance: Maximum time difference for frame sync
        **kwargs: Additional arguments passed to CSEDataset
    """
    
    def __init__(
        self,
        run_dir: str,
        cameras: Optional[List[Dict]] = None,
        img_wh: Tuple[int, int] = (640, 360),
        sync_tolerance: float = 0.05,
        **kwargs
    ):
        self.run_dir = run_dir
        self.img_wh = img_wh
        self.sync_tolerance = sync_tolerance
        
        # Default to stereo pair if no cameras specified
        if cameras is None:
            cameras = [
                {'side': 'left', 'extrinsic': np.eye(4)},
                {'side': 'right', 'extrinsic': self._get_stereo_extrinsic()},
            ]
            
        self.cameras = cameras
        
        # Load individual camera datasets
        self.camera_datasets = {}
        for cam in cameras:
            self.camera_datasets[cam['side']] = CSEDataset(
                run_dir=run_dir,
                side=cam['side'],
                img_wh=img_wh,
                sync_tolerance=sync_tolerance,
                **kwargs
            )
            
        # Find common timestamps across all cameras
        self._synchronize_cameras()
        
        print(f"CSEMultiCameraDataset: {len(self.frames)} synchronized frames "
              f"from {len(cameras)} cameras")
              
    def _get_stereo_extrinsic(self) -> np.ndarray:
        """Get default stereo baseline extrinsic (approximate)."""
        # Typical stereo baseline ~0.12m
        extrinsic = np.eye(4)
        extrinsic[0, 3] = 0.12  # Baseline in x direction
        return extrinsic
        
    def _synchronize_cameras(self):
        """Find frames synchronized across all cameras."""
        # Get timestamps from first camera
        base_dataset = list(self.camera_datasets.values())[0]
        base_timestamps = [f['timestamp'] for f in base_dataset.frames]
        
        self.frames = []
        
        for ts in base_timestamps:
            frame_data = {'timestamp': ts, 'cameras': {}}
            all_synced = True
            
            for cam in self.cameras:
                dataset = self.camera_datasets[cam['side']]
                
                # Find nearest frame in this camera
                cam_timestamps = [f['timestamp'] for f in dataset.frames]
                idx = np.argmin(np.abs(np.array(cam_timestamps) - ts))
                
                if abs(cam_timestamps[idx] - ts) < self.sync_tolerance:
                    frame_data['cameras'][cam['side']] = {
                        'idx': idx,
                        'extrinsic': cam['extrinsic']
                    }
                else:
                    all_synced = False
                    break
                    
            if all_synced:
                self.frames.append(frame_data)
                
    def __len__(self) -> int:
        return len(self.frames)
        
    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get synchronized samples from all cameras.
        
        Returns:
            Dictionary with camera names as keys, each containing
            rgb, depth, pose, K tensors.
        """
        frame = self.frames[idx]
        samples = {}
        
        for cam_side, cam_data in frame['cameras'].items():
            sample = self.camera_datasets[cam_side][cam_data['idx']]
            
            # Apply extrinsic transformation to pose
            extrinsic = torch.from_numpy(cam_data['extrinsic']).float()
            sample['pose'] = sample['pose'] @ extrinsic
            
            samples[cam_side] = sample
            
        samples['timestamp'] = torch.tensor(frame['timestamp'])
        
        return samples
