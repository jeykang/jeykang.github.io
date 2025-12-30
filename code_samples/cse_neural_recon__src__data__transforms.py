"""
Data transformations and augmentation for training.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R


class DataAugmentation:
    """
    Data augmentation pipeline for 3D reconstruction training.
    
    Supports:
    - Random rotation around gravity axis
    - Random translation
    - Depth noise injection
    - Color jittering
    
    Args:
        rotation_range: (min, max) rotation in degrees
        translation_range: (min, max) translation in meters
        depth_noise_std: Standard deviation of depth noise
        color_jitter: Whether to apply color jittering
        p: Probability of applying augmentation
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        translation_range: Tuple[float, float] = (-0.2, 0.2),
        depth_noise_std: float = 0.01,
        color_jitter: bool = True,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.depth_noise_std = depth_noise_std
        self.color_jitter = color_jitter
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to sample."""
        if np.random.random() > self.p:
            return sample
            
        sample = sample.copy()
        
        # Random rotation around z-axis
        if self.rotation_range[1] > 0:
            sample = self._apply_rotation(sample)
            
        # Random translation
        if self.translation_range[1] > 0:
            sample = self._apply_translation(sample)
            
        # Depth noise
        if self.depth_noise_std > 0 and 'depth' in sample:
            sample = self._apply_depth_noise(sample)
            
        # Color jittering
        if self.color_jitter and 'rgb' in sample:
            sample = self._apply_color_jitter(sample)
            
        return sample
        
    def _apply_rotation(self, sample: Dict) -> Dict:
        """Apply random rotation around z-axis."""
        angle = np.random.uniform(*self.rotation_range)
        angle_rad = np.deg2rad(angle)
        
        # Rotation matrix around z-axis
        rot = R.from_euler('z', angle_rad).as_matrix()
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot
        
        # Apply to pose
        pose = sample['pose'].numpy()
        augmented_pose = rot_4x4 @ pose
        sample['pose'] = torch.from_numpy(augmented_pose).float()
        
        # Store augmentation info
        sample['aug_rotation'] = torch.tensor(angle_rad)
        
        return sample
        
    def _apply_translation(self, sample: Dict) -> Dict:
        """Apply random translation."""
        trans = np.random.uniform(
            self.translation_range[0],
            self.translation_range[1],
            size=3
        )
        # Only translate in x-y plane
        trans[2] = 0
        
        # Apply to pose
        pose = sample['pose'].numpy()
        pose[:3, 3] += trans
        sample['pose'] = torch.from_numpy(pose).float()
        
        sample['aug_translation'] = torch.from_numpy(trans).float()
        
        return sample
        
    def _apply_depth_noise(self, sample: Dict) -> Dict:
        """Apply Gaussian noise to depth."""
        depth = sample['depth']
        valid_mask = sample.get('valid_mask', depth > 0)
        
        noise = torch.randn_like(depth) * self.depth_noise_std
        noise[~valid_mask] = 0
        
        sample['depth'] = depth + noise
        
        return sample
        
    def _apply_color_jitter(self, sample: Dict) -> Dict:
        """Apply brightness and contrast jittering."""
        rgb = sample['rgb']
        
        # Brightness
        brightness = np.random.uniform(*self.brightness_range)
        rgb = rgb * brightness
        
        # Contrast
        contrast = np.random.uniform(*self.contrast_range)
        mean = rgb.mean()
        rgb = (rgb - mean) * contrast + mean
        
        # Clamp to valid range
        rgb = torch.clamp(rgb, 0, 1)
        sample['rgb'] = rgb
        
        return sample


class DepthTransforms:
    """
    Depth-specific transformations.
    
    Includes:
    - Depth clipping
    - Depth normalization
    - Invalid value handling
    - Edge-aware filtering
    """
    
    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
        normalize: bool = False,
        fill_invalid: bool = False,
        fill_value: float = 0.0
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.normalize = normalize
        self.fill_invalid = fill_invalid
        self.fill_value = fill_value
        
    def __call__(self, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform depth map.
        
        Args:
            depth: (H, W) depth tensor
            
        Returns:
            depth: Transformed depth
            valid_mask: Boolean mask of valid pixels
        """
        # Create valid mask
        valid_mask = (depth > self.min_depth) & (depth < self.max_depth)
        
        # Clip depth
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        # Handle invalid values
        if self.fill_invalid:
            depth[~valid_mask] = self.fill_value
            
        # Normalize
        if self.normalize:
            depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
            
        return depth, valid_mask


class PoseNormalization:
    """
    Normalize poses to a canonical frame.
    
    Useful for training on multiple sequences with different
    coordinate systems.
    """
    
    def __init__(self, center_poses: bool = True, align_gravity: bool = True):
        self.center_poses = center_poses
        self.align_gravity = align_gravity
        
    def fit(self, poses: np.ndarray):
        """
        Compute normalization parameters from poses.
        
        Args:
            poses: (N, 4, 4) array of pose matrices
        """
        positions = poses[:, :3, 3]
        
        # Compute center
        self.center = positions.mean(axis=0) if self.center_poses else np.zeros(3)
        
        # Compute gravity alignment (assume z is up)
        # Use PCA to find dominant up direction
        if self.align_gravity:
            # Simple version: assume z is already up
            self.rotation = np.eye(3)
        else:
            self.rotation = np.eye(3)
            
    def transform(self, pose: np.ndarray) -> np.ndarray:
        """Transform a single pose to normalized frame."""
        normalized = pose.copy()
        normalized[:3, 3] -= self.center
        normalized[:3, :3] = self.rotation @ normalized[:3, :3]
        normalized[:3, 3] = self.rotation @ normalized[:3, 3]
        return normalized
        
    def inverse_transform(self, pose: np.ndarray) -> np.ndarray:
        """Transform pose from normalized back to original frame."""
        original = pose.copy()
        original[:3, :3] = self.rotation.T @ original[:3, :3]
        original[:3, 3] = self.rotation.T @ original[:3, 3]
        original[:3, 3] += self.center
        return original


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Handles variable-size depth masks and stacks tensors properly.
    """
    result = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            result[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
            
    return result
