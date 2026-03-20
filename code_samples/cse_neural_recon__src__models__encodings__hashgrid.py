"""
Multi-resolution Hash Grid Encoding for neural implicit representations.

Based on Instant-NGP (Müller et al., 2022).
Provides fast, memory-efficient encoding using hash tables.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class HashGridEncoding(nn.Module):
    """
    Multi-resolution hash grid encoding.
    
    Encodes 3D coordinates using multiple resolution levels,
    each stored in a hash table for memory efficiency.
    
    Args:
        num_levels: Number of resolution levels
        base_resolution: Resolution of coarsest level
        max_resolution: Resolution of finest level
        features_per_level: Feature dimension at each level
        log2_hashmap_size: Log2 of hash table size
        input_dim: Input coordinate dimension (2 or 3)
    """
    
    def __init__(
        self,
        num_levels: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        input_dim: int = 3
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        self.input_dim = input_dim
        
        # Output dimension
        self.output_dim = num_levels * features_per_level
        
        # Compute resolution at each level (geometric progression)
        # N_l = floor(N_min * b^l) where b = exp(ln(N_max/N_min) / (L-1))
        if num_levels > 1:
            growth = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1))
        else:
            growth = 1.0
            
        resolutions = [int(np.floor(base_resolution * (growth ** l))) for l in range(num_levels)]
        self.register_buffer('resolutions', torch.tensor(resolutions))
        
        # Initialize hash tables (one per level)
        # Each entry is a feature vector
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, features_per_level) * 0.001)
            for _ in range(num_levels)
        ])
        
        # Prime numbers for spatial hashing
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using hash grid.
        
        Args:
            x: (*, input_dim) coordinates in [0, 1]
            
        Returns:
            features: (*, output_dim) encoded features
        """
        original_shape = x.shape[:-1]
        x = x.view(-1, self.input_dim)  # (N, input_dim)
        
        all_features = []
        
        for level in range(self.num_levels):
            resolution = self.resolutions[level].item()
            features = self._interpolate_level(x, level, resolution)
            all_features.append(features)
            
        # Concatenate all levels
        output = torch.cat(all_features, dim=-1)  # (N, output_dim)
        
        # Reshape to original batch dimensions
        output = output.view(*original_shape, self.output_dim)
        
        return output
        
    def _interpolate_level(
        self,
        x: torch.Tensor,
        level: int,
        resolution: int
    ) -> torch.Tensor:
        """
        Trilinear interpolation at a single resolution level.
        
        Args:
            x: (N, 3) coordinates in [0, 1]
            level: Resolution level index
            resolution: Grid resolution at this level
            
        Returns:
            features: (N, features_per_level) interpolated features
        """
        N = x.shape[0]
        device = x.device
        
        # Scale coordinates to grid vertices.
        # We treat `resolution` as the number of grid vertices per axis, so valid
        # integer indices are in [0, resolution - 1]. Mapping [0, 1] -> [0, resolution - 1]
        # avoids producing an out-of-range `ceil` index (which previously caused
        # excessive hash collisions near the boundary and can lead to early
        # representation collapse).
        x_scaled = x * (resolution - 1)
        
        # Get corner indices
        x_floor = torch.floor(x_scaled).long()
        x_ceil = x_floor + 1
        
        # Clamp to valid range
        x_floor = torch.clamp(x_floor, 0, resolution - 1)
        x_ceil = torch.clamp(x_ceil, 0, resolution - 1)
        
        # Interpolation weights
        w = x_scaled - x_floor.float()  # (N, 3)
        
        # Get all 8 corner features for 3D (or 4 for 2D)
        if self.input_dim == 3:
            corners = self._get_corners_3d(x_floor, x_ceil)  # (N, 8, 3)
            weights = self._get_weights_3d(w)  # (N, 8)
        else:
            corners = self._get_corners_2d(x_floor, x_ceil)  # (N, 4, 2)
            weights = self._get_weights_2d(w)  # (N, 4)
            
        # Hash corner indices
        hash_indices = self._hash(corners, level)  # (N, num_corners)
        
        # Lookup features
        hash_table = self.hash_tables[level]
        corner_features = hash_table[hash_indices]  # (N, num_corners, features)
        
        # Weighted sum
        features = (corner_features * weights.unsqueeze(-1)).sum(dim=1)  # (N, features)
        
        return features
        
    def _get_corners_3d(
        self,
        floor_idx: torch.Tensor,
        ceil_idx: torch.Tensor
    ) -> torch.Tensor:
        """Get 8 corner indices for trilinear interpolation."""
        # floor_idx, ceil_idx: (N, 3)
        N = floor_idx.shape[0]
        
        # All 8 combinations of floor/ceil for each dimension
        corners = torch.stack([
            torch.stack([floor_idx[:, 0], floor_idx[:, 1], floor_idx[:, 2]], dim=1),
            torch.stack([ceil_idx[:, 0], floor_idx[:, 1], floor_idx[:, 2]], dim=1),
            torch.stack([floor_idx[:, 0], ceil_idx[:, 1], floor_idx[:, 2]], dim=1),
            torch.stack([ceil_idx[:, 0], ceil_idx[:, 1], floor_idx[:, 2]], dim=1),
            torch.stack([floor_idx[:, 0], floor_idx[:, 1], ceil_idx[:, 2]], dim=1),
            torch.stack([ceil_idx[:, 0], floor_idx[:, 1], ceil_idx[:, 2]], dim=1),
            torch.stack([floor_idx[:, 0], ceil_idx[:, 1], ceil_idx[:, 2]], dim=1),
            torch.stack([ceil_idx[:, 0], ceil_idx[:, 1], ceil_idx[:, 2]], dim=1),
        ], dim=1)  # (N, 8, 3)
        
        return corners
        
    def _get_weights_3d(self, w: torch.Tensor) -> torch.Tensor:
        """Get trilinear interpolation weights."""
        # w: (N, 3) - fractional parts
        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
        
        weights = torch.stack([
            (1 - wx) * (1 - wy) * (1 - wz),
            wx * (1 - wy) * (1 - wz),
            (1 - wx) * wy * (1 - wz),
            wx * wy * (1 - wz),
            (1 - wx) * (1 - wy) * wz,
            wx * (1 - wy) * wz,
            (1 - wx) * wy * wz,
            wx * wy * wz,
        ], dim=1)  # (N, 8)
        
        return weights
        
    def _get_corners_2d(
        self,
        floor_idx: torch.Tensor,
        ceil_idx: torch.Tensor
    ) -> torch.Tensor:
        """Get 4 corner indices for bilinear interpolation."""
        corners = torch.stack([
            torch.stack([floor_idx[:, 0], floor_idx[:, 1]], dim=1),
            torch.stack([ceil_idx[:, 0], floor_idx[:, 1]], dim=1),
            torch.stack([floor_idx[:, 0], ceil_idx[:, 1]], dim=1),
            torch.stack([ceil_idx[:, 0], ceil_idx[:, 1]], dim=1),
        ], dim=1)  # (N, 4, 2)
        
        return corners
        
    def _get_weights_2d(self, w: torch.Tensor) -> torch.Tensor:
        """Get bilinear interpolation weights."""
        wx, wy = w[:, 0], w[:, 1]
        
        weights = torch.stack([
            (1 - wx) * (1 - wy),
            wx * (1 - wy),
            (1 - wx) * wy,
            wx * wy,
        ], dim=1)  # (N, 4)
        
        return weights
        
    def _hash(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """
        Spatial hash function.
        
        h(x) = (⊕_i (x_i * π_i)) mod T
        
        Args:
            indices: (N, num_corners, dim) integer grid indices
            level: Resolution level (unused, for potential level-specific hashing)
            
        Returns:
            hash_indices: (N, num_corners) indices into hash table
        """
        # XOR-based spatial hashing (Instant-NGP style).
        primes = self.primes[:self.input_dim]
        
        # indices: (N, num_corners, dim)
        # primes: (dim,)
        hashed = torch.zeros(indices.shape[:-1], device=indices.device, dtype=torch.long)
        for d in range(self.input_dim):
            hashed ^= (indices[..., d].long() * primes[d].long())
        
        # Modulo hash table size
        hash_indices = hashed % self.hashmap_size
        
        return hash_indices.long()
        
    def get_output_dim(self) -> int:
        """Return output dimension of encoding."""
        return self.output_dim


class MultiResolutionHashEncoding(HashGridEncoding):
    """
    Alias for HashGridEncoding with commonly used defaults.
    """
    
    def __init__(
        self,
        bounding_box: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store bounding box for coordinate normalization
        if bounding_box is not None:
            self.register_buffer('bbox_min', bounding_box[0])
            self.register_buffer('bbox_max', bounding_box[1])
        else:
            self.bbox_min = None
            self.bbox_max = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with automatic normalization if bounding box is set.
        """
        if self.bbox_min is not None:
            # Normalize to [0, 1]
            x = (x - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-6)
            x = torch.clamp(x, 0, 1)
            
        return super().forward(x)
        
    def set_bounding_box(
        self,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor
    ):
        """Set bounding box for coordinate normalization."""
        self.register_buffer('bbox_min', bbox_min)
        self.register_buffer('bbox_max', bbox_max)
