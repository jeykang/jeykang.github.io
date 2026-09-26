"""
Planar attention module for incorporating structural priors.

Implements cross-attention between point features and detected
plane features to enforce flatness in planar regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class PlanarAttention(nn.Module):
    """
    Cross-attention module for planar feature integration.
    
    Allows point features to attend to detected plane features,
    incorporating structural priors into the representation.
    
    Args:
        embed_dim: Feature embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_planes: Maximum number of planes to attend to
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_planes: int = 20
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_planes = max_planes
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Learnable position encoding for planes
        self.plane_pos_encoding = nn.Parameter(
            torch.randn(1, max_planes, embed_dim) * 0.02
        )
        
    def forward(
        self,
        point_features: torch.Tensor,
        plane_features: torch.Tensor,
        plane_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply planar attention.
        
        Args:
            point_features: (B, N, D) or (N, D) point features
            plane_features: (B, P, D) or (P, D) plane features
            plane_mask: (B, P) or (P,) valid plane mask
            
        Returns:
            enhanced_features: (B, N, D) point features enhanced with planar info
        """
        # Handle unbatched input
        if point_features.dim() == 2:
            point_features = point_features.unsqueeze(0)
            plane_features = plane_features.unsqueeze(0)
            if plane_mask is not None:
                plane_mask = plane_mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, N, D = point_features.shape
        P = plane_features.shape[1]
        
        # Add positional encoding to planes
        pos_enc = self.plane_pos_encoding[:, :P, :]
        plane_features = plane_features + pos_enc
        
        # Create attention mask (True = ignore)
        if plane_mask is not None:
            # Invert mask: plane_mask=True means valid, attention needs True=ignore
            attn_mask = ~plane_mask
        else:
            attn_mask = None
            
        # Cross-attention: points attend to planes
        # query: points, key/value: planes
        attended, attn_weights = self.attention(
            query=point_features,
            key=plane_features,
            value=plane_features,
            key_padding_mask=attn_mask,
            need_weights=True
        )
        
        # Project and residual
        output = self.output_proj(attended)
        enhanced = point_features + output
        
        if squeeze_output:
            enhanced = enhanced.squeeze(0)
            
        return enhanced


class PlaneFeatureExtractor(nn.Module):
    """
    Extract features from detected planes.
    
    Converts plane parameters (normal, offset, boundary) into
    learnable feature representations.
    
    Args:
        embed_dim: Output feature dimension
        use_boundary: Whether to use plane boundary information
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        use_boundary: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_boundary = use_boundary
        
        # Plane parameter encoding
        # Input: normal (3) + offset (1) + optional boundary encoding
        input_dim = 4  # normal + offset
        
        self.plane_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Boundary encoding (if used)
        if use_boundary:
            self.boundary_encoder = nn.Sequential(
                nn.Linear(4, embed_dim // 4),  # bbox: x_min, x_max, y_min, y_max
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim // 2)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim + embed_dim // 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            
    def forward(
        self,
        normals: torch.Tensor,
        offsets: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract plane features.
        
        Args:
            normals: (B, P, 3) plane normals
            offsets: (B, P, 1) plane offsets (distance from origin)
            boundaries: (B, P, 4) plane bounding boxes (optional)
            
        Returns:
            features: (B, P, D) plane feature vectors
        """
        # Concatenate plane parameters
        plane_params = torch.cat([normals, offsets], dim=-1)  # (B, P, 4)
        
        # Encode plane parameters
        features = self.plane_encoder(plane_params)  # (B, P, D)
        
        # Add boundary information
        if self.use_boundary and boundaries is not None:
            boundary_features = self.boundary_encoder(boundaries)
            features = self.fusion(torch.cat([features, boundary_features], dim=-1))
            
        return features


class ManhattanConstraint(nn.Module):
    """
    Manhattan World constraint module.
    
    Encourages detected planes to align with one of three
    orthogonal axis directions.
    
    Args:
        soft_threshold: Threshold for soft assignment
    """
    
    def __init__(self, soft_threshold: float = 0.1):
        super().__init__()
        self.soft_threshold = soft_threshold
        
        # Fixed Manhattan axes (x, y, z)
        axes = torch.eye(3)
        self.register_buffer('manhattan_axes', axes)
        
    def forward(self, normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Manhattan constraint loss and axis assignments.
        
        Args:
            normals: (B, P, 3) plane normals (should be unit vectors)
            
        Returns:
            loss: Scalar Manhattan constraint loss
            assignments: (B, P) axis assignments (0, 1, or 2)
        """
        # Compute alignment with each axis
        # |n · axis| should be close to 1 for aligned planes
        alignments = torch.abs(torch.einsum('bpd,ad->bpa', normals, self.manhattan_axes))
        
        # Best alignment for each plane
        max_alignment, assignments = alignments.max(dim=-1)
        
        # Loss: 1 - max_alignment (want alignment close to 1)
        loss = (1.0 - max_alignment).mean()
        
        return loss, assignments
        
    def project_to_manhattan(self, normals: torch.Tensor) -> torch.Tensor:
        """
        Project normals to nearest Manhattan axis.
        
        Args:
            normals: (B, P, 3) plane normals
            
        Returns:
            projected: (B, P, 3) axis-aligned normals
        """
        alignments = torch.abs(torch.einsum('bpd,ad->bpa', normals, self.manhattan_axes))
        _, assignments = alignments.max(dim=-1)
        
        # Get sign of alignment
        signed_alignments = torch.einsum('bpd,ad->bpa', normals, self.manhattan_axes)
        signs = torch.sign(signed_alignments.gather(-1, assignments.unsqueeze(-1)))
        
        # Project to axis
        projected = self.manhattan_axes[assignments] * signs
        
        return projected


def compute_point_to_plane_distance(
    points: torch.Tensor,
    plane_normals: torch.Tensor,
    plane_offsets: torch.Tensor
) -> torch.Tensor:
    """
    Compute signed distance from points to planes.
    
    distance = n · p + d
    
    Args:
        points: (B, N, 3) or (N, 3) 3D points
        plane_normals: (B, P, 3) or (P, 3) plane normals (unit vectors)
        plane_offsets: (B, P) or (P,) plane offsets
        
    Returns:
        distances: (B, N, P) or (N, P) signed distances
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        plane_normals = plane_normals.unsqueeze(0)
        plane_offsets = plane_offsets.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
        
    # n · p: (B, N, P)
    dot_product = torch.einsum('bnd,bpd->bnp', points, plane_normals)
    
    # Add offset
    distances = dot_product + plane_offsets.unsqueeze(1)
    
    if squeeze:
        distances = distances.squeeze(0)
        
    return distances


def assign_points_to_planes(
    points: torch.Tensor,
    plane_normals: torch.Tensor,
    plane_offsets: torch.Tensor,
    threshold: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assign points to nearest plane (if within threshold).
    
    Args:
        points: (B, N, 3) 3D points
        plane_normals: (B, P, 3) plane normals
        plane_offsets: (B, P) plane offsets
        threshold: Maximum distance for assignment
        
    Returns:
        assignments: (B, N) plane index for each point (-1 if unassigned)
        distances: (B, N) distance to assigned plane
    """
    distances = compute_point_to_plane_distance(points, plane_normals, plane_offsets)
    abs_distances = torch.abs(distances)
    
    # Find nearest plane
    min_distances, nearest_planes = abs_distances.min(dim=-1)
    
    # Only assign if within threshold
    valid = min_distances < threshold
    assignments = torch.where(valid, nearest_planes, torch.full_like(nearest_planes, -1))
    
    return assignments, min_distances
