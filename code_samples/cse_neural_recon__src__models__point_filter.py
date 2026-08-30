"""
Neural Point Cloud Filtering Network.

Implements iterative point displacement prediction inspired by
3DMambaIPF, using PointNet++ encoder and displacement prediction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer.
    
    Groups points and extracts local features using PointNet.
    
    Args:
        npoint: Number of output points (centroids)
        radius: Ball query radius
        nsample: Number of points per group
        in_channel: Input feature dimension
        mlp: List of output channels for MLP
        group_all: Whether to group all points (for global feature)
    """
    
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False
    ):
        super().__init__()
        
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP for feature extraction
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3  # +3 for relative coordinates
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features (optional)
            
        Returns:
            new_xyz: (B, npoint, 3) sampled point coordinates
            new_features: (B, npoint, C') aggregated features
        """
        B, N, _ = xyz.shape
        
        if self.group_all:
            # Global feature: use all points
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            
            if features is not None:
                grouped_features = features.unsqueeze(1)  # (B, 1, N, C)
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped_features = grouped_xyz
        else:
            # Farthest point sampling
            new_xyz = self._farthest_point_sample(xyz, self.npoint)
            
            # Ball query
            grouped_xyz, grouped_features = self._ball_query_and_group(
                xyz, features, new_xyz
            )
            
        # Relative coordinates
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        
        if grouped_features is not None:
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped_features = grouped_xyz
            
        # (B, npoint, nsample, C) -> (B, C, npoint, nsample)
        grouped_features = grouped_features.permute(0, 3, 1, 2)
        
        # Apply MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_features = F.relu(bn(conv(grouped_features)))
            
        # Max pooling over samples
        new_features = grouped_features.max(dim=-1)[0]  # (B, C, npoint)
        new_features = new_features.permute(0, 2, 1)  # (B, npoint, C)
        
        return new_xyz, new_features
        
    def _farthest_point_sample(
        self,
        xyz: torch.Tensor,
        npoint: int
    ) -> torch.Tensor:
        """
        Farthest point sampling.
        
        Args:
            xyz: (B, N, 3) point coordinates
            npoint: Number of points to sample
            
        Returns:
            centroids: (B, npoint, 3) sampled points
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # Start from random point
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            
            # Get coordinates of farthest point
            centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
            
            # Compute distances to all points
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
            
            # Update distance (keep minimum)
            distance = torch.min(distance, dist)
            
            # Select farthest point
            farthest = distance.argmax(dim=-1)
            
        # Gather sampled coordinates
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, npoint)
        sampled_xyz = xyz[batch_indices, centroids]
        
        return sampled_xyz
        
    def _ball_query_and_group(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor],
        new_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Ball query and group points.
        
        Args:
            xyz: (B, N, 3) all points
            features: (B, N, C) point features
            new_xyz: (B, npoint, 3) query points (centroids)
            
        Returns:
            grouped_xyz: (B, npoint, nsample, 3)
            grouped_features: (B, npoint, nsample, C) or None
        """
        B, N, _ = xyz.shape
        _, npoint, _ = new_xyz.shape
        device = xyz.device
        
        # Compute pairwise distances
        # (B, npoint, 1, 3) - (B, 1, N, 3) -> (B, npoint, N)
        dist = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
        
        # Find points within radius
        group_idx = torch.zeros(B, npoint, self.nsample, dtype=torch.long, device=device)
        
        for b in range(B):
            for i in range(npoint):
                # Get indices of points within radius
                within = (dist[b, i] < self.radius ** 2).nonzero(as_tuple=True)[0]
                
                if len(within) == 0:
                    # No points in radius, use nearest point
                    nearest = dist[b, i].argmin()
                    group_idx[b, i] = nearest
                elif len(within) >= self.nsample:
                    # Sample from points within radius
                    perm = torch.randperm(len(within))[:self.nsample]
                    group_idx[b, i] = within[perm]
                else:
                    # Pad with first point
                    group_idx[b, i, :len(within)] = within
                    group_idx[b, i, len(within):] = within[0]
                    
        # Gather grouped coordinates
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, npoint, self.nsample)
        grouped_xyz = xyz[batch_indices, group_idx]  # (B, npoint, nsample, 3)
        
        if features is not None:
            grouped_features = features[batch_indices, group_idx]
        else:
            grouped_features = None
            
        return grouped_xyz, grouped_features


class PointNetEncoder(nn.Module):
    """
    PointNet++ encoder for point cloud feature extraction.
    
    Args:
        input_dim: Input feature dimension (3 for xyz only)
        hidden_dims: Hidden dimensions for each abstraction layer
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [64, 128, 256]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=input_dim - 3 if input_dim > 3 else 0,
            mlp=[32, 32, hidden_dims[0]]
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=64,
            in_channel=hidden_dims[0],
            mlp=[64, 64, hidden_dims[1]]
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=128,
            in_channel=hidden_dims[1],
            mlp=[128, 128, hidden_dims[2]]
        )
        
        self.output_dim = hidden_dims[2]
        
    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Encode point cloud.
        
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) additional features (optional)
            
        Returns:
            xyz3: (B, 64, 3) final point coordinates
            features3: (B, 64, 256) final features
            skip_connections: List of intermediate features for decoding
        """
        # Abstraction layers
        xyz1, features1 = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)
        
        skip_connections = [
            (xyz, features),
            (xyz1, features1),
            (xyz2, features2)
        ]
        
        return xyz3, features3, skip_connections


class PointFilterNet(nn.Module):
    """
    Neural network for point cloud filtering/denoising.
    
    Predicts displacement vectors to move noisy points toward
    the true surface.
    
    Args:
        hidden_dim: Hidden feature dimension
        num_iterations: Number of refinement iterations during inference
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_iterations: int = 15
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # Point encoder
        self.encoder = PointNetEncoder(
            input_dim=3,
            hidden_dims=[64, 128, hidden_dim]
        )
        
        # Displacement prediction head
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # (dx, dy, dz)
        )
        
        # Feature propagation for upsampling
        self.fp_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        xyz: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict point displacements.
        
        Args:
            xyz: (B, N, 3) noisy point coordinates
            return_features: Whether to return intermediate features
            
        Returns:
            displacement: (B, N, 3) predicted displacement vectors
            features: (B, N, hidden_dim) point features (if requested)
        """
        B, N, _ = xyz.shape
        
        # Encode point cloud
        xyz_down, features_down, skip = self.encoder(xyz)
        
        # Propagate features back to original points
        # Using nearest-neighbor interpolation
        features_full = self._propagate_features(
            xyz, xyz_down, features_down
        )
        
        # Predict displacements
        displacement = self.displacement_head(features_full)
        
        if return_features:
            return displacement, features_full
        return displacement, None
        
    def _propagate_features(
        self,
        xyz_target: torch.Tensor,
        xyz_source: torch.Tensor,
        features_source: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate features from subsampled to original points.
        
        Uses distance-weighted interpolation from k nearest neighbors.
        """
        B, N, _ = xyz_target.shape
        _, M, C = features_source.shape
        
        # Compute distances (B, N, M)
        dist = torch.cdist(xyz_target, xyz_source)
        
        # Find k nearest neighbors
        k = min(3, M)
        dists, indices = dist.topk(k, dim=-1, largest=False)
        
        # Distance-based weights (inverse distance)
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Gather neighbor features
        batch_idx = torch.arange(B, device=xyz_target.device).view(B, 1, 1).expand(-1, N, k)
        neighbor_features = features_source[batch_idx, indices]  # (B, N, k, C)
        
        # Weighted sum
        interpolated = (neighbor_features * weights.unsqueeze(-1)).sum(dim=2)
        
        # Concatenate with coordinates and refine
        combined = torch.cat([interpolated, xyz_target], dim=-1)
        refined = self.fp_mlp(combined)
        
        return refined


class IterativePointFilter(nn.Module):
    """
    Iterative point cloud filtering with learned step sizes.
    
    Applies multiple refinement iterations with decreasing step sizes.
    
    Args:
        filter_net: Base point filter network
        num_iterations: Number of refinement iterations
        initial_step_size: Initial displacement step size
        step_decay: Step size decay factor per iteration
    """
    
    def __init__(
        self,
        filter_net: Optional[PointFilterNet] = None,
        num_iterations: int = 15,
        initial_step_size: float = 0.5,
        step_decay: float = 0.9
    ):
        super().__init__()
        
        self.filter_net = filter_net or PointFilterNet()
        self.num_iterations = num_iterations
        self.initial_step_size = initial_step_size
        self.step_decay = step_decay
        
        # Learnable per-iteration step sizes
        self.step_sizes = nn.Parameter(
            torch.ones(num_iterations) * initial_step_size
        )
        
    def forward(
        self,
        xyz: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Iteratively refine point positions.
        
        Args:
            xyz: (B, N, 3) initial noisy point coordinates
            num_iterations: Override default iteration count
            return_trajectory: Return all intermediate positions
            
        Returns:
            refined: (B, N, 3) or (B, T, N, 3) refined coordinates
        """
        n_iter = num_iterations or self.num_iterations
        
        current = xyz.clone()
        trajectory = [current] if return_trajectory else None
        
        for i in range(n_iter):
            # Predict displacement
            displacement, _ = self.filter_net(current)
            
            # Apply with step size
            step = self.step_sizes[min(i, len(self.step_sizes) - 1)]
            current = current + step * displacement
            
            if return_trajectory:
                trajectory.append(current)
                
        if return_trajectory:
            return torch.stack(trajectory, dim=1)
        return current
        
    def train_step(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        num_unroll: int = 3
    ) -> torch.Tensor:
        """
        Training step with unrolled iterations.
        
        Args:
            noisy: (B, N, 3) noisy point cloud
            clean: (B, N, 3) clean target point cloud
            num_unroll: Number of iterations to unroll for training
            
        Returns:
            loss: Chamfer distance loss
        """
        current = noisy
        total_loss = 0.0
        
        for i in range(num_unroll):
            displacement, _ = self.filter_net(current)
            step = self.step_sizes[min(i, len(self.step_sizes) - 1)]
            current = current + step * displacement
            
            # Compute loss at each step
            loss = chamfer_distance(current, clean)
            total_loss = total_loss + loss
            
        return total_loss / num_unroll


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds.
    
    CD(P, Q) = mean(min_q ||p - q||²) + mean(min_p ||q - p||²)
    
    Args:
        pred: (B, N, 3) predicted point cloud
        target: (B, M, 3) target point cloud
        
    Returns:
        loss: Scalar Chamfer distance
    """
    # Pairwise distances (B, N, M)
    dist = torch.cdist(pred, target, p=2)
    
    # Minimum distance from pred to target
    min_pred_to_target = dist.min(dim=2)[0]  # (B, N)
    
    # Minimum distance from target to pred
    min_target_to_pred = dist.min(dim=1)[0]  # (B, M)
    
    # Chamfer distance
    cd = min_pred_to_target.mean() + min_target_to_pred.mean()
    
    return cd
