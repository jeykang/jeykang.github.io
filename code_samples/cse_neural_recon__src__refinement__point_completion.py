"""
Point cloud completion network.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FoldingNet(nn.Module):
    """
    FoldingNet-style decoder for point generation.
    
    Folds a 2D grid into 3D space using learned transformations.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_points: int = 2048,
        grid_size: int = 45
    ):
        super().__init__()
        
        self.num_points = num_points
        self.grid_size = grid_size
        
        # Create 2D grid
        self.register_buffer(
            'grid',
            self._create_grid(grid_size)
        )
        
        # Folding layers
        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )
        
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )
        
    def _create_grid(self, size: int) -> torch.Tensor:
        """Create 2D grid points."""
        x = torch.linspace(-0.5, 0.5, size)
        y = torch.linspace(-0.5, 0.5, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        return grid
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate points from latent code.
        
        Args:
            latent: Latent code [B, latent_dim]
            
        Returns:
            Generated points [B, num_points, 3]
        """
        B = latent.shape[0]
        
        # Select subset of grid points
        num_grid = self.grid.shape[0]
        if num_grid >= self.num_points:
            indices = torch.randperm(num_grid)[:self.num_points]
            grid = self.grid[indices]
        else:
            # Repeat grid to get enough points
            repeats = (self.num_points // num_grid) + 1
            grid = self.grid.repeat(repeats, 1)[:self.num_points]
            
        # Expand for batch
        grid = grid.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        
        # Expand latent
        latent_expanded = latent.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # First folding
        fold1_input = torch.cat([grid, latent_expanded], dim=-1)
        points1 = self.fold1(fold1_input)  # [B, N, 3]
        
        # Second folding
        fold2_input = torch.cat([points1, latent_expanded], dim=-1)
        points2 = self.fold2(fold2_input)  # [B, N, 3]
        
        return points2


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for extracting global features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 512
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud to latent vector.
        
        Args:
            points: Input points [B, N, 3]
            
        Returns:
            Latent code [B, latent_dim]
        """
        # Per-point features
        features = self.mlp(points)  # [B, N, latent_dim]
        
        # Global max pooling
        latent = features.max(dim=1)[0]  # [B, latent_dim]
        
        return latent


class PointCompletionNetwork(nn.Module):
    """
    Point cloud completion network.
    
    Takes partial point cloud and generates complete point cloud.
    
    Architecture:
    1. Encode partial input with PointNet
    2. Generate coarse output with FoldingNet
    3. Refine with point upsampling
    
    Args:
        latent_dim: Dimension of latent code
        num_coarse: Number of coarse points
        num_fine: Number of fine (output) points
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_coarse: int = 512,
        num_fine: int = 2048
    ):
        super().__init__()
        
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        
        # Encoder
        self.encoder = PointNetEncoder(
            in_channels=3,
            latent_dim=latent_dim
        )
        
        # Coarse generator
        self.coarse_generator = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_coarse * 3)
        )
        
        # Fine generator (local refinement)
        self.fine_generator = nn.Sequential(
            nn.Linear(latent_dim + 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )
        
        # Upsampling ratio
        self.upsample_ratio = num_fine // num_coarse
        
    def forward(
        self,
        partial: torch.Tensor,
        return_coarse: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Complete partial point cloud.
        
        Args:
            partial: Partial input points [B, N, 3]
            return_coarse: If True, also return coarse output
            
        Returns:
            Completed points [B, num_fine, 3]
            Optionally coarse points [B, num_coarse, 3]
        """
        B = partial.shape[0]
        
        # Encode
        latent = self.encoder(partial)  # [B, latent_dim]
        
        # Generate coarse
        coarse = self.coarse_generator(latent)  # [B, num_coarse * 3]
        coarse = coarse.view(B, self.num_coarse, 3)
        
        # Generate fine (upsample coarse)
        # Repeat coarse points
        coarse_expanded = coarse.unsqueeze(2).repeat(
            1, 1, self.upsample_ratio, 1
        ).view(B, self.num_fine, 3)  # [B, num_fine, 3]
        
        # Add noise for diversity
        noise = torch.randn(B, self.num_fine, 3, device=partial.device) * 0.01
        coarse_noisy = coarse_expanded + noise
        
        # Expand latent
        latent_expanded = latent.unsqueeze(1).expand(-1, self.num_fine, -1)
        
        # Refine
        fine_input = torch.cat([coarse_noisy, latent_expanded], dim=-1)
        offsets = self.fine_generator(fine_input)
        
        fine = coarse_noisy + offsets
        
        if return_coarse:
            return fine, coarse
        return fine


class CompletionLoss(nn.Module):
    """
    Loss function for point completion training.
    
    Uses Chamfer distance between predicted and ground truth.
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Weight for coarse loss
        
    def chamfer_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chamfer distance.
        
        Args:
            pred: Predicted points [B, N, 3]
            target: Target points [B, M, 3]
            
        Returns:
            Chamfer distance [B]
        """
        # Pairwise distances
        dist = torch.cdist(pred, target)  # [B, N, M]
        
        # Min distance from pred to target
        min_pred_to_target = dist.min(dim=2)[0].mean(dim=1)  # [B]
        
        # Min distance from target to pred
        min_target_to_pred = dist.min(dim=1)[0].mean(dim=1)  # [B]
        
        return min_pred_to_target + min_target_to_pred
        
    def forward(
        self,
        fine: torch.Tensor,
        coarse: Optional[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute completion loss.
        
        Args:
            fine: Fine predictions [B, N, 3]
            coarse: Coarse predictions [B, M, 3] (optional)
            target: Ground truth [B, K, 3]
            
        Returns:
            Total loss
        """
        # Fine loss
        fine_loss = self.chamfer_distance(fine, target)
        
        if coarse is not None:
            # Coarse loss
            coarse_loss = self.chamfer_distance(coarse, target)
            loss = fine_loss + self.alpha * coarse_loss
        else:
            loss = fine_loss
            
        return loss.mean()


def complete_point_cloud(
    partial: torch.Tensor,
    model: PointCompletionNetwork,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Complete a partial point cloud.
    
    Args:
        partial: Partial input [N, 3] or [B, N, 3]
        model: Trained completion model
        device: Computation device
        
    Returns:
        Completed point cloud
    """
    model = model.to(device)
    model.eval()
    
    # Handle batch dimension
    squeeze = False
    if partial.dim() == 2:
        partial = partial.unsqueeze(0)
        squeeze = True
        
    partial = partial.to(device)
    
    with torch.no_grad():
        complete = model(partial)
        
    if squeeze:
        complete = complete.squeeze(0)
        
    return complete
