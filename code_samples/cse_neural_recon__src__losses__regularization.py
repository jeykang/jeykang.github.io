"""
Regularization losses for neural 3D reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization for SDF values.
    
    Penalizes large second-order derivatives (Laplacian)
    to encourage smooth surfaces.
    
    L_smooth = ||∇²SDF||²
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        model: nn.Module,
        points: torch.Tensor,
        eps: float = 1e-3
    ) -> torch.Tensor:
        """
        Compute smoothness loss via finite differences.
        
        Args:
            model: Neural SDF model
            points: (N, 3) or (B, N, 3) query points
            eps: Step size for finite differences
            
        Returns:
            loss: Smoothness regularization loss
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            
        B, N, _ = points.shape
        device = points.device
        
        # Compute Laplacian via central differences
        # ∇²f ≈ Σ (f(x+ε_i) + f(x-ε_i) - 2f(x)) / ε²
        laplacian = torch.zeros(B, N, device=device)
        
        # Get center value
        center_sdf = model(points)['sdf'].squeeze(-1)  # (B, N)
        
        for dim in range(3):
            # Perturbation in positive direction
            pos_perturb = points.clone()
            pos_perturb[..., dim] += eps
            pos_sdf = model(pos_perturb)['sdf'].squeeze(-1)
            
            # Perturbation in negative direction
            neg_perturb = points.clone()
            neg_perturb[..., dim] -= eps
            neg_sdf = model(neg_perturb)['sdf'].squeeze(-1)
            
            # Second derivative approximation
            laplacian = laplacian + (pos_sdf + neg_sdf - 2 * center_sdf) / (eps ** 2)
            
        loss = laplacian ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SparsityLoss(nn.Module):
    """
    Sparsity regularization for SDF values.
    
    Encourages SDF to be close to 0 or large (not small intermediate values),
    which helps with sharp surface boundaries.
    
    Args:
        threshold: Values below this are penalized
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        
    def forward(self, sdf_values: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Args:
            sdf_values: SDF values to regularize
            
        Returns:
            loss: Sparsity regularization loss
        """
        abs_sdf = torch.abs(sdf_values.squeeze(-1))
        
        # Penalize values in the "uncertain" region
        # Using soft hinge: max(0, threshold - |sdf|)²
        loss = F.relu(self.threshold - abs_sdf) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TVLoss(nn.Module):
    """
    Total Variation loss for spatial smoothness.
    
    L_TV = Σ |f(x+1,y,z) - f(x,y,z)| + |f(x,y+1,z) - f(x,y,z)| + ...
    
    Useful for volumetric representations.
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss on a volume.
        
        Args:
            volume: (B, D, H, W) or (D, H, W) SDF volume
            
        Returns:
            loss: Total variation loss
        """
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
            
        # Differences in each direction
        tv_d = torch.abs(volume[:, 1:, :, :] - volume[:, :-1, :, :])
        tv_h = torch.abs(volume[:, :, 1:, :] - volume[:, :, :-1, :])
        tv_w = torch.abs(volume[:, :, :, 1:] - volume[:, :, :, :-1])
        
        loss = tv_d.mean() + tv_h.mean() + tv_w.mean()
        
        return loss


class ColorConsistencyLoss(nn.Module):
    """
    Color consistency loss for multi-view reconstruction.
    
    Penalizes color variation for the same 3D point seen from
    different viewpoints.
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        predicted_colors: torch.Tensor,
        target_colors: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute color consistency loss.
        
        Args:
            predicted_colors: (N, 3) or (B, N, 3) predicted RGB
            target_colors: (N, 3) or (B, N, 3) target RGB
            weights: Optional per-point weights
            
        Returns:
            loss: Color consistency loss
        """
        loss = F.l1_loss(predicted_colors, target_colors, reduction='none')
        loss = loss.mean(dim=-1)  # Average over RGB channels
        
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DepthConsistencyLoss(nn.Module):
    """
    Depth consistency loss between rendered and observed depth.
    
    Args:
        threshold: Depth difference threshold for robust loss
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        
    def forward(
        self,
        predicted_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth consistency loss.
        
        Args:
            predicted_depth: (H, W) or (B, H, W) predicted depth
            target_depth: (H, W) or (B, H, W) observed depth
            valid_mask: (H, W) or (B, H, W) valid depth mask
            
        Returns:
            loss: Depth consistency loss
        """
        diff = torch.abs(predicted_depth - target_depth)
        
        # Robust loss (Huber-like)
        loss = torch.where(
            diff < self.threshold,
            0.5 * diff ** 2,
            self.threshold * (diff - 0.5 * self.threshold)
        )
        
        if valid_mask is not None:
            loss = loss * valid_mask
            # Normalize by number of valid pixels
            if self.reduction == 'mean':
                return loss.sum() / (valid_mask.sum() + 1e-6)
            return loss.sum()
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightDecayLoss(nn.Module):
    """
    L2 weight decay regularization.
    
    Useful for preventing overfitting in implicit neural representations.
    """
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L2 norm of model weights.
        
        Args:
            model: Neural network model
            
        Returns:
            loss: Sum of squared weights
        """
        total = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total = total + (param ** 2).sum()
        return total
