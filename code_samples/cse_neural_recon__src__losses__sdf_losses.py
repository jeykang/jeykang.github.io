"""
SDF-related loss functions for neural implicit surface learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SurfaceLoss(nn.Module):
    """
    Surface constraint loss: SDF should be zero at surface points.
    
    L_surface = |SDF(p_surface)|
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        sdf_values: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute surface loss.
        
        Args:
            sdf_values: (N,) or (B, N) SDF values at surface points
            weights: Optional per-point weights
            
        Returns:
            loss: Surface constraint loss
        """
        loss = torch.abs(sdf_values)
        
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FreespaceLoss(nn.Module):
    """
    Freespace constraint loss: SDF should equal ray distance for freespace points.
    
    L_freespace = |SDF(p_free) - d|
    
    where d is the distance along the ray from p_free to the surface.
    
    Args:
        reduction: 'mean' or 'sum'
        truncation: Maximum distance to consider
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        truncation: float = 1.0
    ):
        super().__init__()
        self.reduction = reduction
        self.truncation = truncation
        
    def forward(
        self,
        sdf_values: torch.Tensor,
        target_distances: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute freespace loss.
        
        Args:
            sdf_values: (N,) or (B, N) SDF values at freespace points
            target_distances: (N,) or (B, N) target SDF values (ray distance)
            weights: Optional per-point weights
            
        Returns:
            loss: Freespace constraint loss
        """
        # Truncate targets
        target_distances = torch.clamp(target_distances, 0, self.truncation)
        
        loss = torch.abs(sdf_values.squeeze(-1) - target_distances)
        
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EikonalLoss(nn.Module):
    """
    Eikonal regularization: gradient magnitude should be 1.
    
    L_eikonal = (||∇SDF|| - 1)²
    
    This ensures the SDF is a valid signed distance function.
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Compute Eikonal loss.
        
        Args:
            gradients: (N, 3) or (B, N, 3) SDF gradients
            
        Returns:
            loss: Eikonal regularization loss
        """
        gradient_norm = torch.norm(gradients, dim=-1)
        loss = (gradient_norm - 1.0) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SDFLoss(nn.Module):
    """
    Combined SDF loss with all components.
    
    L = λ_surf * L_surface + λ_free * L_freespace + λ_eik * L_eikonal
    
    Args:
        surface_weight: Weight for surface loss
        freespace_weight: Weight for freespace loss
        eikonal_weight: Weight for Eikonal loss
        truncation: Truncation distance for freespace
    """
    
    def __init__(
        self,
        surface_weight: float = 1.0,
        freespace_weight: float = 0.5,
        eikonal_weight: float = 0.1,
        truncation: float = 1.0
    ):
        super().__init__()
        
        self.surface_weight = surface_weight
        self.freespace_weight = freespace_weight
        self.eikonal_weight = eikonal_weight
        
        self.surface_loss = SurfaceLoss()
        self.freespace_loss = FreespaceLoss(truncation=truncation)
        self.eikonal_loss = EikonalLoss()
        
    def forward(
        self,
        sdf_surface: torch.Tensor,
        sdf_freespace: Optional[torch.Tensor] = None,
        freespace_distances: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined SDF loss.
        
        Args:
            sdf_surface: SDF values at surface points
            sdf_freespace: SDF values at freespace points
            freespace_distances: Target distances for freespace points
            gradients: SDF gradients for Eikonal loss
            return_components: Whether to return individual loss components
            
        Returns:
            loss: Combined loss (and components dict if requested)
        """
        components = {}
        total_loss = 0.0
        
        # Surface loss
        l_surface = self.surface_loss(sdf_surface)
        components['surface'] = l_surface
        total_loss = total_loss + self.surface_weight * l_surface
        
        # Freespace loss
        if sdf_freespace is not None and freespace_distances is not None:
            l_freespace = self.freespace_loss(sdf_freespace, freespace_distances)
            components['freespace'] = l_freespace
            total_loss = total_loss + self.freespace_weight * l_freespace
            
        # Eikonal loss
        if gradients is not None:
            l_eikonal = self.eikonal_loss(gradients)
            components['eikonal'] = l_eikonal
            total_loss = total_loss + self.eikonal_weight * l_eikonal
            
        if return_components:
            return total_loss, components
        return total_loss


class BehindSurfaceLoss(nn.Module):
    """
    Loss for points behind the surface (negative SDF region).
    
    L_behind = max(0, SDF(p_behind) + margin)
    
    Points behind the surface should have negative SDF.
    
    Args:
        margin: Margin for the hinge loss
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        margin: float = 0.01,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(
        self,
        sdf_values: torch.Tensor,
        target_distances: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute behind-surface loss.
        
        Args:
            sdf_values: SDF values at points behind surface
            target_distances: Optional target negative distances
            
        Returns:
            loss: Behind-surface constraint loss
        """
        if target_distances is not None:
            # Target is negative distance
            loss = torch.abs(sdf_values.squeeze(-1) - target_distances)
        else:
            # Just enforce negativity
            loss = F.relu(sdf_values.squeeze(-1) + self.margin)
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_sdf_gradients(
    model: nn.Module,
    points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SDF values and gradients.
    
    Args:
        model: Neural SDF model with forward(coords) method
        points: (N, 3) or (B, N, 3) query points
        
    Returns:
        sdf: SDF values
        gradients: SDF gradients (∇SDF)
    """
    points = points.requires_grad_(True)
    
    outputs = model(points)
    sdf = outputs['sdf'] if isinstance(outputs, dict) else outputs[0]
    
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=points,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True
    )[0]
    
    return sdf, gradients


def sample_freespace_points(
    surface_points: torch.Tensor,
    camera_centers: torch.Tensor,
    jitter_range: Tuple[float, float] = (0.05, 0.5)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample freespace points between camera and surface.
    
    Args:
        surface_points: (N, 3) points on surface
        camera_centers: (N, 3) or (3,) camera center(s)
        jitter_range: (min, max) distance to move toward camera
        
    Returns:
        freespace_points: (N, 3) sampled points in freespace
        distances: (N,) distance from freespace point to surface
    """
    if camera_centers.dim() == 1:
        camera_centers = camera_centers.unsqueeze(0).expand(surface_points.shape[0], -1)
        
    # Ray direction (from surface to camera)
    ray_dirs = camera_centers - surface_points
    ray_lengths = torch.norm(ray_dirs, dim=-1, keepdim=True)
    ray_dirs = ray_dirs / (ray_lengths + 1e-8)
    
    # Random jitter distance
    jitter_min, jitter_max = jitter_range
    jitter = torch.rand(surface_points.shape[0], 1, device=surface_points.device)
    jitter = jitter * (jitter_max - jitter_min) + jitter_min
    
    # Clamp to not go past camera
    jitter = torch.min(jitter, ray_lengths * 0.9)
    
    # Sample freespace points
    freespace_points = surface_points + ray_dirs * jitter
    
    # Distance to surface (the SDF target)
    distances = jitter.squeeze(-1)
    
    return freespace_points, distances


def sample_behind_surface_points(
    surface_points: torch.Tensor,
    normals: torch.Tensor,
    jitter_range: Tuple[float, float] = (0.01, 0.1)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points behind the surface (inside the object).
    
    Args:
        surface_points: (N, 3) points on surface
        normals: (N, 3) surface normals (pointing outward)
        jitter_range: (min, max) distance to move behind surface
        
    Returns:
        behind_points: (N, 3) points behind surface
        distances: (N,) negative distances (SDF targets)
    """
    # Normalize normals
    normals = F.normalize(normals, dim=-1)
    
    # Random jitter distance
    jitter_min, jitter_max = jitter_range
    jitter = torch.rand(surface_points.shape[0], 1, device=surface_points.device)
    jitter = jitter * (jitter_max - jitter_min) + jitter_min
    
    # Move in negative normal direction (behind surface)
    behind_points = surface_points - normals * jitter
    
    # Distance is negative (inside the surface)
    distances = -jitter.squeeze(-1)
    
    return behind_points, distances
