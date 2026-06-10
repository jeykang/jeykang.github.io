"""
Planar consistency and Manhattan world losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PlanarConsistencyLoss(nn.Module):
    """
    Planar consistency loss: enforce flatness in detected plane regions.
    
    For points assigned to a plane, penalize deviation from the plane.
    
    L_planar = Σ |n·p + d|² for points in plane region
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        points: torch.Tensor,
        plane_normals: torch.Tensor,
        plane_offsets: torch.Tensor,
        plane_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute planar consistency loss.
        
        Args:
            points: (B, N, 3) point coordinates
            plane_normals: (B, P, 3) plane normals
            plane_offsets: (B, P) plane offsets
            plane_assignments: (B, N) plane index for each point (-1 = not on plane)
            
        Returns:
            loss: Planar consistency loss
        """
        B, N, _ = points.shape
        P = plane_normals.shape[1]
        device = points.device
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            for p in range(P):
                # Get points assigned to this plane
                mask = plane_assignments[b] == p
                if not mask.any():
                    continue
                    
                pts = points[b, mask]  # (M, 3)
                normal = plane_normals[b, p]  # (3,)
                offset = plane_offsets[b, p]  # scalar
                
                # Point-to-plane distance
                dist = torch.abs(torch.sum(pts * normal, dim=-1) + offset)
                total_loss = total_loss + dist.sum()
                count += mask.sum().item()
                
        if count > 0:
            return total_loss / count
        return total_loss


class NormalAlignmentLoss(nn.Module):
    """
    Normal alignment loss: SDF gradient should align with estimated surface normal.
    
    L_normal = 1 - (∇SDF · n̂)² for surface points
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        sdf_gradients: torch.Tensor,
        target_normals: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute normal alignment loss.
        
        Args:
            sdf_gradients: (N, 3) or (B, N, 3) SDF gradients
            target_normals: (N, 3) or (B, N, 3) target surface normals
            weights: Optional per-point weights
            
        Returns:
            loss: Normal alignment loss
        """
        # Normalize
        grad_normalized = F.normalize(sdf_gradients, dim=-1)
        normal_normalized = F.normalize(target_normals, dim=-1)
        
        # Cosine similarity (should be ±1 for aligned)
        cos_sim = torch.sum(grad_normalized * normal_normalized, dim=-1)
        
        # Loss: 1 - cos² (allows for opposite directions)
        loss = 1.0 - cos_sim ** 2
        
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ManhattanLoss(nn.Module):
    """
    Manhattan world constraint: planes should align with principal axes.
    
    L_manhattan = min_axis (1 - |n · axis|²)
    
    Encourages detected plane normals to be axis-aligned.
    
    Args:
        reduction: 'mean' or 'sum'
        soft: Use soft assignment instead of hard
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        soft: bool = True,
        temperature: float = 0.1
    ):
        super().__init__()
        self.reduction = reduction
        self.soft = soft
        self.temperature = temperature
        
        # Manhattan axes (x, y, z)
        axes = torch.eye(3)
        self.register_buffer('axes', axes)
        
    def forward(
        self,
        normals: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Manhattan constraint loss.
        
        Args:
            normals: (N, 3) or (B, N, 3) plane/surface normals
            weights: Optional per-normal weights
            
        Returns:
            loss: Manhattan alignment loss
        """
        # Ensure 3D tensor
        if normals.dim() == 2:
            normals = normals.unsqueeze(0)
            
        B, N, _ = normals.shape
        
        # Normalize normals
        normals = F.normalize(normals, dim=-1)
        
        # Compute alignment with each axis
        # |n · axis| should be close to 1 for aligned normals
        # (B, N, 3) x (3, 3)^T -> (B, N, 3)
        alignments = torch.abs(torch.matmul(normals, self.axes.T))
        
        if self.soft:
            # Soft assignment via softmax
            alignment_weights = F.softmax(alignments / self.temperature, dim=-1)
            # Weighted sum of alignment errors
            errors = 1.0 - alignments ** 2
            loss = (alignment_weights * errors).sum(dim=-1)
        else:
            # Hard assignment: use best alignment
            best_alignment = alignments.max(dim=-1)[0]
            loss = 1.0 - best_alignment ** 2
            
        if weights is not None:
            loss = loss * weights
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class OrthogonalPlanesLoss(nn.Module):
    """
    Loss to encourage detected planes to be mutually orthogonal.
    
    For a valid Manhattan world, walls should be perpendicular
    to floor/ceiling.
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self,
        normals: torch.Tensor,
        semantic_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between planes.
        
        Args:
            normals: (P, 3) or (B, P, 3) plane normals
            semantic_labels: Optional (P,) labels to identify wall/floor/ceiling
            
        Returns:
            loss: Orthogonality constraint loss
        """
        if normals.dim() == 2:
            normals = normals.unsqueeze(0)
            
        B, P, _ = normals.shape
        
        if P < 2:
            return torch.tensor(0.0, device=normals.device)
            
        # Normalize
        normals = F.normalize(normals, dim=-1)
        
        # Pairwise dot products: (B, P, P)
        dot_products = torch.bmm(normals, normals.transpose(1, 2))
        
        # For orthogonal planes, dot product should be 0 or ±1
        # Loss: penalize intermediate values
        # cos(θ)² * (1 - cos(θ)²) is 0 when θ = 0, 90°, or 180°
        loss_matrix = dot_products ** 2 * (1 - dot_products ** 2)
        
        # Only count off-diagonal elements
        mask = ~torch.eye(P, dtype=torch.bool, device=normals.device).unsqueeze(0)
        losses = loss_matrix[mask.expand(B, -1, -1)]
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        return losses


def detect_planes_ransac(
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    distance_threshold: float = 0.02,
    num_iterations: int = 1000,
    min_points: int = 100,
    max_planes: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Detect planes in point cloud using RANSAC.
    
    Args:
        points: (N, 3) point cloud
        normals: Optional (N, 3) point normals for guided sampling
        distance_threshold: Inlier distance threshold
        num_iterations: RANSAC iterations per plane
        min_points: Minimum points to form a plane
        max_planes: Maximum number of planes to detect
        
    Returns:
        plane_normals: (P, 3) detected plane normals
        plane_offsets: (P,) plane offsets
        point_assignments: (N,) plane index for each point (-1 = not on plane)
    """
    device = points.device
    N = points.shape[0]
    
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    
    plane_normals = []
    plane_offsets = []
    point_assignments = torch.full((N,), -1, dtype=torch.long, device=device)
    
    for plane_idx in range(max_planes):
        remaining_points = points[remaining_mask]
        remaining_indices = torch.where(remaining_mask)[0]
        
        if len(remaining_points) < min_points:
            break
            
        # RANSAC for one plane
        best_inliers = None
        best_count = 0
        best_normal = None
        best_offset = None
        
        for _ in range(num_iterations):
            # Sample 3 random points
            idx = torch.randperm(len(remaining_points))[:3]
            p1, p2, p3 = remaining_points[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = torch.cross(v1, v2)
            
            if torch.norm(normal) < 1e-6:
                continue
                
            normal = F.normalize(normal, dim=0)
            offset = -torch.dot(normal, p1)
            
            # Count inliers
            distances = torch.abs(
                torch.sum(remaining_points * normal, dim=1) + offset
            )
            inliers = distances < distance_threshold
            count = inliers.sum().item()
            
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_normal = normal
                best_offset = offset
                
        if best_count < min_points:
            break
            
        # Refine plane using all inliers (least squares)
        inlier_points = remaining_points[best_inliers]
        centroid = inlier_points.mean(dim=0)
        centered = inlier_points - centroid
        
        # SVD for plane fitting
        _, _, Vh = torch.linalg.svd(centered)
        refined_normal = Vh[-1]  # Smallest singular vector
        refined_offset = -torch.dot(refined_normal, centroid)
        
        # Store plane
        plane_normals.append(refined_normal)
        plane_offsets.append(refined_offset)
        
        # Assign points
        global_inliers = remaining_indices[best_inliers]
        point_assignments[global_inliers] = plane_idx
        remaining_mask[global_inliers] = False
        
    if len(plane_normals) == 0:
        return (
            torch.empty(0, 3, device=device),
            torch.empty(0, device=device),
            point_assignments
        )
        
    plane_normals = torch.stack(plane_normals)
    plane_offsets = torch.stack(plane_offsets)
    
    return plane_normals, plane_offsets, point_assignments
