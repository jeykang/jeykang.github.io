"""
Depth processing utilities including filtering and hole filling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import cv2


class DepthProcessor:
    """
    Depth map processing pipeline.
    
    Provides filtering, edge preservation, and quality enhancement
    for depth maps from RGB-D sensors.
    
    Args:
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)
        bilateral_filter: Whether to apply bilateral filtering
        bilateral_d: Diameter of bilateral filter
        bilateral_sigma_color: Filter sigma in color space
        bilateral_sigma_space: Filter sigma in coordinate space
        median_filter: Whether to apply median filtering
        median_size: Kernel size for median filter
    """
    
    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
        bilateral_filter: bool = True,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50,
        bilateral_sigma_space: float = 50,
        median_filter: bool = True,
        median_size: int = 3
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bilateral_filter = bilateral_filter
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.median_filter = median_filter
        self.median_size = median_size
        
    def __call__(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process depth map.
        
        Args:
            depth: (H, W) depth map
            rgb: Optional (H, W, 3) RGB image for guided filtering
            
        Returns:
            processed_depth: Filtered depth map
            valid_mask: Boolean mask of valid pixels
        """
        # Create valid mask
        valid_mask = (depth > self.min_depth) & (depth < self.max_depth)
        
        # Make a copy for processing
        processed = depth.copy()
        
        # Set invalid values to 0
        processed[~valid_mask] = 0
        
        # Apply median filter to remove outliers
        if self.median_filter:
            processed = cv2.medianBlur(
                processed.astype(np.float32),
                self.median_size
            )
            
        # Apply bilateral filter for edge-preserving smoothing
        if self.bilateral_filter:
            if rgb is not None:
                # Joint bilateral filter using RGB guidance
                processed = self._joint_bilateral_filter(processed, rgb)
            else:
                processed = cv2.bilateralFilter(
                    processed.astype(np.float32),
                    self.bilateral_d,
                    self.bilateral_sigma_color,
                    self.bilateral_sigma_space
                )
                
        # Restore invalid regions
        processed[~valid_mask] = 0
        
        return processed, valid_mask
        
    def _joint_bilateral_filter(
        self,
        depth: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """Apply joint bilateral filter using RGB guidance."""
        # Convert RGB to grayscale for guidance
        if rgb.ndim == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
            
        # OpenCV's joint bilateral filter
        # Using ximgproc if available, otherwise fall back to standard bilateral
        try:
            import cv2.ximgproc as xip
            filtered = xip.jointBilateralFilter(
                gray.astype(np.float32),
                depth.astype(np.float32),
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
        except (ImportError, AttributeError):
            # Fall back to standard bilateral
            filtered = cv2.bilateralFilter(
                depth.astype(np.float32),
                self.bilateral_d,
                self.bilateral_sigma_color,
                self.bilateral_sigma_space
            )
            
        return filtered


class HoleFiller:
    """
    Fill holes in depth maps using various methods.
    
    Methods:
    - nearest: Nearest-neighbor interpolation
    - inpaint: OpenCV inpainting (Navier-Stokes or Telea)
    - plane: Plane fitting for large planar regions
    
    Args:
        method: Hole filling method ('nearest', 'inpaint', 'plane')
        inpaint_radius: Radius for inpainting method
        min_hole_size: Minimum hole size to fill
    """
    
    def __init__(
        self,
        method: str = 'inpaint',
        inpaint_radius: int = 5,
        min_hole_size: int = 10
    ):
        self.method = method
        self.inpaint_radius = inpaint_radius
        self.min_hole_size = min_hole_size
        
    def __call__(
        self,
        depth: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fill holes in depth map.
        
        Args:
            depth: (H, W) depth map with holes (0 or invalid values)
            valid_mask: (H, W) boolean mask of valid pixels
            
        Returns:
            filled_depth: Depth map with holes filled
        """
        if valid_mask.all():
            return depth
            
        hole_mask = ~valid_mask
        
        if self.method == 'nearest':
            return self._fill_nearest(depth, hole_mask)
        elif self.method == 'inpaint':
            return self._fill_inpaint(depth, hole_mask)
        elif self.method == 'plane':
            return self._fill_plane(depth, hole_mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
    def _fill_nearest(
        self,
        depth: np.ndarray,
        hole_mask: np.ndarray
    ) -> np.ndarray:
        """Fill using nearest neighbor interpolation."""
        from scipy.ndimage import distance_transform_edt
        
        filled = depth.copy()
        
        # Get distance to nearest valid pixel and indices
        dist, indices = distance_transform_edt(
            hole_mask,
            return_indices=True
        )
        
        # Fill holes with nearest valid values
        filled[hole_mask] = depth[indices[0, hole_mask], indices[1, hole_mask]]
        
        return filled
        
    def _fill_inpaint(
        self,
        depth: np.ndarray,
        hole_mask: np.ndarray
    ) -> np.ndarray:
        """Fill using OpenCV inpainting."""
        # Normalize depth for inpainting
        valid_depth = depth[~hole_mask]
        if len(valid_depth) == 0:
            return depth
            
        depth_min, depth_max = valid_depth.min(), valid_depth.max()
        depth_range = depth_max - depth_min + 1e-6
        
        normalized = ((depth - depth_min) / depth_range * 255).astype(np.uint8)
        mask = hole_mask.astype(np.uint8) * 255
        
        # Inpaint using Telea method
        inpainted = cv2.inpaint(
            normalized,
            mask,
            self.inpaint_radius,
            cv2.INPAINT_TELEA
        )
        
        # Convert back to original scale
        filled = inpainted.astype(np.float32) / 255 * depth_range + depth_min
        
        # Only fill holes, keep original valid values
        result = depth.copy()
        result[hole_mask] = filled[hole_mask]
        
        return result
        
    def _fill_plane(
        self,
        depth: np.ndarray,
        hole_mask: np.ndarray
    ) -> np.ndarray:
        """Fill holes by fitting planes to surrounding regions."""
        filled = depth.copy()
        
        # Find connected components of holes
        num_labels, labels = cv2.connectedComponents(
            hole_mask.astype(np.uint8)
        )
        
        for label in range(1, num_labels):
            component_mask = labels == label
            
            # Skip small holes
            if component_mask.sum() < self.min_hole_size:
                continue
                
            # Get surrounding valid pixels
            dilated = cv2.dilate(
                component_mask.astype(np.uint8),
                np.ones((5, 5), np.uint8),
                iterations=2
            )
            border_mask = dilated.astype(bool) & ~component_mask & ~hole_mask
            
            if border_mask.sum() < 10:
                # Not enough border pixels, use nearest neighbor
                continue
                
            # Fit plane to border pixels
            y_coords, x_coords = np.where(border_mask)
            z_coords = depth[border_mask]
            
            # Least squares plane fitting: z = ax + by + c
            A = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z_coords, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Fill hole with plane values
            y_hole, x_hole = np.where(component_mask)
            z_plane = coeffs[0] * x_hole + coeffs[1] * y_hole + coeffs[2]
            
            filled[component_mask] = z_plane
            
        return filled


def compute_depth_edges(
    depth: torch.Tensor,
    threshold: float = 0.1
) -> torch.Tensor:
    """
    Compute depth discontinuity edges.
    
    Args:
        depth: (B, H, W) or (H, W) depth tensor
        threshold: Discontinuity threshold (meters)
        
    Returns:
        edges: Boolean tensor marking depth edges
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        
    # Sobel gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    
    depth_4d = depth.unsqueeze(1)  # (B, 1, H, W)
    
    grad_x = F.conv2d(depth_4d, sobel_x, padding=1)
    grad_y = F.conv2d(depth_4d, sobel_y, padding=1)
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    edges = gradient_magnitude.squeeze(1) > threshold
    
    return edges


def depth_to_disparity(
    depth: torch.Tensor,
    baseline: float,
    focal_length: float
) -> torch.Tensor:
    """
    Convert depth to disparity.
    
    disparity = baseline * focal_length / depth
    
    Args:
        depth: Depth tensor
        baseline: Stereo baseline (meters)
        focal_length: Focal length (pixels)
        
    Returns:
        disparity: Disparity tensor
    """
    disparity = baseline * focal_length / (depth + 1e-6)
    return disparity


def disparity_to_depth(
    disparity: torch.Tensor,
    baseline: float,
    focal_length: float
) -> torch.Tensor:
    """
    Convert disparity to depth.
    
    depth = baseline * focal_length / disparity
    
    Args:
        disparity: Disparity tensor
        baseline: Stereo baseline (meters)
        focal_length: Focal length (pixels)
        
    Returns:
        depth: Depth tensor
    """
    depth = baseline * focal_length / (disparity + 1e-6)
    return depth
