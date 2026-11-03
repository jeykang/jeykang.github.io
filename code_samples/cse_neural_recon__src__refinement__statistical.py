"""
Statistical point cloud refinement methods.
"""

from typing import Optional, Tuple
import numpy as np


class StatisticalOutlierRemoval:
    """
    Remove statistical outliers from point cloud.
    
    Points are considered outliers if their average distance to
    k nearest neighbors is more than n standard deviations from
    the mean.
    
    Args:
        k_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation multiplier for threshold
    """
    
    def __init__(self, k_neighbors: int = 20, std_ratio: float = 2.0):
        self.k_neighbors = k_neighbors
        self.std_ratio = std_ratio
        
    def filter(
        self,
        points: np.ndarray,
        return_mask: bool = False
    ) -> np.ndarray:
        """
        Filter outliers from point cloud.
        
        Args:
            points: Input points [N, 3]
            return_mask: If True, also return boolean mask
            
        Returns:
            Filtered points [M, 3], optionally with mask
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            # Fallback: return original
            if return_mask:
                return points, np.ones(len(points), dtype=bool)
            return points
            
        n_points = len(points)
        if n_points <= self.k_neighbors:
            if return_mask:
                return points, np.ones(n_points, dtype=bool)
            return points
            
        # Find k nearest neighbors
        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        
        # Average distance to neighbors (excluding self)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # Compute threshold
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + self.std_ratio * global_std
        
        # Filter
        mask = mean_distances < threshold
        
        if return_mask:
            return points[mask], mask
        return points[mask]


class RadiusOutlierRemoval:
    """
    Remove outliers based on radius neighbor count.
    
    Points with fewer than min_neighbors within radius are removed.
    
    Args:
        radius: Search radius
        min_neighbors: Minimum number of neighbors required
    """
    
    def __init__(self, radius: float = 0.05, min_neighbors: int = 5):
        self.radius = radius
        self.min_neighbors = min_neighbors
        
    def filter(
        self,
        points: np.ndarray,
        return_mask: bool = False
    ) -> np.ndarray:
        """
        Filter outliers from point cloud.
        
        Args:
            points: Input points [N, 3]
            return_mask: If True, also return boolean mask
            
        Returns:
            Filtered points [M, 3], optionally with mask
        """
        try:
            from sklearn.neighbors import BallTree
        except ImportError:
            if return_mask:
                return points, np.ones(len(points), dtype=bool)
            return points
            
        # Build ball tree
        tree = BallTree(points)
        
        # Count neighbors within radius
        counts = tree.query_radius(points, r=self.radius, count_only=True)
        
        # Filter (count includes self, so add 1)
        mask = counts >= (self.min_neighbors + 1)
        
        if return_mask:
            return points[mask], mask
        return points[mask]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.01,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Input points [N, 3]
        voxel_size: Size of voxel grid
        normals: Optional normals [N, 3]
        colors: Optional colors [N, 3] or [N, 4]
        
    Returns:
        Downsampled points and optionally normals/colors
    """
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Create unique key for each voxel
    min_idx = voxel_indices.min(axis=0)
    voxel_indices = voxel_indices - min_idx
    
    max_idx = voxel_indices.max(axis=0) + 1
    keys = (voxel_indices[:, 0] * max_idx[1] * max_idx[2] +
            voxel_indices[:, 1] * max_idx[2] +
            voxel_indices[:, 2])
    
    # Find unique voxels
    unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)
    
    # Average points in each voxel
    downsampled_points = np.zeros((n_voxels, 3))
    counts = np.zeros(n_voxels)
    
    np.add.at(downsampled_points, inverse_indices, points)
    np.add.at(counts, inverse_indices, 1)
    
    downsampled_points /= counts[:, None]
    
    result = [downsampled_points]
    
    # Average normals if provided
    if normals is not None:
        downsampled_normals = np.zeros((n_voxels, 3))
        np.add.at(downsampled_normals, inverse_indices, normals)
        # Normalize
        norms = np.linalg.norm(downsampled_normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        downsampled_normals /= norms
        result.append(downsampled_normals)
        
    # Average colors if provided
    if colors is not None:
        downsampled_colors = np.zeros((n_voxels, colors.shape[1]))
        np.add.at(downsampled_colors, inverse_indices, colors)
        downsampled_colors /= counts[:, None]
        result.append(downsampled_colors)
        
    if len(result) == 1:
        return downsampled_points
    return tuple(result)


def estimate_normals(
    points: np.ndarray,
    k_neighbors: int = 30,
    orient_to_viewpoint: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Estimate normals for point cloud using PCA.
    
    Args:
        points: Input points [N, 3]
        k_neighbors: Number of neighbors for PCA
        orient_to_viewpoint: Optional viewpoint for normal orientation
        
    Returns:
        Estimated normals [N, 3]
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        # Return placeholder normals pointing up
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0
        return normals
        
    n_points = len(points)
    k = min(k_neighbors, n_points - 1)
    
    # Find neighbors
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(points)
    _, indices = nn.kneighbors(points)
    
    normals = np.zeros_like(points)
    
    for i in range(n_points):
        # Get neighborhood
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        
        # PCA
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Normal is eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]
        
        # Orient consistently
        if orient_to_viewpoint is not None:
            view_dir = orient_to_viewpoint - points[i]
            if np.dot(normal, view_dir) < 0:
                normal = -normal
                
        normals[i] = normal
        
    return normals


def smooth_point_cloud(
    points: np.ndarray,
    k_neighbors: int = 10,
    iterations: int = 1,
    lambda_smooth: float = 0.5
) -> np.ndarray:
    """
    Smooth point cloud using Laplacian smoothing.
    
    Args:
        points: Input points [N, 3]
        k_neighbors: Number of neighbors for smoothing
        iterations: Number of smoothing iterations
        lambda_smooth: Smoothing factor (0-1)
        
    Returns:
        Smoothed points [N, 3]
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return points
        
    n_points = len(points)
    k = min(k_neighbors, n_points - 1)
    
    smoothed = points.copy()
    
    for _ in range(iterations):
        # Find neighbors
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(smoothed)
        _, indices = nn.kneighbors(smoothed)
        
        # Compute Laplacian
        laplacian = np.zeros_like(smoothed)
        for i in range(n_points):
            neighbors = smoothed[indices[i, 1:]]  # Exclude self
            laplacian[i] = neighbors.mean(axis=0) - smoothed[i]
            
        # Apply smoothing
        smoothed = smoothed + lambda_smooth * laplacian
        
    return smoothed


def remove_isolated_clusters(
    points: np.ndarray,
    min_cluster_size: int = 100,
    eps: float = 0.05
) -> np.ndarray:
    """
    Remove small isolated clusters from point cloud.
    
    Args:
        points: Input points [N, 3]
        min_cluster_size: Minimum points in a cluster to keep
        eps: DBSCAN epsilon parameter
        
    Returns:
        Points from large clusters only
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return points
        
    # Cluster points
    clustering = DBSCAN(eps=eps, min_samples=5).fit(points)
    labels = clustering.labels_
    
    # Count cluster sizes
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    
    # Find large clusters
    large_clusters = unique_labels[counts >= min_cluster_size]
    
    # Keep points from large clusters
    mask = np.isin(labels, large_clusters)
    
    return points[mask]
