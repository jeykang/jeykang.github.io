"""
Point cloud samplers for training.
"""

from typing import Optional, Union, Tuple
import numpy as np


class PointSampler:
    """
    Base class for point sampling strategies.
    
    Handles sampling points from point clouds for training neural SDFs.
    """
    
    def __init__(self, num_points: int = 4096):
        self.num_points = num_points
        
    def sample(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Sample points from point cloud.
        
        Args:
            points: Input points [N, 3]
            
        Returns:
            Sampled points [num_points, 3]
        """
        raise NotImplementedError
        
        
class UniformSampler(PointSampler):
    """
    Uniform random sampling from point cloud.
    """
    
    def sample(
        self,
        points: np.ndarray,
        replace: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Uniformly sample points.
        
        Args:
            points: Input points [N, 3]
            replace: Whether to sample with replacement
            
        Returns:
            Sampled points [num_points, 3]
        """
        n_points = len(points)
        
        if n_points >= self.num_points and not replace:
            indices = np.random.choice(n_points, self.num_points, replace=False)
        else:
            indices = np.random.choice(n_points, self.num_points, replace=True)
            
        return points[indices]


class FarthestPointSampler(PointSampler):
    """
    Farthest Point Sampling (FPS) for uniform coverage.
    
    Iteratively selects points that are farthest from already selected points.
    """
    
    def sample(
        self,
        points: np.ndarray,
        start_idx: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample using farthest point sampling.
        
        Args:
            points: Input points [N, 3]
            start_idx: Starting point index (random if None)
            
        Returns:
            Sampled points [num_points, 3]
        """
        n_points = len(points)
        
        if n_points <= self.num_points:
            # Pad with random repeats
            if n_points == self.num_points:
                return points.copy()
            indices = np.concatenate([
                np.arange(n_points),
                np.random.choice(n_points, self.num_points - n_points)
            ])
            return points[indices]
            
        # Initialize
        selected = np.zeros(self.num_points, dtype=np.int64)
        distances = np.full(n_points, np.inf)
        
        # Start with random point
        if start_idx is None:
            start_idx = np.random.randint(n_points)
        selected[0] = start_idx
        
        # Iteratively select farthest points
        for i in range(1, self.num_points):
            # Update distances
            last_point = points[selected[i-1]]
            dist_to_last = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_last)
            
            # Select farthest
            selected[i] = np.argmax(distances)
            distances[selected[i]] = -1  # Mark as selected
            
        return points[selected]


class ImportanceSampler(PointSampler):
    """
    Importance sampling based on local geometry.
    
    Samples more points from regions with high curvature or 
    geometric detail.
    """
    
    def __init__(
        self,
        num_points: int = 4096,
        k_neighbors: int = 16,
        curvature_weight: float = 0.5
    ):
        super().__init__(num_points)
        self.k_neighbors = k_neighbors
        self.curvature_weight = curvature_weight
        
    def sample(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample with importance weighting.
        
        Args:
            points: Input points [N, 3]
            normals: Optional normals [N, 3]
            
        Returns:
            Sampled points [num_points, 3]
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            # Fall back to uniform sampling
            return UniformSampler(self.num_points).sample(points)
            
        n_points = len(points)
        
        if n_points <= self.num_points:
            indices = np.concatenate([
                np.arange(n_points),
                np.random.choice(n_points, self.num_points - n_points)
            ])
            return points[indices]
            
        # Compute local curvature/complexity
        k = min(self.k_neighbors, n_points - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        
        # Use local density variation as importance
        local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
        
        # If normals provided, also consider normal variation
        if normals is not None:
            _, indices_k = nn.kneighbors(points)
            neighbor_normals = normals[indices_k[:, 1:]]
            normal_var = np.var(neighbor_normals, axis=1).sum(axis=1)
            importance = (1 - self.curvature_weight) * local_density + \
                         self.curvature_weight * normal_var
        else:
            importance = local_density
            
        # Normalize to probability distribution
        importance = importance / importance.sum()
        
        # Sample according to importance
        indices = np.random.choice(
            n_points,
            self.num_points,
            replace=False,
            p=importance
        )
        
        return points[indices]


class HybridSampler(PointSampler):
    """
    Hybrid sampling combining multiple strategies.
    
    Uses a mix of uniform, FPS, and importance sampling.
    """
    
    def __init__(
        self,
        num_points: int = 4096,
        fps_ratio: float = 0.3,
        importance_ratio: float = 0.3,
        uniform_ratio: float = 0.4
    ):
        super().__init__(num_points)
        
        total = fps_ratio + importance_ratio + uniform_ratio
        self.fps_ratio = fps_ratio / total
        self.importance_ratio = importance_ratio / total
        self.uniform_ratio = uniform_ratio / total
        
        self.fps_sampler = FarthestPointSampler(
            int(num_points * self.fps_ratio)
        )
        self.importance_sampler = ImportanceSampler(
            int(num_points * self.importance_ratio)
        )
        self.uniform_sampler = UniformSampler(
            num_points - int(num_points * self.fps_ratio) - 
            int(num_points * self.importance_ratio)
        )
        
    def sample(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample using hybrid strategy.
        
        Args:
            points: Input points [N, 3]
            normals: Optional normals [N, 3]
            
        Returns:
            Sampled points [num_points, 3]
        """
        samples = []
        
        # FPS for coverage
        if self.fps_sampler.num_points > 0:
            fps_points = self.fps_sampler.sample(points)
            samples.append(fps_points)
            
        # Importance sampling for detail
        if self.importance_sampler.num_points > 0:
            imp_points = self.importance_sampler.sample(points, normals)
            samples.append(imp_points)
            
        # Uniform for diversity
        if self.uniform_sampler.num_points > 0:
            uni_points = self.uniform_sampler.sample(points)
            samples.append(uni_points)
            
        return np.concatenate(samples, axis=0)


class SDFSampler:
    """
    Sampler for SDF training with near-surface and free-space points.
    
    Generates:
    - Surface points (from input)
    - Near-surface points (perturbed from surface)
    - Free-space points (random in volume)
    """
    
    def __init__(
        self,
        num_surface: int = 2048,
        num_near_surface: int = 2048,
        num_free_space: int = 1024,
        near_surface_std: float = 0.01,
        volume_bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.num_surface = num_surface
        self.num_near_surface = num_near_surface
        self.num_free_space = num_free_space
        self.near_surface_std = near_surface_std
        self.volume_bounds = volume_bounds
        
        self.surface_sampler = UniformSampler(num_surface)
        
    def sample(
        self,
        surface_points: np.ndarray,
        surface_normals: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points for SDF training.
        
        Args:
            surface_points: Surface points [N, 3]
            surface_normals: Optional surface normals [N, 3]
            
        Returns:
            Tuple of:
                - query_points: [num_total, 3]
                - sdf_values: [num_total] (0 for surface, estimated for others)
        """
        all_points = []
        all_sdf = []
        
        # Surface points (SDF = 0)
        surface_pts = self.surface_sampler.sample(surface_points)
        all_points.append(surface_pts)
        all_sdf.append(np.zeros(len(surface_pts)))
        
        # Near-surface points
        if self.num_near_surface > 0:
            # Sample base points
            base_indices = np.random.choice(
                len(surface_points),
                self.num_near_surface
            )
            base_pts = surface_points[base_indices]
            
            # Add noise
            noise = np.random.randn(self.num_near_surface, 3) * self.near_surface_std
            near_pts = base_pts + noise
            
            # Estimate SDF as signed distance along normal
            if surface_normals is not None:
                base_normals = surface_normals[base_indices]
                sdf_values = np.sum(noise * base_normals, axis=1)
            else:
                # Just use unsigned distance
                sdf_values = np.linalg.norm(noise, axis=1)
                # Random sign
                sdf_values *= np.random.choice([-1, 1], self.num_near_surface)
                
            all_points.append(near_pts)
            all_sdf.append(sdf_values)
            
        # Free-space points (SDF = large positive)
        if self.num_free_space > 0:
            low, high = self.volume_bounds
            free_pts = np.random.uniform(
                low, high,
                (self.num_free_space, 3)
            )
            
            # Estimate SDF as distance to nearest surface point
            # This is an approximation
            dists = np.min(
                np.linalg.norm(
                    free_pts[:, None] - surface_points[None, :],
                    axis=2
                ),
                axis=1
            )
            
            all_points.append(free_pts)
            all_sdf.append(dists)  # Always positive in free space
            
        query_points = np.concatenate(all_points, axis=0)
        sdf_values = np.concatenate(all_sdf, axis=0)
        
        return query_points, sdf_values


def get_sampler(
    sampler_type: str,
    num_points: int = 4096,
    **kwargs
) -> PointSampler:
    """
    Create point sampler by type.
    
    Args:
        sampler_type: Type of sampler ('uniform', 'fps', 'importance', 'hybrid')
        num_points: Number of points to sample
        **kwargs: Additional sampler arguments
        
    Returns:
        Point sampler instance
    """
    samplers = {
        'uniform': UniformSampler,
        'fps': FarthestPointSampler,
        'importance': ImportanceSampler,
        'hybrid': HybridSampler
    }
    
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
        
    return samplers[sampler_type](num_points, **kwargs)
