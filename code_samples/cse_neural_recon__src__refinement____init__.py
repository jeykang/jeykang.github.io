"""
Refinement module for post-processing point clouds and meshes.
"""

from .mesh_extraction import (
    MarchingCubesExtractor,
    SDFMeshExtractor,
    extract_mesh_from_sdf
)
from .statistical import (
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
    voxel_downsample,
    estimate_normals
)
from .neural_refinement import (
    NeuralRefiner,
    IterativeRefinement
)
from .point_completion import (
    PointCompletionNetwork,
    complete_point_cloud
)

__all__ = [
    # Mesh extraction
    'MarchingCubesExtractor',
    'SDFMeshExtractor',
    'extract_mesh_from_sdf',
    # Statistical refinement
    'StatisticalOutlierRemoval',
    'RadiusOutlierRemoval',
    'voxel_downsample',
    'estimate_normals',
    # Neural refinement
    'NeuralRefiner',
    'IterativeRefinement',
    # Point completion
    'PointCompletionNetwork',
    'complete_point_cloud'
]
