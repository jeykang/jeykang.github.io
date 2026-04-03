"""
Utility modules for neural 3D reconstruction.
"""

from .visualization import (
    visualize_point_cloud,
    visualize_mesh,
    visualize_depth,
    visualize_sdf_slice,
    create_video_from_frames
)
from .io import (
    load_point_cloud,
    save_point_cloud,
    load_mesh,
    save_mesh,
    load_image,
    save_image
)
from .metrics import (
    chamfer_distance,
    point_cloud_accuracy,
    point_cloud_completeness,
    f_score,
    hausdorff_distance
)
from .productivity_metrics import (
    ProductivityMetricsCollector,
    ProductivityReport,
    GPUMetricsCollector,
    TrainingProfiler,
    create_comparison_baseline
)

__all__ = [
    # Visualization
    'visualize_point_cloud',
    'visualize_mesh',
    'visualize_depth',
    'visualize_sdf_slice',
    'create_video_from_frames',
    # IO
    'load_point_cloud',
    'save_point_cloud',
    'load_mesh',
    'save_mesh',
    'load_image',
    'save_image',
    # Metrics
    'chamfer_distance',
    'point_cloud_accuracy',
    'point_cloud_completeness',
    'f_score',
    'hausdorff_distance',
    # Productivity Metrics
    'ProductivityMetricsCollector',
    'ProductivityReport',
    'GPUMetricsCollector',
    'TrainingProfiler',
    'create_comparison_baseline',
]
