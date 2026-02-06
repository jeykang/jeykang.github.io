"""
Neural 3D Reconstruction for MobileX Poles Surveillance Robot.

This package provides a complete pipeline for converting multi-camera video 
feeds into high-fidelity, watertight 3D point clouds for surveillance 
coverage analysis.

Modules:
    - data: Data loading and preprocessing (CSE dataset support)
    - models: Neural network architectures (Neural SDF, encodings, etc.)
    - losses: Loss functions (SDF, planar, regularization)
    - training: Training infrastructure (trainer, schedulers, checkpoints)
    - refinement: Post-processing (mesh extraction, statistical filtering)
    - utils: Utilities (visualization, I/O, metrics)
"""

__version__ = '0.1.0'

# Avoid importing heavy optional deps (e.g., Open3D) at package import time.
# Training scripts typically import from `src.data` / `src.models` directly.

_exported = {'__version__'}

try:
    from .models import (  # noqa: F401
        NeuralSDF,
        NeuralSDFWithPlanar,
        HashGridEncoding,
        PositionalEncoding,
        PlanarAttention,
        IterativePointFilter,
        ScoreNetwork,
    )
    _exported.update({
        'NeuralSDF',
        'NeuralSDFWithPlanar',
        'HashGridEncoding',
        'PositionalEncoding',
        'PlanarAttention',
        'IterativePointFilter',
        'ScoreNetwork',
    })
except Exception:
    pass

try:
    from .losses import (  # noqa: F401
        SDFLoss,
        SurfaceLoss,
        FreespaceLoss,
        EikonalLoss,
        PlanarConsistencyLoss,
        ManhattanLoss,
        SmoothnessLoss,
    )
    _exported.update({
        'SDFLoss',
        'SurfaceLoss',
        'FreespaceLoss',
        'EikonalLoss',
        'PlanarConsistencyLoss',
        'ManhattanLoss',
        'SmoothnessLoss',
    })
except Exception:
    pass

try:
    from .training import (  # noqa: F401
        Trainer,
        TrainingConfig,
        CheckpointManager,
        get_scheduler,
        get_sampler,
    )
    _exported.update({
        'Trainer',
        'TrainingConfig',
        'CheckpointManager',
        'get_scheduler',
        'get_sampler',
    })
except Exception:
    pass

try:
    from .utils import (  # noqa: F401
        visualize_point_cloud,
        visualize_mesh,
        visualize_depth,
        load_point_cloud,
        save_point_cloud,
        chamfer_distance,
        f_score,
    )
    _exported.update({
        'visualize_point_cloud',
        'visualize_mesh',
        'visualize_depth',
        'load_point_cloud',
        'save_point_cloud',
        'chamfer_distance',
        'f_score',
    })
except Exception:
    pass

try:
    from .data import (  # noqa: F401
        CSEDataset,
        CSEMultiCameraDataset,
        MultiCameraSynchronizer,
        CameraRig,
        DataAugmentation,
        DepthProcessor,
    )
    _exported.update({
        'CSEDataset',
        'CSEMultiCameraDataset',
        'MultiCameraSynchronizer',
        'CameraRig',
        'DataAugmentation',
        'DepthProcessor',
    })
except Exception:
    pass

try:
    from .refinement import (  # noqa: F401
        extract_mesh_from_sdf,
        StatisticalOutlierRemoval,
        RadiusOutlierRemoval,
        voxel_downsample,
        NeuralRefiner,
        PointCompletionNetwork,
    )
    _exported.update({
        'extract_mesh_from_sdf',
        'StatisticalOutlierRemoval',
        'RadiusOutlierRemoval',
        'voxel_downsample',
        'NeuralRefiner',
        'PointCompletionNetwork',
    })
except Exception:
    pass

__all__ = sorted(_exported)
