"""
Training infrastructure for neural 3D reconstruction.
"""

from .trainer import Trainer, TrainingConfig
from .scheduler import (
    get_scheduler,
    WarmupScheduler,
    CosineAnnealingWarmRestarts
)
from .checkpoint import (
    CheckpointManager,
    CheckpointInfo,
    save_checkpoint,
    load_checkpoint
)
from .samplers import (
    PointSampler,
    UniformSampler,
    FarthestPointSampler,
    ImportanceSampler,
    HybridSampler,
    SDFSampler,
    get_sampler
)
from .visualizer import TrainingVisualizer, generate_gt_point_cloud_from_batch

__all__ = [
    # Trainer
    'Trainer',
    'TrainingConfig',
    # Schedulers
    'get_scheduler',
    'WarmupScheduler',
    'CosineAnnealingWarmRestarts',
    # Checkpoints
    'CheckpointManager',
    'CheckpointInfo',
    'save_checkpoint',
    'load_checkpoint',
    # Samplers
    'PointSampler',
    'UniformSampler',
    'FarthestPointSampler',
    'ImportanceSampler',
    'HybridSampler',
    'SDFSampler',
    'get_sampler',
    # Visualization
    'TrainingVisualizer',
    'generate_gt_point_cloud_from_batch',
]
