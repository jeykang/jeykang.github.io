#!/usr/bin/env python3
"""
Training script for neural 3D reconstruction.

Usage:
    python scripts/train.py --config config/experiment/cse_warehouse.yaml
    python scripts/train.py --config config/default.yaml --data_path data/warehouse_extracted/static_warehouse_robot1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CSEDataset,
    CSEMultiCameraDataset,
    create_multi_sequence_dataset,
    create_multi_environment_dataset,
    CSEViewSetDataset,
    MultiViewConfig,
)
from src.models import NeuralSDF, NeuralSDFWithPlanar, HashGridSDF, PixelAlignedConditionalSDF
from src.losses import SDFLoss, PlanarConsistencyLoss, ManhattanLoss
from src.training import Trainer, TrainingConfig
from src.utils.auto_batch_size import auto_batch_size_for_sdf, get_gpu_memory_info


def setup_logging(output_dir: Path, name: str = 'train'):
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with defaults handling."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle 'defaults' key for inheritance
    if 'defaults' in config:
        config_dir = Path(config_path).parent
        base_configs = config.pop('defaults')
        
        # Load base configs
        for base in base_configs:
            if isinstance(base, str):
                base_path = config_dir / f'{base}.yaml'
                if base_path.exists():
                    with open(base_path, 'r') as f:
                        base_config = yaml.safe_load(f)
                    # Merge base config (base config is overridden by current)
                    base_config = deep_merge(base_config, config)
                    config = base_config
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_model(config: dict, device: str = 'cuda') -> nn.Module:
    """Create neural SDF model from config."""
    model_config = config.get('model', {})
    sdf_config = model_config.get('neural_sdf', {})
    planar_config = model_config.get('planar_attention', {})
    
    # Get encoding configuration
    encoding_type = sdf_config.get('encoding_type', 'hashgrid')
    
    if encoding_type == 'hashgrid':
        hashgrid_config = sdf_config.get('hashgrid', {})
        encoding_config = {
            'num_levels': hashgrid_config.get('num_levels', 16),
            'features_per_level': hashgrid_config.get('features_per_level', 2),
            'log2_hashmap_size': hashgrid_config.get('log2_hashmap_size', 19),
            'base_resolution': hashgrid_config.get('base_resolution', 16),
            'max_resolution': hashgrid_config.get('max_resolution', 2048),
        }
    elif encoding_type == 'positional':
        pos_config = sdf_config.get('positional', {})
        encoding_config = {
            'num_frequencies': pos_config.get('num_frequencies', 10),
            'include_input': pos_config.get('include_input', True),
        }
    else:
        encoding_config = None
    
    hidden_features = sdf_config.get('hidden_features', 256)
    hidden_layers = sdf_config.get('hidden_layers', 6)

    # Conditional pixel-aligned model (multi-view RGB(+pose) -> local geometry).
    if bool(sdf_config.get('conditional', False)):
        cond_cfg = sdf_config.get('conditioning', {}) or {}
        model = PixelAlignedConditionalSDF(
            encoding_config=encoding_config,
            cond_dim=int(cond_cfg.get('cond_dim', 64)),
            encoder_out_dim=int(cond_cfg.get('encoder_out_dim', 64)),
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            use_weight_norm=bool(sdf_config.get('use_weight_norm', True)),
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Conditional model parameters: {num_params:,}")
        return model.to(device)
    
    # Use HashGridSDF when encoding is hashgrid (proper architecture for hash grids)
    use_planar = planar_config.get('enabled', False)  # Disable planar by default for stability
    
    if encoding_type == 'hashgrid' and not use_planar:
        # Use the new HashGridSDF architecture designed for hash grid encoding
        # This uses ReLU MLP instead of SIREN, with geometric initialization
        print("Using HashGridSDF model with geometric initialization")
        
        model = HashGridSDF(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            encoding_config=encoding_config,
            output_color=sdf_config.get('output_color', True),
            use_weight_norm=sdf_config.get('use_weight_norm', True),
            geometric_init=sdf_config.get('geometric_init', True),
            sphere_init_radius=sdf_config.get('sphere_init_radius', 0.5),
        )
    elif use_planar:
        # Build planar attention config
        planar_attention_config = {
            'num_heads': planar_config.get('num_heads', 8),
            'max_planes': planar_config.get('max_planes', 30),
        }
        
        model = NeuralSDFWithPlanar(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=sdf_config.get('omega_0', 30.0),
            encoding_type=encoding_type,
            encoding_config=encoding_config,
            output_color=sdf_config.get('output_color', True),
            planar_attention_config=planar_attention_config,
        )
    else:
        model = NeuralSDF(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=sdf_config.get('omega_0', 30.0),
            encoding_type=encoding_type,
            encoding_config=encoding_config,
            output_color=sdf_config.get('output_color', True),
        )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    return model.to(device)


def create_dataset(config: dict):
    """Create dataset from config.
    
    Supports:
    - Single sequence training
    - Multi-sequence training (same environment, multiple trajectories)
    - Multi-environment training (different environments with all sequences)
    """
    data_config = config.get('data', {})
    
    # Get data path
    data_path = data_config.get('data_path', 'data/warehouse_extracted/static_warehouse_robot1')
    
    # Image size
    img_width = data_config.get('img_width', 640)
    img_height = data_config.get('img_height', 360)
    
    # Depth settings
    min_depth = data_config.get('min_depth', 0.1)
    max_depth = data_config.get('max_depth', 20.0)
    # CSE dataset depth is in mm (uint16), need to scale to meters
    depth_scale = data_config.get('depth_scale', 0.001)
    
    # Common dataset kwargs
    dataset_kwargs = dict(
        side=data_config.get('side', 'left'),
        img_wh=(img_width, img_height),
        min_depth=min_depth,
        max_depth=max_depth,
        depth_scale=depth_scale,
        cache_frames=data_config.get('cache_frames', False),
    )

    # Multi-view wrapper configuration (optional)
    mv_cfg = data_config.get('multiview', {}) or {}
    use_multiview = bool(mv_cfg.get('enabled', False))
    # Conditional SDF requires a viewset batch format.
    is_conditional = bool(config.get('model', {}).get('neural_sdf', {}).get('conditional', False))
    if is_conditional:
        use_multiview = True
    mv_num_views = int(mv_cfg.get('num_views', 6))
    mv_window = int(mv_cfg.get('window', 6))
    mv_include_ref = bool(mv_cfg.get('include_reference', True))

    # Stereo multi-camera dataset (optional; useful as 2 views per timestamp).
    multicam_cfg = data_config.get('multicamera', {}) or {}
    use_multicamera = bool(multicam_cfg.get('enabled', False))
    cameras = multicam_cfg.get('cameras', None)
    sync_tolerance = float(multicam_cfg.get('sync_tolerance', 0.05))
    # Normalize camera config (YAML-friendly).
    if isinstance(cameras, list) and cameras:
        if all(isinstance(c, str) for c in cameras):
            # Let CSEMultiCameraDataset use its default stereo pair.
            cameras = None
        else:
            norm = []
            for cam in cameras:
                if not isinstance(cam, dict) or 'side' not in cam:
                    raise ValueError("multicamera.cameras must be a list of {side, extrinsic} dicts or omitted")
                extr = cam.get('extrinsic', None)
                if extr is not None and not isinstance(extr, np.ndarray):
                    extr = np.array(extr, dtype=np.float32)
                norm.append({'side': cam['side'], 'extrinsic': extr if extr is not None else np.eye(4, dtype=np.float32)})
            cameras = norm
    
    # Check for multi-environment training (highest priority)
    multi_environment = data_config.get('multi_environment', False)
    environments = data_config.get('environments', None)
    
    if multi_environment and environments:
        # Multi-environment mode: combine data from different scenes
        print("Loading MULTI-ENVIRONMENT dataset")
        dataset = create_multi_environment_dataset(
            environments=environments,
            **dataset_kwargs
        )
    elif data_config.get('multi_sequence', False):
        # Multi-sequence mode: combine multiple trajectories from same environment
        sequences = data_config.get('sequences', None)
        sequence_pattern = data_config.get('sequence_pattern', 'static_*')
        
        base_dir = data_config.get('base_dir', data_path)
        if not os.path.isdir(base_dir):
            # data_path might be a specific sequence - use its parent
            base_dir = os.path.dirname(data_path)
        
        print(f"Loading multi-sequence dataset from {base_dir}")
        if use_multicamera:
            # Build a ConcatDataset of multi-camera datasets for each sequence.
            seqs = sequences
            if seqs is None:
                import glob
                seq_dirs = sorted(glob.glob(os.path.join(base_dir, sequence_pattern)))
                seqs = [os.path.basename(d) for d in seq_dirs if os.path.isdir(d)]
            if not seqs:
                raise ValueError(f"No sequences found in {base_dir} matching {sequence_pattern}")
            cams = cameras  # None => dataset default stereo extrinsic
            datasets = []
            for seq_name in seqs:
                seq_path = os.path.join(base_dir, seq_name)
                if not os.path.isdir(seq_path):
                    continue
                datasets.append(
                    CSEMultiCameraDataset(
                        run_dir=seq_path,
                        cameras=cams,
                        img_wh=(img_width, img_height),
                        sync_tolerance=sync_tolerance,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        depth_scale=depth_scale,
                        cache_frames=data_config.get('cache_frames', False),
                    )
                )
            dataset = torch.utils.data.ConcatDataset(datasets)
        else:
            dataset = create_multi_sequence_dataset(
                base_dir=base_dir,
                sequences=sequences,
                pattern=sequence_pattern,
                **dataset_kwargs
            )
    else:
        # Single sequence mode
        if use_multicamera:
            cams = cameras  # None => dataset default stereo extrinsic
            dataset = CSEMultiCameraDataset(
                run_dir=str(data_path),
                cameras=cams,
                img_wh=(img_width, img_height),
                sync_tolerance=sync_tolerance,
                min_depth=min_depth,
                max_depth=max_depth,
                depth_scale=depth_scale,
                cache_frames=data_config.get('cache_frames', False),
            )
        else:
            dataset = CSEDataset(run_dir=str(data_path), **dataset_kwargs)

    if use_multiview:
        dataset = CSEViewSetDataset(
            dataset,
            config=MultiViewConfig(
                num_views=mv_num_views,
                window=mv_window,
                include_reference=mv_include_ref,
            ),
        )
    
    return dataset


def create_dataloader(
    dataset: CSEDataset, 
    config: dict, 
    split: str = 'train'
) -> torch.utils.data.DataLoader:
    """Create dataloader from dataset and config."""
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    batch_size = train_config.get('batch_size', 4) if split == 'train' else 1
    shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing when real data is not available."""
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        H, W = 360, 640
        
        # Random 3D points
        points = torch.randn(1024, 3)
        
        # Random depth image with valid values
        depth = torch.rand(H, W) * 9.5 + 0.5  # 0.5 to 10.0 meters
        
        # Valid mask - all points are valid for dummy data
        valid_mask = torch.ones(H, W, dtype=torch.bool)
        
        # Random RGB image
        rgb = torch.rand(3, H, W)
        
        # Random pose (identity with small perturbation)
        pose = torch.eye(4)
        pose[:3, 3] = torch.randn(3) * 0.1
        
        # Camera intrinsics (typical values for 640x360)
        K = torch.tensor([
            [610.0, 0.0, 320.0],
            [0.0, 610.0, 180.0],
            [0.0, 0.0, 1.0]
        ])
        
        return {
            'points': points,
            'depth': depth,
            'valid_mask': valid_mask,
            'rgb': rgb,
            'pose': pose,
            'K': K,
            'intrinsics': K,  # Alias for compatibility
            'frame_idx': idx,
        }


def main():
    parser = argparse.ArgumentParser(description='Train neural 3D reconstruction')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path in config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs in config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--auto-batch-size', action='store_true',
                        help='Automatically determine optimal batch size for GPU')
    parser.add_argument('--target-memory', type=float, default=0.80,
                        help='Target GPU memory utilization (0.0-1.0) for auto batch sizing')
    parser.add_argument('--max-batch-size', type=int, default=64,
                        help='Maximum batch size to try when auto-scaling')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # Override epochs if provided
    if args.epochs is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['epochs'] = args.epochs
    
    # Create output directory
    exp_name = config.get('experiment_name', 'default')
    output_config = config.get('output', {})
    output_base = Path(output_config.get('output_dir', args.output))
    output_dir = output_base / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")
    
    # Auto batch size scaling
    train_config = config.get('training', {})
    auto_batch = getattr(args, 'auto_batch_size', False)
    # Auto batch size profiling does not account for multi-view image encoder memory.
    # In conditional mode, it tends to recommend absurdly large batch sizes.
    if hasattr(model, 'encode_views') or bool(config.get('data', {}).get('multiview', {}).get('enabled', False)):
        if auto_batch:
            logger.info("Disabling --auto-batch-size for multi-view/conditional training (encoder memory not profiled).")
        auto_batch = False
    
    if auto_batch and torch.cuda.is_available():
        logger.info("=" * 60)
        logger.info("Auto batch size scaling enabled")
        total_gb, available_gb = get_gpu_memory_info(torch.device(device))
        logger.info(f"GPU Memory: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
        
        # Get sampling parameters for memory profiling
        num_points = (
            train_config.get('num_surface_samples', 8192) +
            train_config.get('num_freespace_samples', 8192) +
            train_config.get('num_random_samples', 4096)
        )
        
        logger.info(f"Profiling with {num_points:,} points per sample...")
        logger.info(f"Target memory utilization: {args.target_memory * 100:.0f}%")
        
        try:
            memory_profile = auto_batch_size_for_sdf(
                model=model,
                num_points_per_sample=num_points,
                min_batch_size=1,
                max_batch_size=args.max_batch_size,
                target_memory_fraction=args.target_memory,
                device=torch.device(device),
                verbose=True,
            )
            
            optimal_batch_size = memory_profile.recommended_batch_size
            logger.info(f"Recommended batch size: {optimal_batch_size}")
            logger.info(f"Memory per sample: {memory_profile.memory_per_sample_mb:.1f} MB")
            logger.info(f"Model memory: {memory_profile.model_memory_mb:.1f} MB")
            
            # Update config with optimal batch size
            if 'training' not in config:
                config['training'] = {}
            config['training']['batch_size'] = optimal_batch_size
            train_config['batch_size'] = optimal_batch_size
            
            # Save updated config
            with open(output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            logger.info(f"Batch size set to: {optimal_batch_size}")
        except Exception as e:
            logger.warning(f"Auto batch size detection failed: {e}")
            logger.info("Falling back to config batch size")
        
        logger.info("=" * 60)
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    try:
        train_dataset = create_dataset(config)
        train_loader = create_dataloader(train_dataset, config, 'train')
        logger.info(f"Train samples: {len(train_dataset)}")
    except Exception as e:
        logger.warning(f"Failed to create dataset: {e}")
        logger.info("Using dummy dataset for testing...")
        train_dataset = DummyDataset(100)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        logger.info("Using dummy dataset with 100 samples")
    
    # Create training configuration (train_config was already loaded above)
    training_config = TrainingConfig(
        epochs=int(train_config.get('epochs', 50)),
        batch_size=int(train_config.get('batch_size', 8)),
        num_workers=int(config.get('data', {}).get('num_workers', 4)),
        optimizer=str(train_config.get('optimizer', 'adamw')),
        learning_rate=float(train_config.get('learning_rate', 5e-4)),
        weight_decay=float(train_config.get('weight_decay', 1e-4)),
        encoding_lr_mult=float(train_config.get('encoding_lr_mult', 10.0)),
        encoding_weight_decay=float(train_config.get('encoding_weight_decay', 0.0)),
        scheduler=str(train_config.get('scheduler', 'cosine')),
        warmup_epochs=int(train_config.get('warmup_epochs', 3)),
        min_lr=float(train_config.get('min_lr', 1e-6)),
        use_amp=bool(train_config.get('use_amp', True)),
        grad_clip=float(train_config.get('gradient_clip', 1.0)),
        # Sampling
        num_surface_samples=int(train_config.get('num_surface_samples', 8192)),
        num_freespace_samples=int(train_config.get('num_freespace_samples', 8192)),
        num_random_samples=int(train_config.get('num_random_samples', 4096)),
        freespace_jitter_min=float(train_config.get('freespace_jitter_min', 0.1)),
        freespace_jitter_max=float(train_config.get('freespace_jitter_max', 3.0)),
        tsdf_num_depth_samples=int(train_config.get('tsdf_num_depth_samples', 8)),
        tsdf_num_rays=train_config.get('tsdf_num_rays', None),
        # Loss weights
        surface_weight=float(train_config.get('surface_weight', 1.0)),
        freespace_weight=float(train_config.get('freespace_weight', 2.0)),
        eikonal_weight=float(train_config.get('eikonal_weight', 0.05)),
        random_space_weight=float(train_config.get('random_space_weight', 1.5)),
        tsdf_weight=float(train_config.get('tsdf_weight', 1.0)),
        tsdf_behind_weight=float(train_config.get('tsdf_behind_weight', 0.1)),
        use_tsdf_supervision=bool(train_config.get('use_tsdf_supervision', True)),
        truncation_dist=float(train_config.get('truncation_dist', 0.2)),
        # Logging
        log_every=int(train_config.get('log_every', 100)),
        val_every=int(train_config.get('val_every', 1)),
        save_every=int(train_config.get('save_every', 5)),
        visualize_every=int(train_config.get('visualize_every', 5)),
        output_dir=str(output_dir),
        experiment_name=exp_name,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Using same loader for now
        config=training_config,
        device=torch.device(device),
    )
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    logger.info("Training complete!")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
