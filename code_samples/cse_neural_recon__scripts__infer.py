#!/usr/bin/env python3
"""
Inference script for neural 3D reconstruction.

Generates point clouds and meshes from trained models.

Usage:
    python scripts/infer.py --checkpoint output/exp/checkpoints/best_checkpoint.pt \
                            --data_root data/warehouse_extracted/static_warehouse_robot1 \
                            --output output/inference
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CSEDataset
from src.models import NeuralSDF, NeuralSDFWithPlanar, HashGridEncoding
from src.refinement import (
    extract_mesh_from_sdf,
    StatisticalOutlierRemoval,
    voxel_downsample
)
from src.utils import save_point_cloud, save_mesh


def setup_logging(output_dir: Path, name: str = 'infer'):
    """Configure logging."""
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


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Recreate encoding
    encoding_config = model_config.get('encoding', {})
    encoding_type = encoding_config.get('type', 'hashgrid')
    
    if encoding_type == 'hashgrid':
        encoding = HashGridEncoding(
            n_levels=encoding_config.get('n_levels', 16),
            n_features_per_level=encoding_config.get('n_features_per_level', 2),
            log2_hashmap_size=encoding_config.get('log2_hashmap_size', 19),
            base_resolution=encoding_config.get('base_resolution', 16),
            finest_resolution=encoding_config.get('finest_resolution', 512)
        )
        input_dim = encoding.output_dim
    else:
        encoding = None
        input_dim = 3
    
    # Recreate model
    model_type = model_config.get('type', 'neural_sdf_planar')
    
    if model_type == 'neural_sdf':
        model = NeuralSDF(
            in_features=input_dim,
            hidden_features=model_config.get('hidden_dim', 256),
            hidden_layers=model_config.get('num_layers', 4),
            out_features=1,
            encoding=encoding
        )
    else:
        model = NeuralSDFWithPlanar(
            in_features=input_dim,
            hidden_features=model_config.get('hidden_dim', 256),
            hidden_layers=model_config.get('num_layers', 4),
            num_planes=model_config.get('num_planes', 32),
            encoding=encoding
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def extract_point_cloud(
    model,
    resolution: int = 256,
    bounds: tuple = ((-5, -5, -1), (5, 5, 3)),
    threshold: float = 0.0,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract point cloud from SDF model by finding zero-crossings.
    """
    min_bound = np.array(bounds[0])
    max_bound = np.array(bounds[1])
    
    # Create dense grid
    x = np.linspace(min_bound[0], max_bound[0], resolution)
    y = np.linspace(min_bound[1], max_bound[1], resolution)
    z = np.linspace(min_bound[2], max_bound[2], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Evaluate SDF in batches
    batch_size = 65536
    sdf_values = []
    
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch = torch.from_numpy(grid_points[i:i+batch_size]).float().to(device)
            sdf = model(batch)
            if isinstance(sdf, dict):
                sdf = sdf['sdf']
            sdf_values.append(sdf.cpu().numpy())
            
    sdf_values = np.concatenate(sdf_values)
    
    # Find near-surface points
    near_surface = np.abs(sdf_values) < threshold + 0.01
    points = grid_points[near_surface]
    
    return points


def main():
    parser = argparse.ArgumentParser(description='Run inference on neural SDF')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to data for context (optional)')
    parser.add_argument('--output', type=str, default='output/inference',
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Grid resolution for extraction')
    parser.add_argument('--extract_mesh', action='store_true',
                        help='Also extract triangle mesh')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Running inference from: {args.checkpoint}")
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model, config = load_model(args.checkpoint, device)
    logger.info("Model loaded successfully")
    
    # Get bounds from config or use defaults
    bounds_config = config.get('bounds', {})
    bounds = (
        tuple(bounds_config.get('min', [-5, -5, -1])),
        tuple(bounds_config.get('max', [5, 5, 3]))
    )
    logger.info(f"Volume bounds: {bounds}")
    
    # Extract point cloud
    logger.info(f"Extracting point cloud (resolution={args.resolution})...")
    points = extract_point_cloud(
        model,
        resolution=args.resolution,
        bounds=bounds,
        device=device
    )
    logger.info(f"Extracted {len(points)} points")
    
    # Refine point cloud
    logger.info("Refining point cloud...")
    
    # Statistical outlier removal
    outlier_filter = StatisticalOutlierRemoval(k_neighbors=20, std_ratio=2.0)
    points = outlier_filter.filter(points)
    logger.info(f"After outlier removal: {len(points)} points")
    
    # Voxel downsampling
    points = voxel_downsample(points, voxel_size=0.02)
    logger.info(f"After downsampling: {len(points)} points")
    
    # Save point cloud
    pc_path = output_dir / 'reconstruction.ply'
    save_point_cloud(str(pc_path), points)
    logger.info(f"Saved point cloud to: {pc_path}")
    
    # Extract and save mesh if requested
    if args.extract_mesh:
        logger.info("Extracting mesh...")
        mesh = extract_mesh_from_sdf(
            model,
            resolution=args.resolution,
            bounds=bounds,
            device=device,
            refine=True
        )
        
        mesh_path = output_dir / 'reconstruction_mesh.ply'
        mesh.save(str(mesh_path))
        logger.info(f"Saved mesh to: {mesh_path}")
        logger.info(f"Mesh: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    
    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
