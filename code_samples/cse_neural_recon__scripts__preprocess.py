#!/usr/bin/env python3
"""
Preprocessing script for CSE dataset.

Prepares data for training:
- Generates depth from stereo (if needed)
- Computes point clouds from depth + poses
- Creates train/val splits
- Generates SDF samples for training

Usage:
    python scripts/preprocess.py --data_root data/warehouse_extracted \
                                  --output data/warehouse_processed \
                                  --sequences static_warehouse_robot1 static_warehouse_robot2
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CSEDataset
from src.data.depth_processing import DepthProcessor


def setup_logging(output_dir: Path, name: str = 'preprocess'):
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


def load_intrinsics(json_path: Path) -> dict:
    """Load camera intrinsics from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        'fx': data['K'][0],
        'fy': data['K'][4],
        'cx': data['K'][2],
        'cy': data['K'][5],
        'width': data['width'],
        'height': data['height']
    }


def load_poses(pose_file: Path) -> List[dict]:
    """Load poses from TUM format file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                poses.append({
                    'timestamp': float(parts[0]),
                    'tx': float(parts[1]),
                    'ty': float(parts[2]),
                    'tz': float(parts[3]),
                    'qx': float(parts[4]),
                    'qy': float(parts[5]),
                    'qz': float(parts[6]),
                    'qw': float(parts[7])
                })
    return poses


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: dict,
    pose: Optional[np.ndarray] = None,
    max_depth: float = 10.0
) -> np.ndarray:
    """Convert depth image to point cloud."""
    h, w = depth.shape
    
    # Create pixel coordinates
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Back-project to 3D
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack and filter
    points = np.stack([x, y, z], axis=-1)
    valid = (z > 0) & (z < max_depth)
    points = points[valid]
    
    # Transform to world coordinates
    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3]
        points = points @ R.T + t
        
    return points


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Convert quaternion to 4x4 transformation matrix."""
    # Normalize
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    
    # Rotation matrix
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    
    return T


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    logger,
    depth_processor: DepthProcessor,
    frame_skip: int = 1,
    max_frames: int = -1
):
    """Process a single sequence."""
    logger.info(f"Processing sequence: {sequence_dir.name}")
    
    # Create output directory
    seq_output = output_dir / sequence_dir.name
    seq_output.mkdir(parents=True, exist_ok=True)
    
    # Load intrinsics
    intrinsics_file = sequence_dir / 'camera_info_left_intrinsics.json'
    intrinsics = load_intrinsics(intrinsics_file)
    logger.info(f"Loaded intrinsics: {intrinsics['width']}x{intrinsics['height']}")
    
    # Load poses
    poses_file = sequence_dir / 'ground_truth.txt'
    poses = load_poses(poses_file)
    logger.info(f"Loaded {len(poses)} poses")
    
    # Get depth files
    depth_dir = sequence_dir / 'depth_left'
    depth_files = sorted(depth_dir.glob('*.png'))
    logger.info(f"Found {len(depth_files)} depth frames")
    
    # Process frames
    all_points = []
    processed = 0
    
    for i, depth_file in enumerate(depth_files):
        if i % frame_skip != 0:
            continue
            
        if max_frames > 0 and processed >= max_frames:
            break
            
        # Load depth
        depth = np.array(load_depth_image(depth_file))
        
        # Process depth
        depth = depth_processor.process(depth)
        
        # Get corresponding pose (by timestamp or index)
        pose_idx = min(i, len(poses) - 1)
        pose_data = poses[pose_idx]
        
        # Convert to matrix
        pose_matrix = quaternion_to_matrix(
            pose_data['qx'], pose_data['qy'],
            pose_data['qz'], pose_data['qw']
        )
        pose_matrix[:3, 3] = [pose_data['tx'], pose_data['ty'], pose_data['tz']]
        
        # Generate point cloud
        points = depth_to_point_cloud(depth, intrinsics, pose_matrix)
        all_points.append(points)
        
        processed += 1
        
        if processed % 100 == 0:
            logger.info(f"  Processed {processed} frames")
    
    # Combine all points
    combined_points = np.concatenate(all_points, axis=0)
    logger.info(f"Total points: {len(combined_points)}")
    
    # Downsample
    from src.refinement import voxel_downsample
    combined_points = voxel_downsample(combined_points, voxel_size=0.02)
    logger.info(f"After downsampling: {len(combined_points)} points")
    
    # Save point cloud
    pc_file = seq_output / 'point_cloud.npz'
    np.savez(pc_file, points=combined_points)
    logger.info(f"Saved point cloud to: {pc_file}")
    
    # Save metadata
    metadata = {
        'sequence': sequence_dir.name,
        'num_frames': processed,
        'num_points': len(combined_points),
        'intrinsics': intrinsics,
        'bounds': {
            'min': combined_points.min(axis=0).tolist(),
            'max': combined_points.max(axis=0).tolist()
        }
    }
    
    meta_file = seq_output / 'metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return combined_points


def load_depth_image(path: Path) -> np.ndarray:
    """Load depth image from file."""
    try:
        from PIL import Image
        img = Image.open(path)
        depth = np.array(img).astype(np.float32)
        
        # Convert from mm to meters if needed
        if depth.max() > 100:
            depth = depth / 1000.0
            
        return depth
        
    except ImportError:
        import cv2
        depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        
        if depth.max() > 100:
            depth = depth.astype(np.float32) / 1000.0
            
        return depth


def generate_train_val_split(
    sequences: List[str],
    train_ratio: float = 0.8
) -> tuple:
    """Generate train/val split."""
    n_train = int(len(sequences) * train_ratio)
    
    np.random.shuffle(sequences)
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]
    
    return train_seqs, val_seqs


def main():
    parser = argparse.ArgumentParser(description='Preprocess CSE dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to extracted data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequences to process')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Max frames per sequence (-1 for all)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting preprocessing")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output: {args.output}")
    
    # Get sequences
    data_root = Path(args.data_root)
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = [d.name for d in data_root.iterdir() if d.is_dir()]
    
    logger.info(f"Processing {len(sequences)} sequences")
    
    # Create depth processor
    depth_processor = DepthProcessor(
        max_depth=10.0,
        min_depth=0.1,
        bilateral_filter=True,
        hole_filling=True
    )
    
    # Process each sequence
    all_points = []
    
    for seq_name in sequences:
        seq_dir = data_root / seq_name
        if not seq_dir.exists():
            logger.warning(f"Sequence not found: {seq_name}")
            continue
            
        points = process_sequence(
            seq_dir,
            output_dir,
            logger,
            depth_processor,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
        all_points.append(points)
    
    # Generate train/val split
    logger.info("Generating train/val split...")
    train_seqs, val_seqs = generate_train_val_split(
        list(sequences),
        train_ratio=args.train_ratio
    )
    
    split_info = {
        'train': train_seqs,
        'val': val_seqs
    }
    
    split_file = output_dir / 'split.json'
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Train sequences: {len(train_seqs)}")
    logger.info(f"Val sequences: {len(val_seqs)}")
    logger.info(f"Split saved to: {split_file}")
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()
