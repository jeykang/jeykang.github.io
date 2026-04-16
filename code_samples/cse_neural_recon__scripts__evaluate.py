#!/usr/bin/env python3
"""
Evaluation script for neural 3D reconstruction.

Computes metrics against ground truth point clouds/meshes.

Usage:
    python scripts/evaluate.py --prediction output/inference/reconstruction.ply \
                               --ground_truth data/ground_truth.ply
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_point_cloud,
    chamfer_distance,
    f_score,
    hausdorff_distance,
    compute_all_metrics
)
from src.refinement import StatisticalOutlierRemoval


def setup_logging(output_dir: Path, name: str = 'evaluate'):
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


def load_and_preprocess(
    path: str,
    max_points: int = 100000,
    remove_outliers: bool = True
) -> np.ndarray:
    """Load and preprocess point cloud."""
    points, _, _ = load_point_cloud(path)
    
    # Remove outliers
    if remove_outliers:
        filter = StatisticalOutlierRemoval(k_neighbors=20, std_ratio=2.0)
        points = filter.filter(points)
    
    # Subsample if too large
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        
    return points


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50
) -> np.ndarray:
    """
    Align source to target using ICP.
    Returns transformed source points.
    """
    try:
        import open3d as o3d
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        # Compute normals for colored ICP
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        # ICP registration
        threshold = 0.1  # Max correspondence distance
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )
        
        # Transform source
        source_pcd.transform(result.transformation)
        return np.asarray(source_pcd.points)
        
    except ImportError:
        # Skip alignment if Open3D not available
        return source


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D reconstruction')
    parser.add_argument('--prediction', type=str, required=True,
                        help='Path to predicted point cloud')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth point cloud')
    parser.add_argument('--output', type=str, default='output/evaluation',
                        help='Output directory for results')
    parser.add_argument('--align', action='store_true',
                        help='Align prediction to ground truth using ICP')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Distance threshold for F-score')
    parser.add_argument('--max_points', type=int, default=100000,
                        help='Max points for evaluation')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting evaluation")
    logger.info(f"Prediction: {args.prediction}")
    logger.info(f"Ground truth: {args.ground_truth}")
    
    # Load point clouds
    logger.info("Loading point clouds...")
    pred_points = load_and_preprocess(args.prediction, args.max_points)
    gt_points = load_and_preprocess(args.ground_truth, args.max_points)
    logger.info(f"Prediction: {len(pred_points)} points")
    logger.info(f"Ground truth: {len(gt_points)} points")
    
    # Align if requested
    if args.align:
        logger.info("Aligning point clouds...")
        pred_points = align_point_clouds(pred_points, gt_points)
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    metrics = compute_all_metrics(
        pred_points,
        gt_points,
        threshold=args.threshold
    )
    
    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    
    logger.info(f"Chamfer Distance: {metrics['chamfer_distance']:.6f}")
    logger.info(f"F-Score (τ={args.threshold}): {metrics['f_score']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Hausdorff Distance: {metrics['hausdorff']:.6f}")
    logger.info(f"Hausdorff 95%: {metrics['hausdorff_95']:.6f}")
    
    # Compute F-scores at multiple thresholds
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
    f_scores_multi = {}
    
    for t in thresholds:
        f, p, r = f_score(pred_points, gt_points, threshold=t)
        f_scores_multi[f'f_score_tau{t}'] = f
        logger.info(f"F-Score (τ={t}): {f:.4f}")
    
    # Save results
    results = {
        **metrics,
        **f_scores_multi,
        'prediction_path': args.prediction,
        'ground_truth_path': args.ground_truth,
        'num_pred_points': len(pred_points),
        'num_gt_points': len(gt_points),
        'aligned': args.align
    }
    
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
