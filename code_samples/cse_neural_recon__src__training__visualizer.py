"""
Training visualization module for checkpoint evaluation.

Generates visual comparisons between model predictions and ground truth,
along with detailed metric breakdowns.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D


class TrainingVisualizer:
    """
    Generates visualizations and metrics for training checkpoints.
    
    Creates:
    - Side-by-side point cloud comparisons (GT vs Predicted)
    - Multiple view angles of reconstructions
    - Depth map comparisons
    - SDF slice visualizations
    - Per-epoch metric plots
    - Detailed metric reports
    """
    
    def __init__(
        self,
        output_dir: str,
        model: nn.Module,
        device: torch.device,
        scene_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        grid_resolution: int = 128,
    ):
        """
        Args:
            output_dir: Directory to save visualizations
            model: Neural SDF model
            device: Compute device
            scene_bounds: World coordinate scene bounds (min, max) as tensors - for display
            grid_resolution: Resolution for SDF grid extraction
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.device = device
        self.grid_resolution = grid_resolution
        
        # Store scene bounds for converting back to world coordinates
        # Model always operates in normalized [0, 1] space
        self.scene_bounds = scene_bounds  # Will be set by trainer
        
        # Model operates in normalized [0, 1] coordinate space
        self.bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
            
        # Metric history for plotting
        self.metric_history: Dict[str, List[float]] = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'surface_loss': [],
            'freespace_loss': [],
            'eikonal_loss': [],
            'chamfer_distance': [],
            'accuracy_1cm': [],
            'accuracy_5cm': [],
            'completeness': [],
        }
    
    def set_scene_bounds(self, scene_min: torch.Tensor, scene_max: torch.Tensor):
        """Set scene bounds for coordinate denormalization."""
        self.scene_bounds = (scene_min.cpu().numpy(), scene_max.cpu().numpy())
        print(f"Visualizer scene bounds set: min={self.scene_bounds[0]}, max={self.scene_bounds[1]}")
    
    def _denormalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert normalized [0, 1] points back to world coordinates for visualization."""
        if self.scene_bounds is None:
            return points  # No denormalization if bounds not set
        
        scene_min, scene_max = self.scene_bounds
        return points * (scene_max - scene_min) + scene_min
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to normalized [0, 1] coordinates."""
        if self.scene_bounds is None:
            return points
        
        scene_min, scene_max = self.scene_bounds
        return (points - scene_min) / (scene_max - scene_min + 1e-8)
        
    def visualize_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        sample_batch: Optional[Dict[str, torch.Tensor]] = None,
        gt_points: Optional[np.ndarray] = None,
        gt_groups: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Generate full visualization suite for a checkpoint.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            sample_batch: Sample batch for reconstruction
            gt_points: Ground truth point cloud for comparison
            
        Returns:
            Dictionary of computed evaluation metrics
        """
        epoch_dir = self.viz_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        eval_metrics = {}
        
        # Update metric history
        self._update_history(epoch, train_metrics, val_metrics)
        
        # 1. Generate reconstructed point cloud
        print(f"  Generating reconstruction for epoch {epoch}...")
        pred_points, pred_sdf = self._extract_point_cloud(sample_batch=sample_batch)
        
        # If conditional model and we have view poses, crop GT to local region.
        gt_points_local = gt_points
        crop_mn = None
        crop_mx = None
        if sample_batch is not None and gt_points is not None and len(gt_points) > 0:
            if 'views_pose' in sample_batch:
                cam_centers = sample_batch['views_pose'][0, :, :3, 3].detach().cpu().numpy()
                cmin = cam_centers.min(axis=0)
                cmax = cam_centers.max(axis=0)
                # Margin in meters; wide enough to include local context.
                margin = 10.0
                mn = cmin - margin
                mx = cmax + margin
                crop_mn, crop_mx = mn, mx
                mask = np.all((gt_points >= mn) & (gt_points <= mx), axis=1)
                if mask.any():
                    gt_points_local = gt_points[mask]

        # 2. Compare with ground truth if available and non-empty
        if gt_points_local is not None and len(gt_points_local) > 0 and len(pred_points) > 0:
            eval_metrics = self._compute_metrics(pred_points, gt_points_local)
            self._update_eval_history(eval_metrics)
            
            # Side-by-side comparison
            self._visualize_comparison(
                pred_points, gt_points_local, epoch, epoch_dir
            )
        # Optional: also generate per-group comparisons (e.g., per sequence).
        if gt_groups is not None and len(pred_points) > 0:
            groups_dir = epoch_dir / 'gt_groups'
            groups_dir.mkdir(parents=True, exist_ok=True)
            group_rows: List[str] = []
            for name, pts in sorted(gt_groups.items(), key=lambda kv: kv[0]):
                if pts is None or len(pts) == 0:
                    continue
                if crop_mn is not None and crop_mx is not None:
                    m = np.all((pts >= crop_mn) & (pts <= crop_mx), axis=1)
                    pts = pts[m] if m.any() else pts
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
                out_dir = groups_dir / safe
                out_dir.mkdir(parents=True, exist_ok=True)
                m = self._compute_metrics(pred_points, pts)
                group_rows.append(
                    f"{name}\tchamfer={m.get('chamfer_distance', 0):.6f}\t"
                    f"acc5cm={m.get('accuracy_5cm', 0)*100:.3f}%\t"
                    f"fscore5cm={m.get('fscore_5cm', 0)*100:.3f}%\t"
                    f"pred/gt={m.get('point_ratio', 0):.3f}"
                )
                self._visualize_comparison(pred_points, pts, epoch, out_dir)

            if group_rows:
                with open(groups_dir / 'metrics_by_group.txt', 'w') as f:
                    f.write("Per-group GT comparisons (same prediction vs different GT subsets)\n")
                    f.write("name\tchamfer\tacc5cm\tfscore5cm\tpred/gt\n")
                    for row in group_rows:
                        f.write(row + "\n")
        elif len(pred_points) > 0:
            # Just visualize prediction (no GT available)
            print(f"  No ground truth available - visualizing prediction only")
            self._visualize_point_cloud(
                pred_points, 
                title=f'Reconstruction - Epoch {epoch}',
                save_path=epoch_dir / 'reconstruction.png'
            )
        
        # 3. Visualize SDF slices
        self._visualize_sdf_slices(epoch, epoch_dir, sample_batch=sample_batch)
        
        # 4. Visualize depth comparison if batch provided
        if sample_batch is not None:
            self._visualize_depth_comparison(sample_batch, epoch, epoch_dir)
        
        # 5. Plot training curves
        self._plot_training_curves(epoch_dir)
        
        # 6. Generate metric report
        self._save_metric_report(epoch, train_metrics, val_metrics, eval_metrics, epoch_dir)
        
        # 7. Create summary image
        self._create_summary_image(epoch, train_metrics, eval_metrics, epoch_dir)
        
        return eval_metrics
    
    def _extract_point_cloud(
        self,
        threshold: float = 0.0,
        sample_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract point cloud from neural SDF using marching cubes.
        
        The model operates in normalized [0, 1] coordinates, but we return 
        points in world coordinates for visualization and comparison.
        """
        self.model.eval()
        
        # Create 3D grid in normalized [0, 1] space (model's input space).
        # For conditional models, focus on a local region around the provided views.
        min_bound, max_bound = self.bounds  # default: [0,0,0] to [1,1,1]
        if sample_batch is not None and 'views_pose' in sample_batch and self.scene_bounds is not None:
            # Compute local bounds in world space around camera centers and map to normalized.
            cam_centers = sample_batch['views_pose'][0, :, :3, 3].detach().cpu().numpy()
            margin = 10.0
            mn_w = cam_centers.min(axis=0) - margin
            mx_w = cam_centers.max(axis=0) + margin
            mn_n = self._normalize_points(mn_w[None, :])[0]
            mx_n = self._normalize_points(mx_w[None, :])[0]
            min_bound = np.maximum(min_bound, np.clip(mn_n, 0.0, 1.0))
            max_bound = np.minimum(max_bound, np.clip(mx_n, 0.0, 1.0))
        x = np.linspace(min_bound[0], max_bound[0], self.grid_resolution)
        y = np.linspace(min_bound[1], max_bound[1], self.grid_resolution)
        z = np.linspace(min_bound[2], max_bound[2], self.grid_resolution)
        
        # Query SDF in batches
        sdf_grid = np.zeros((self.grid_resolution, self.grid_resolution, self.grid_resolution))
        
        batch_size = 64 * 64 * 64  # Process in chunks
        
        with torch.no_grad():
            # Conditional models need view conditioning for meaningful evaluation.
            cond_kwargs = {}
            views_feats = None
            if sample_batch is not None and 'views_rgb' in sample_batch and hasattr(self.model, 'encode_views'):
                views_rgb = sample_batch['views_rgb'].to(self.device)
                views_pose = sample_batch['views_pose'].to(self.device)
                views_K = sample_batch['views_K'].to(self.device)
                views_feats = self.model.encode_views(views_rgb)
                cond_kwargs = dict(
                    views_rgb=views_rgb,
                    views_pose=views_pose,
                    views_K=views_K,
                    views_feats=views_feats,
                )

            for i, xi in enumerate(x):
                # Create grid slice
                yy, zz = np.meshgrid(y, z, indexing='ij')
                xx = np.full_like(yy, xi)
                
                points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
                points_tensor = torch.from_numpy(points).float().to(self.device)
                
                # Query model (expects normalized coords); conditional models also need world coords.
                if cond_kwargs:
                    world_pts = torch.from_numpy(self._denormalize_points(points)).float().to(self.device)
                    output = self.model(points_tensor.unsqueeze(0), world_coords=world_pts.unsqueeze(0), **cond_kwargs)
                else:
                    output = self.model(points_tensor.unsqueeze(0))
                sdf = output['sdf'].squeeze().cpu().numpy()
                
                sdf_grid[i] = sdf.reshape(self.grid_resolution, self.grid_resolution)
        
        # Prefer marching cubes for a surface-aligned extraction (more stable than
        # thresholding |SDF| on a grid, especially when SDF magnitudes are small).
        try:
            from skimage.measure import marching_cubes
        except Exception:
            marching_cubes = None

        if marching_cubes is not None:
            try:
                verts, _faces, _normals, values = marching_cubes(sdf_grid, level=0.0)
                # verts are in voxel coordinates [0, res-1] in (x, y, z) order.
                denom = max(self.grid_resolution - 1, 1)
                points_normalized = np.stack(
                    [
                        min_bound[0] + (verts[:, 0] / denom) * (max_bound[0] - min_bound[0]),
                        min_bound[1] + (verts[:, 1] / denom) * (max_bound[1] - min_bound[1]),
                        min_bound[2] + (verts[:, 2] / denom) * (max_bound[2] - min_bound[2]),
                    ],
                    axis=-1,
                )
                points = self._denormalize_points(points_normalized)
                return points, values
            except Exception as e:
                print(f"  [Viz] Marching cubes failed ({e}); falling back to thresholding.")

        # Fallback: extract points where |SDF| is small.
        surface_mask = np.abs(sdf_grid) < 0.005
        indices = np.where(surface_mask)
        if len(indices[0]) == 0:
            surface_mask = np.abs(sdf_grid) < 0.01
            indices = np.where(surface_mask)
            if len(indices[0]) == 0:
                return np.zeros((0, 3)), np.zeros((0,))

        points_normalized = np.stack(
            [
                min_bound[0] + indices[0] * (max_bound[0] - min_bound[0]) / self.grid_resolution,
                min_bound[1] + indices[1] * (max_bound[1] - min_bound[1]) / self.grid_resolution,
                min_bound[2] + indices[2] * (max_bound[2] - min_bound[2]) / self.grid_resolution,
            ],
            axis=-1,
        )
        points = self._denormalize_points(points_normalized)
        sdf_values = sdf_grid[indices]
        return points, sdf_values
    
    def _compute_metrics(
        self, 
        pred: np.ndarray, 
        gt: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics between predicted and GT point clouds."""
        from sklearn.neighbors import NearestNeighbors
        
        metrics = {}
        
        # Subsample for efficiency
        max_points = 50000
        if len(pred) > max_points:
            indices = np.random.choice(len(pred), max_points, replace=False)
            pred_sub = pred[indices]
        else:
            pred_sub = pred
            
        if len(gt) > max_points:
            indices = np.random.choice(len(gt), max_points, replace=False)
            gt_sub = gt[indices]
        else:
            gt_sub = gt
        
        # Chamfer Distance
        nn_gt = NearestNeighbors(n_neighbors=1).fit(gt_sub)
        nn_pred = NearestNeighbors(n_neighbors=1).fit(pred_sub)
        
        dist_pred_to_gt, _ = nn_gt.kneighbors(pred_sub)
        dist_gt_to_pred, _ = nn_pred.kneighbors(gt_sub)
        
        chamfer = (np.mean(dist_pred_to_gt**2) + np.mean(dist_gt_to_pred**2)) / 2
        metrics['chamfer_distance'] = float(chamfer)
        
        # Accuracy at different thresholds
        for threshold, name in [(0.01, '1cm'), (0.05, '5cm'), (0.1, '10cm')]:
            accuracy = np.mean(dist_pred_to_gt < threshold)
            metrics[f'accuracy_{name}'] = float(accuracy)
        
        # Completeness (fraction of GT covered)
        for threshold, name in [(0.01, '1cm'), (0.05, '5cm')]:
            completeness = np.mean(dist_gt_to_pred < threshold)
            metrics[f'completeness_{name}'] = float(completeness)
        
        # F-score (harmonic mean of accuracy and completeness)
        for threshold in ['1cm', '5cm']:
            acc = metrics[f'accuracy_{threshold}']
            comp = metrics[f'completeness_{threshold}']
            if acc + comp > 0:
                fscore = 2 * acc * comp / (acc + comp)
            else:
                fscore = 0.0
            metrics[f'fscore_{threshold}'] = float(fscore)
        
        # Point count ratio
        metrics['point_count_pred'] = len(pred)
        metrics['point_count_gt'] = len(gt)
        metrics['point_ratio'] = len(pred) / max(len(gt), 1)
        
        return metrics
    
    def _visualize_comparison(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        epoch: int,
        save_dir: Path,
    ):
        """Create side-by-side point cloud comparison from multiple angles."""
        fig = plt.figure(figsize=(20, 15))
        
        # Define viewing angles
        angles = [
            (30, 45, 'Front-Right'),
            (30, 135, 'Front-Left'),
            (90, 0, 'Top'),
            (0, 0, 'Front'),
            (0, 90, 'Side'),
        ]
        
        # Subsample for visualization
        max_viz = 20000
        pred_viz = pred[np.random.choice(len(pred), min(max_viz, len(pred)), replace=False)] if len(pred) > 0 else pred
        gt_viz = gt[np.random.choice(len(gt), min(max_viz, len(gt)), replace=False)] if len(gt) > 0 else gt
        
        for i, (elev, azim, title) in enumerate(angles):
            # Ground truth
            ax1 = fig.add_subplot(3, len(angles), i + 1, projection='3d')
            if len(gt_viz) > 0:
                ax1.scatter(gt_viz[:, 0], gt_viz[:, 1], gt_viz[:, 2], 
                           c='blue', s=0.5, alpha=0.5)
            ax1.set_title(f'GT - {title}')
            ax1.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax1, gt_viz if len(gt_viz) > 0 else pred_viz)
            
            # Prediction
            ax2 = fig.add_subplot(3, len(angles), len(angles) + i + 1, projection='3d')
            if len(pred_viz) > 0:
                ax2.scatter(pred_viz[:, 0], pred_viz[:, 1], pred_viz[:, 2], 
                           c='red', s=0.5, alpha=0.5)
            ax2.set_title(f'Pred - {title}')
            ax2.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax2, gt_viz if len(gt_viz) > 0 else pred_viz)
            
            # Overlay
            ax3 = fig.add_subplot(3, len(angles), 2*len(angles) + i + 1, projection='3d')
            if len(gt_viz) > 0:
                ax3.scatter(gt_viz[:, 0], gt_viz[:, 1], gt_viz[:, 2], 
                           c='blue', s=0.5, alpha=0.3, label='GT')
            if len(pred_viz) > 0:
                ax3.scatter(pred_viz[:, 0], pred_viz[:, 1], pred_viz[:, 2], 
                           c='red', s=0.5, alpha=0.3, label='Pred')
            ax3.set_title(f'Overlay - {title}')
            ax3.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax3, gt_viz if len(gt_viz) > 0 else pred_viz)
        
        plt.suptitle(f'Point Cloud Comparison - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'comparison_multiview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save individual higher-quality images
        self._save_individual_views(pred_viz, gt_viz, epoch, save_dir)
    
    def _save_individual_views(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        epoch: int,
        save_dir: Path,
    ):
        """Save individual high-quality view images."""
        for view_name, (elev, azim) in [
            ('front', (30, 45)),
            ('top', (90, 0)),
            ('side', (0, 90)),
        ]:
            # Combined view
            fig = plt.figure(figsize=(12, 5))
            
            ax1 = fig.add_subplot(121, projection='3d')
            if len(gt) > 0:
                ax1.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', s=1, alpha=0.5)
            ax1.set_title('Ground Truth')
            ax1.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax1, gt if len(gt) > 0 else pred)
            
            ax2 = fig.add_subplot(122, projection='3d')
            if len(pred) > 0:
                ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', s=1, alpha=0.5)
            ax2.set_title(f'Prediction (Epoch {epoch})')
            ax2.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax2, gt if len(gt) > 0 else pred)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'comparison_{view_name}.png', dpi=200, bbox_inches='tight')
            plt.close()
    
    def _visualize_point_cloud(
        self,
        points: np.ndarray,
        title: str,
        save_path: Path,
    ):
        """Visualize a single point cloud."""
        fig = plt.figure(figsize=(15, 5))
        
        max_viz = 20000
        if len(points) > max_viz:
            points = points[np.random.choice(len(points), max_viz, replace=False)]
        
        for i, (elev, azim) in enumerate([(30, 45), (90, 0), (0, 90)]):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, alpha=0.5)
            ax.view_init(elev=elev, azim=azim)
            self._set_axes_equal(ax, points)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_sdf_slices(self, epoch: int, save_dir: Path, sample_batch: Optional[Dict[str, torch.Tensor]] = None):
        """Visualize 2D slices through the SDF volume."""
        self.model.eval()
        
        min_bound, max_bound = self.bounds
        if sample_batch is not None and 'views_pose' in sample_batch and self.scene_bounds is not None:
            cam_centers = sample_batch['views_pose'][0, :, :3, 3].detach().cpu().numpy()
            margin = 10.0
            mn_w = cam_centers.min(axis=0) - margin
            mx_w = cam_centers.max(axis=0) + margin
            mn_n = self._normalize_points(mn_w[None, :])[0]
            mx_n = self._normalize_points(mx_w[None, :])[0]
            min_bound = np.maximum(min_bound, np.clip(mn_n, 0.0, 1.0))
            max_bound = np.minimum(max_bound, np.clip(mx_n, 0.0, 1.0))
        resolution = 256
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        slice_positions = [0.25, 0.5, 0.75]  # Relative positions
        
        with torch.no_grad():
            cond_kwargs = {}
            if sample_batch is not None and 'views_rgb' in sample_batch and hasattr(self.model, 'encode_views'):
                views_rgb = sample_batch['views_rgb'].to(self.device)
                views_pose = sample_batch['views_pose'].to(self.device)
                views_K = sample_batch['views_K'].to(self.device)
                views_feats = self.model.encode_views(views_rgb)
                cond_kwargs = dict(
                    views_rgb=views_rgb,
                    views_pose=views_pose,
                    views_K=views_K,
                    views_feats=views_feats,
                )

            # XY slices (varying Z)
            for col, z_rel in enumerate(slice_positions):
                z = min_bound[2] + z_rel * (max_bound[2] - min_bound[2])
                
                x = np.linspace(min_bound[0], max_bound[0], resolution)
                y = np.linspace(min_bound[1], max_bound[1], resolution)
                xx, yy = np.meshgrid(x, y, indexing='ij')
                zz = np.full_like(xx, z)
                
                points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
                points_tensor = torch.from_numpy(points).float().to(self.device)
                
                if cond_kwargs:
                    world_pts = torch.from_numpy(self._denormalize_points(points)).float().to(self.device)
                    output = self.model(points_tensor.unsqueeze(0), world_coords=world_pts.unsqueeze(0), **cond_kwargs)
                else:
                    output = self.model(points_tensor.unsqueeze(0))
                sdf = output['sdf'].squeeze().cpu().numpy().reshape(resolution, resolution)
                
                ax = axes[0, col]
                im = ax.imshow(sdf.T, extent=[min_bound[0], max_bound[0], min_bound[1], max_bound[1]],
                              origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5)
                ax.contour(xx, yy, sdf, levels=[0], colors='black', linewidths=2)
                ax.set_title(f'XY Slice (Z={z:.2f}m)')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='SDF')
            
            # XZ slices (varying Y)
            for col, y_rel in enumerate(slice_positions):
                y = min_bound[1] + y_rel * (max_bound[1] - min_bound[1])
                
                x = np.linspace(min_bound[0], max_bound[0], resolution)
                z = np.linspace(min_bound[2], max_bound[2], resolution)
                xx, zz = np.meshgrid(x, z, indexing='ij')
                yy = np.full_like(xx, y)
                
                points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
                points_tensor = torch.from_numpy(points).float().to(self.device)
                
                if cond_kwargs:
                    world_pts = torch.from_numpy(self._denormalize_points(points)).float().to(self.device)
                    output = self.model(points_tensor.unsqueeze(0), world_coords=world_pts.unsqueeze(0), **cond_kwargs)
                else:
                    output = self.model(points_tensor.unsqueeze(0))
                sdf = output['sdf'].squeeze().cpu().numpy().reshape(resolution, resolution)
                
                ax = axes[1, col]
                im = ax.imshow(sdf.T, extent=[min_bound[0], max_bound[0], min_bound[2], max_bound[2]],
                              origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5)
                ax.contour(xx, zz, sdf, levels=[0], colors='black', linewidths=2)
                ax.set_title(f'XZ Slice (Y={y:.2f}m)')
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                plt.colorbar(im, ax=ax, label='SDF')
        
        plt.suptitle(f'SDF Slices - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / 'sdf_slices.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_depth_comparison(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
        save_dir: Path,
    ):
        """Compare input depth with rendered depth from SDF."""
        if 'depth' in batch:
            depth_gt = batch['depth'][0].cpu().numpy()
        elif 'views_depth' in batch:
            depth_gt = batch['views_depth'][0, 0].cpu().numpy()
        else:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth depth
        valid_mask = depth_gt > 0
        vmin, vmax = depth_gt[valid_mask].min(), depth_gt[valid_mask].max() if valid_mask.any() else (0, 10)
        
        im1 = axes[0].imshow(depth_gt, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Input Depth')
        plt.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Valid mask
        axes[1].imshow(valid_mask, cmap='gray')
        axes[1].set_title(f'Valid Mask ({valid_mask.sum()} pixels)')
        
        # Depth histogram
        axes[2].hist(depth_gt[valid_mask].ravel(), bins=50, color='blue', alpha=0.7)
        axes[2].set_xlabel('Depth (m)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Depth Distribution')
        
        plt.suptitle(f'Depth Analysis - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(save_dir / 'depth_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _update_history(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ):
        """Update metric history for plotting."""
        self.metric_history['epoch'].append(epoch)
        self.metric_history['train_loss'].append(train_metrics.get('loss', 0))
        self.metric_history['val_loss'].append(val_metrics.get('loss', 0) if val_metrics else 0)
        self.metric_history['surface_loss'].append(train_metrics.get('surface', 0))
        self.metric_history['freespace_loss'].append(train_metrics.get('freespace', 0))
        self.metric_history['eikonal_loss'].append(train_metrics.get('eikonal', 0))
    
    def _update_eval_history(self, eval_metrics: Dict[str, float]):
        """Update evaluation metric history."""
        self.metric_history['chamfer_distance'].append(eval_metrics.get('chamfer_distance', 0))
        self.metric_history['accuracy_1cm'].append(eval_metrics.get('accuracy_1cm', 0))
        self.metric_history['accuracy_5cm'].append(eval_metrics.get('accuracy_5cm', 0))
        self.metric_history['completeness'].append(eval_metrics.get('completeness_5cm', 0))
    
    def _plot_training_curves(self, save_dir: Path):
        """Plot training curves over epochs."""
        if len(self.metric_history['epoch']) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = self.metric_history['epoch']
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.metric_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if any(self.metric_history['val_loss']):
            ax.plot(epochs, self.metric_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Loss components
        ax = axes[0, 1]
        if any(self.metric_history['surface_loss']):
            ax.plot(epochs, self.metric_history['surface_loss'], label='Surface', linewidth=2)
        if any(self.metric_history['freespace_loss']):
            ax.plot(epochs, self.metric_history['freespace_loss'], label='Freespace', linewidth=2)
        if any(self.metric_history['eikonal_loss']):
            ax.plot(epochs, self.metric_history['eikonal_loss'], label='Eikonal', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Chamfer distance
        ax = axes[1, 0]
        if any(self.metric_history['chamfer_distance']):
            ax.plot(epochs, self.metric_history['chamfer_distance'], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Chamfer Distance')
            ax.set_title('Chamfer Distance (lower is better)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No GT comparison available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Accuracy & Completeness
        ax = axes[1, 1]
        if any(self.metric_history['accuracy_1cm']):
            ax.plot(epochs, self.metric_history['accuracy_1cm'], label='Accuracy @1cm', linewidth=2)
            ax.plot(epochs, self.metric_history['accuracy_5cm'], label='Accuracy @5cm', linewidth=2)
            ax.plot(epochs, self.metric_history['completeness'], label='Completeness @5cm', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Fraction')
            ax.set_title('Accuracy & Completeness (higher is better)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, 'No GT comparison available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'training_curves_latest.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_metric_report(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        eval_metrics: Dict[str, float],
        save_dir: Path,
    ):
        """Save detailed metric report as text file."""
        report_path = save_dir / 'metrics_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"=" * 60 + "\n")
            f.write(f"EPOCH {epoch} METRICS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 60 + "\n\n")
            
            f.write("TRAINING METRICS\n")
            f.write("-" * 40 + "\n")
            for key, value in sorted(train_metrics.items()):
                f.write(f"  {key:25s}: {value:.6f}\n")
            f.write("\n")
            
            if val_metrics:
                f.write("VALIDATION METRICS\n")
                f.write("-" * 40 + "\n")
                for key, value in sorted(val_metrics.items()):
                    f.write(f"  {key:25s}: {value:.6f}\n")
                f.write("\n")
            
            if eval_metrics:
                f.write("RECONSTRUCTION QUALITY METRICS\n")
                f.write("-" * 40 + "\n")
                for key, value in sorted(eval_metrics.items()):
                    if isinstance(value, float):
                        f.write(f"  {key:25s}: {value:.6f}\n")
                    else:
                        f.write(f"  {key:25s}: {value}\n")
                f.write("\n")
                
                # Interpretation
                f.write("INTERPRETATION\n")
                f.write("-" * 40 + "\n")
                
                cd = eval_metrics.get('chamfer_distance', 0)
                if cd < 0.001:
                    f.write("  Chamfer Distance: EXCELLENT (<0.001)\n")
                elif cd < 0.01:
                    f.write("  Chamfer Distance: GOOD (<0.01)\n")
                elif cd < 0.1:
                    f.write("  Chamfer Distance: FAIR (<0.1)\n")
                else:
                    f.write("  Chamfer Distance: POOR (>0.1)\n")
                
                acc = eval_metrics.get('accuracy_5cm', 0)
                if acc > 0.9:
                    f.write("  Accuracy @5cm: EXCELLENT (>90%)\n")
                elif acc > 0.7:
                    f.write("  Accuracy @5cm: GOOD (>70%)\n")
                elif acc > 0.5:
                    f.write("  Accuracy @5cm: FAIR (>50%)\n")
                else:
                    f.write("  Accuracy @5cm: POOR (<50%)\n")
    
    def _create_summary_image(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        save_dir: Path,
    ):
        """Create a single summary image with key visualizations and metrics."""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Training Summary - Epoch {epoch}', fontsize=18, fontweight='bold')
        
        # Metrics panel (text)
        ax_metrics = fig.add_subplot(gs[0, 0])
        ax_metrics.axis('off')
        
        metrics_text = f"Training Metrics\n"
        metrics_text += f"─" * 25 + "\n"
        metrics_text += f"Total Loss: {train_metrics.get('loss', 0):.4f}\n"
        metrics_text += f"Surface: {train_metrics.get('surface', 0):.4f}\n"
        metrics_text += f"Freespace: {train_metrics.get('freespace', 0):.4f}\n"
        metrics_text += f"Eikonal: {train_metrics.get('eikonal', 0):.4f}\n"
        
        if eval_metrics:
            metrics_text += f"\nReconstruction Quality\n"
            metrics_text += f"─" * 25 + "\n"
            metrics_text += f"Chamfer: {eval_metrics.get('chamfer_distance', 0):.4f}\n"
            metrics_text += f"Acc@5cm: {eval_metrics.get('accuracy_5cm', 0)*100:.1f}%\n"
            metrics_text += f"Comp@5cm: {eval_metrics.get('completeness_5cm', 0)*100:.1f}%\n"
            metrics_text += f"F-score: {eval_metrics.get('fscore_5cm', 0)*100:.1f}%\n"
        
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Load and display saved visualizations if they exist
        for idx, (name, pos) in enumerate([
            ('comparison_front.png', gs[0, 1:3]),
            ('comparison_top.png', gs[0, 3]),
            ('sdf_slices.png', gs[1, :2]),
            ('training_curves.png', gs[1, 2:]),
            ('depth_analysis.png', gs[2, :2]),
        ]):
            img_path = save_dir / name
            if img_path.exists():
                ax = fig.add_subplot(pos)
                img = plt.imread(str(img_path))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(name.replace('.png', '').replace('_', ' ').title())
        
        # Progress bar
        ax_progress = fig.add_subplot(gs[2, 2:])
        ax_progress.axis('off')
        
        progress_text = f"Training Progress\n"
        progress_text += f"Epoch {epoch} / {self.metric_history['epoch'][-1] if self.metric_history['epoch'] else '?'}\n"
        if len(self.metric_history['train_loss']) > 1:
            initial_loss = self.metric_history['train_loss'][0]
            current_loss = self.metric_history['train_loss'][-1]
            improvement = (initial_loss - current_loss) / initial_loss * 100
            progress_text += f"Loss improvement: {improvement:.1f}%\n"
        
        ax_progress.text(0.1, 0.8, progress_text, transform=ax_progress.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.savefig(save_dir / 'summary.png', dpi=150, bbox_inches='tight')
        plt.savefig(self.viz_dir / 'summary_latest.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _set_axes_equal(self, ax, points: np.ndarray):
        """Set equal aspect ratio for 3D plot."""
        if len(points) == 0:
            return
            
        max_range = np.max(points.max(axis=0) - points.min(axis=0)) / 2
        mid = points.mean(axis=0)
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


def generate_gt_point_cloud_from_batch(
    batch: Dict[str, torch.Tensor],
    num_samples: int = 50000,
) -> np.ndarray:
    """
    Generate ground truth point cloud from a batch of depth images.
    
    Args:
        batch: Batch containing depth, pose, K
        num_samples: Number of points to sample
        
    Returns:
        Point cloud as numpy array [N, 3]
    """
    all_points = []
    
    depth = batch['depth']  # (B, H, W)
    pose = batch['pose']    # (B, 4, 4)
    K = batch['K']          # (B, 3, 3)
    valid_mask = batch.get('valid_mask', depth > 0)
    
    B, H, W = depth.shape
    
    for b in range(B):
        # Get valid pixels
        valid = valid_mask[b]
        v, u = torch.where(valid)
        
        if len(v) == 0:
            continue
            
        z = depth[b, v, u]
        
        # Back-project
        fx, fy = K[b, 0, 0], K[b, 1, 1]
        cx, cy = K[b, 0, 2], K[b, 1, 2]
        
        x = (u.float() - cx) * z / fx
        y = (v.float() - cy) * z / fy
        
        pts_cam = torch.stack([x, y, z], dim=-1)
        
        # Transform to world
        R = pose[b, :3, :3]
        t = pose[b, :3, 3]
        pts_world = pts_cam @ R.T + t
        
        all_points.append(pts_world.cpu().numpy())
    
    if not all_points:
        return np.zeros((0, 3))
        
    points = np.concatenate(all_points, axis=0)
    
    # Subsample if needed
    if len(points) > num_samples:
        indices = np.random.choice(len(points), num_samples, replace=False)
        points = points[indices]
    
    return points
