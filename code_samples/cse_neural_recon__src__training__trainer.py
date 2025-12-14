"""
Main training loop and trainer class.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# Productivity metrics for DGX Spark evaluation
try:
    from ..utils.productivity_metrics import ProductivityMetricsCollector
    HAS_PRODUCTIVITY_METRICS = True
except ImportError:
    HAS_PRODUCTIVITY_METRICS = False
    ProductivityMetricsCollector = None


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    
    # Basic settings
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    
    # Sampling
    num_surface_samples: int = 8192
    num_freespace_samples: int = 8192
    num_random_samples: int = 4096  # Random points in empty space
    freespace_jitter_min: float = 0.1
    freespace_jitter_max: float = 2.0  # Increased range for large scenes
    
    # Loss weights
    surface_weight: float = 1.0
    freespace_weight: float = 2.0  # Increased to better learn empty space
    eikonal_weight: float = 0.05  # Reduced - was dominating
    random_space_weight: float = 1.0  # For random space sampling
    planar_weight: float = 0.3
    normal_weight: float = 0.2
    manhattan_weight: float = 0.1
    
    # Training options
    use_amp: bool = True
    grad_clip: float = 1.0
    
    # Metrics collection (for cross-device comparison)
    collect_metrics: bool = True  # Enable detailed metrics collection by default
    metrics_sample_rate: int = 10  # Sample GPU metrics every N batches
    
    # Logging
    log_every: int = 100
    val_every: int = 1
    save_every: int = 5
    visualize_every: int = 5  # Generate visualizations every N epochs
    
    # Output
    output_dir: str = "output"
    experiment_name: str = "neural_sdf"


class Trainer:
    """
    Main trainer class for neural 3D reconstruction.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: Neural SDF model
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: Training configuration
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Compute scene bounds from dataset for coordinate normalization
        self._compute_scene_bounds()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Logging
        self.writer = None
        if HAS_TENSORBOARD:
            log_dir = os.path.join(self.config.output_dir, 'logs', self.config.experiment_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
        # Checkpointing
        self.checkpoint_dir = os.path.join(
            self.config.output_dir, 'checkpoints', self.config.experiment_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Visualization
        self.visualizer = None
        self._init_visualizer()
        
        # Cache a sample batch for visualization
        self._viz_batch = None
        self._gt_points = None
        
        # Productivity metrics collection (enabled by default for cross-device comparison)
        self.productivity_collector = None
        if self.config.collect_metrics and HAS_PRODUCTIVITY_METRICS and ProductivityMetricsCollector is not None:
            metrics_dir = os.path.join(self.config.output_dir, 'metrics', self.config.experiment_name)
            self.productivity_collector = ProductivityMetricsCollector(
                output_dir=metrics_dir,
                sample_rate=self.config.metrics_sample_rate
            )
            print(f"[Metrics] Detailed metrics collection enabled (sample rate: every {self.config.metrics_sample_rate} batches)")
            print(f"[Metrics] Reports will be saved to: {metrics_dir}")
    
    def _compute_scene_bounds(self):
        """
        Compute scene bounds from actual depth points in dataset.
        
        If the dataset has a UnifiedCoordinateSystem attached (from multi-environment
        loading), use that for per-environment normalization. Otherwise, compute
        bounds by sampling depth points.
        
        This sets up normalization parameters to map coords to [0, 1] or [-1, 1].
        """
        # Check if dataset has a coordinate system attached (multi-environment mode)
        dataset = self.train_loader.dataset
        
        if hasattr(dataset, 'coordinate_system') and dataset.coordinate_system is not None:
            print("Using unified coordinate system from dataset")
            self.coordinate_system = dataset.coordinate_system
            self.use_unified_coords = True
            
            # For compatibility, set scene_min/max from global bounds
            global_bounds = self.coordinate_system.global_bounds
            self.scene_min = torch.tensor(global_bounds.min_bounds, device=self.device).float()
            self.scene_max = torch.tensor(global_bounds.max_bounds, device=self.device).float()
            
            print(f"Global scene bounds from {len(self.coordinate_system.environments)} environments:")
            for env_name, bounds in self.coordinate_system.environments.items():
                print(f"  {env_name}: center=({bounds.center[0]:.1f}, {bounds.center[1]:.1f}, {bounds.center[2]:.1f}), scale={bounds.scale:.1f}m")
        else:
            print("Computing scene bounds from depth data...")
            self.coordinate_system = None
            self.use_unified_coords = False
            self._compute_scene_bounds_from_depth()
        
        scene_extent = self.scene_max - self.scene_min
        print(f"Scene bounds: min={self.scene_min.cpu().numpy()}, max={self.scene_max.cpu().numpy()}")
        print(f"Scene extent: {scene_extent.cpu().numpy()}")
    
    def _compute_scene_bounds_from_depth(self):
        """
        Compute scene bounds by backprojecting depth samples.
        """
        # Collect backprojected points from depth maps
        all_points = []
        
        # Sample a subset of batches
        num_samples = min(50, len(self.train_loader))
        
        for i, batch in enumerate(self.train_loader):
            if i >= num_samples:
                break
            
            depth = batch['depth']  # (B, H, W)
            pose = batch['pose']    # (B, 4, 4)
            K = batch['K']          # (B, 3, 3)
            
            B, H, W = depth.shape
            
            # Create pixel grid
            v, u = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing='ij'
            )
            
            for b in range(B):
                d = depth[b]
                valid = (d > 0.1) & (d < 30.0)  # Valid depth range
                
                if valid.sum() < 100:
                    continue
                
                # Subsample for efficiency
                valid_idx = torch.where(valid.flatten())[0]
                sample_idx = valid_idx[torch.randperm(len(valid_idx))[:1000]]
                
                # Get intrinsics
                fx, fy = K[b, 0, 0], K[b, 1, 1]
                cx, cy = K[b, 0, 2], K[b, 1, 2]
                
                # Backproject sampled pixels
                u_flat, v_flat = u.flatten(), v.flatten()
                d_flat = d.flatten()
                
                u_s = u_flat[sample_idx]
                v_s = v_flat[sample_idx]
                d_s = d_flat[sample_idx]
                
                X = (u_s - cx) * d_s / fx
                Y = (v_s - cy) * d_s / fy
                Z = d_s
                
                pts_cam = torch.stack([X, Y, Z], dim=-1)
                
                # Transform to world
                R = pose[b, :3, :3]
                t = pose[b, :3, 3]
                pts_world = (R @ pts_cam.T).T + t
                
                all_points.append(pts_world)
        
        if all_points:
            all_points = torch.cat(all_points, dim=0)  # (N, 3)
            
            # Compute bounds with small padding
            min_pt = all_points.min(dim=0)[0]
            max_pt = all_points.max(dim=0)[0]
            
            # Add 5% padding
            extent = max_pt - min_pt
            padding = extent * 0.05
            
            self.scene_min = (min_pt - padding).to(self.device)
            self.scene_max = (max_pt + padding).to(self.device)
        else:
            # Fallback to default bounds
            self.scene_min = torch.tensor([-50.0, -50.0, -15.0], device=self.device)
            self.scene_max = torch.tensor([50.0, 50.0, 15.0], device=self.device)
    
    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize world coordinates to [0, 1] range for the hash grid.
        
        Args:
            coords: (*, 3) world coordinates
            
        Returns:
            (*, 3) normalized coordinates in [0, 1]
        """
        # Map from [scene_min, scene_max] to [0, 1]
        normalized = (coords - self.scene_min) / (self.scene_max - self.scene_min + 1e-8)
        # Clamp to handle points slightly outside bounds
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized
    
    def _denormalize_coords(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized [0, 1] coordinates back to world coordinates.
        
        Args:
            normalized: (*, 3) normalized coordinates in [0, 1]
            
        Returns:
            (*, 3) world coordinates
        """
        return normalized * (self.scene_max - self.scene_min) + self.scene_min
        
    def _init_visualizer(self):
        """Initialize the training visualizer."""
        try:
            from .visualizer import TrainingVisualizer
            self.visualizer = TrainingVisualizer(
                output_dir=self.config.output_dir,
                model=self.model,
                device=self.device,
            )
            # Pass scene bounds to visualizer if available
            if hasattr(self, 'scene_min') and hasattr(self, 'scene_max'):
                self.visualizer.set_scene_bounds(self.scene_min, self.scene_max)
            print("Visualization enabled - will generate point cloud comparisons")
        except Exception as e:
            print(f"Warning: Could not initialize visualizer: {e}")
            self.visualizer = None
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
            
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of final metrics
        """
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Start productivity metrics collection
        if self.productivity_collector is not None:
            config_dict = {
                'training': {
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                    'epochs': self.config.epochs,
                }
            }
            self.productivity_collector.start_experiment(
                self.model, 
                config=config_dict,
                model_name=self.model.__class__.__name__
            )
        
        # Cache visualization batch on first iteration
        self._cache_viz_batch()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Track epoch timing
            if self.productivity_collector is not None:
                self.productivity_collector.start_epoch(epoch)
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # End epoch timing
            if self.productivity_collector is not None:
                self.productivity_collector.end_epoch(epoch)
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                
            # Logging
            self._log_epoch(train_metrics, val_metrics)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
                    
            # Checkpointing and Visualization
            if (epoch + 1) % self.config.save_every == 0:
                ckpt_start = time.time()
                self.save_checkpoint()
                if self.productivity_collector is not None:
                    self.productivity_collector.record_event('checkpoint_save', time.time() - ckpt_start)
                
                # Generate visualizations
                if self.visualizer is not None:
                    viz_start = time.time()
                    eval_metrics = self._generate_visualizations(epoch, train_metrics, val_metrics)
                    if self.productivity_collector is not None:
                        self.productivity_collector.record_event('visualization', time.time() - viz_start)
                        if epoch == self.config.save_every - 1 and self.productivity_collector.experiment_start:
                            # First visualization - record time to first result
                            self.productivity_collector.record_event('first_result', time.time() - self.productivity_collector.experiment_start)
                        # Record quality metrics
                        if eval_metrics and isinstance(eval_metrics, dict):
                            for key, value in eval_metrics.items():
                                self.productivity_collector.record_quality_metric(key, value)
                
            # Best model
            val_loss = val_metrics.get('loss', train_metrics['loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
        
        # Record final quality metrics
        if self.productivity_collector is not None:
            self.productivity_collector.record_quality_metric('loss', train_metrics.get('loss', 0))
                
        # Final save
        self.save_checkpoint('final_model.pt')
        
        # End productivity metrics and generate report
        if self.productivity_collector is not None:
            self.productivity_collector.end_experiment()
            report = self.productivity_collector.generate_report()
            self.productivity_collector.save_report(report, f"productivity_{self.config.experiment_name}")
            print(f"\n[Productivity] Report saved to {self.productivity_collector.output_dir}")
        
        if self.writer:
            self.writer.close()
            
        return {'best_val_loss': self.best_val_loss}
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Start batch timing for productivity metrics
            if self.productivity_collector is not None:
                self.productivity_collector.start_batch()
                self.productivity_collector.start_phase('data_load')
            
            # Move batch to device
            batch = self._to_device(batch)
            
            if self.productivity_collector is not None:
                self.productivity_collector.start_phase('forward')
            
            # Forward pass
            loss, components = self.train_step(batch)
            
            # End batch timing
            if self.productivity_collector is not None:
                batch_size = batch['depth'].shape[0] if 'depth' in batch else self.config.batch_size
                self.productivity_collector.end_batch(
                    self.current_epoch, batch_idx, batch_size, loss.item()
                )
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Logging
            if (self.global_step + 1) % self.config.log_every == 0:
                self._log_step(loss.item(), components)
                
            self.global_step += 1
            
        # Average metrics
        avg_metrics = {'loss': total_loss / num_batches}
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
            
        return avg_metrics
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        Single training step with proper TSDF supervision.
        
        Implements depth-based SDF supervision following MonoSDF/NeuS approaches:
        - Sample points along camera rays at multiple depths
        - Compute truncated SDF targets from depth measurements
        - Use proper inside/outside supervision
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            loss: Total loss
            components: Dictionary of loss components
        """
        self.optimizer.zero_grad()
        
        # Extract data from batch
        depth = batch['depth']  # (B, H, W)
        pose = batch['pose']    # (B, 4, 4)
        K = batch['K']          # (B, 3, 3)
        valid_mask = batch.get('valid_mask', depth > 0)
        
        B, H, W = depth.shape
        
        # Scene scale for normalizing SDF targets
        scene_extent = self.scene_max - self.scene_min
        scene_scale = scene_extent.mean().item()
        
        # Truncation distance in NORMALIZED coordinates
        # Use 1% of scene size for truncation band
        truncation_normalized = 0.02  # 2% of normalized [0,1] space
        truncation_world = truncation_normalized * scene_scale
        
        # Sample TSDF points along rays (key to learning proper SDF)
        tsdf_points_world, tsdf_targets_world = self._sample_tsdf_points(
            depth, pose, K, valid_mask, truncation_world
        )
        
        # Sample surface points (SDF = 0)
        surface_points_world, camera_centers = self._sample_surface_points(
            depth, pose, K, valid_mask
        )
        
        # Sample random far-field points (positive SDF, far from surface)
        far_points_world = self._sample_random_space_points(B)
        
        # Normalize all coordinates to [0, 1]
        tsdf_points = self._normalize_coords(tsdf_points_world)
        surface_points = self._normalize_coords(surface_points_world)
        far_points = self._normalize_coords(far_points_world)
        
        # Normalize SDF targets to same scale as coordinates
        # This ensures TSDF targets are in [-truncation_normalized, +truncation_normalized]
        tsdf_targets = tsdf_targets_world / scene_scale
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp):
            # TSDF points (most important for geometry)
            tsdf_out = self.model(tsdf_points)
            sdf_tsdf = tsdf_out['sdf'].squeeze(-1)  # (B, N)
            
            # Surface points
            surface_out = self.model(surface_points)
            sdf_surface = surface_out['sdf']
            
            # Far-field points (should have large positive SDF)
            far_out = self.model(far_points)
            sdf_far = far_out['sdf']
            
            # === TSDF Loss (most important) ===
            # L1 loss on truncated SDF values
            loss_tsdf = torch.abs(sdf_tsdf - tsdf_targets).mean()
            
            # === Surface Loss ===
            # Surface points should have SDF = 0
            loss_surface = torch.abs(sdf_surface).mean()
            
            # === Far-field Loss ===
            # Far points should have positive SDF (outside surface)
            # Use same scale as TSDF - want far field to be beyond truncation
            min_far_sdf = truncation_normalized * 2  # Twice the truncation distance
            loss_far = torch.relu(min_far_sdf - sdf_far).mean()
        
        # === Eikonal Loss (outside AMP for stability) ===
        if self.config.eikonal_weight > 0:
            num_eikonal_points = min(2048, surface_points.shape[1])
            eikonal_points = surface_points[:, :num_eikonal_points].clone()
            eikonal_points.requires_grad_(True)
            
            eikonal_out = self.model(eikonal_points)
            eikonal_sdf = eikonal_out['sdf']
            
            gradients = torch.autograd.grad(
                outputs=eikonal_sdf,
                inputs=eikonal_points,
                grad_outputs=torch.ones_like(eikonal_sdf),
                create_graph=True,
                retain_graph=True
            )[0]
            
            grad_norm = torch.norm(gradients, dim=-1)
            grad_norm = torch.clamp(grad_norm, 0, 10)
            loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
        else:
            loss_eikonal = torch.tensor(0.0, device=self.device)
        
        # Total loss with proper weighting
        # TSDF is the primary supervision, surface is regularization
        loss = (
            2.0 * loss_tsdf +  # Primary TSDF supervision
            self.config.surface_weight * loss_surface +
            0.5 * loss_far +  # Encourage positive SDF in empty space
            self.config.eikonal_weight * loss_eikonal
        )
            
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                
            self.optimizer.step()
            
        components = {
            'tsdf': loss_tsdf.item(),
            'surface': loss_surface.item(),
            'far': loss_far.item(),
            'eikonal': loss_eikonal.item(),
        }
        
        return loss, components
    
    def _sample_tsdf_points(
        self,
        depth: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        valid_mask: torch.Tensor,
        truncation: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along camera rays with TSDF targets.
        
        For each valid depth pixel:
        - Sample points in front of and behind the surface
        - Compute SDF as signed distance to surface along ray
        - Truncate to [-truncation, truncation]
        
        Args:
            depth: (B, H, W) depth maps
            pose: (B, 4, 4) camera poses
            K: (B, 3, 3) intrinsics
            valid_mask: (B, H, W) valid depth mask
            truncation: Truncation distance in world units
            
        Returns:
            points: (B, N, 3) sampled points in world coordinates
            targets: (B, N) truncated SDF targets
        """
        B, H, W = depth.shape
        device = depth.device
        
        # Number of rays to sample per image
        num_rays = self.config.num_surface_samples // 4
        # Number of depth samples per ray
        num_depth_samples = 8
        
        all_points = []
        all_targets = []
        
        for b in range(B):
            # Get valid pixels
            valid_idx = torch.where(valid_mask[b].flatten())[0]
            
            if len(valid_idx) < num_rays:
                # Not enough valid pixels, sample with replacement
                ray_idx = valid_idx[torch.randint(len(valid_idx), (num_rays,), device=device)]
            else:
                ray_idx = valid_idx[torch.randperm(len(valid_idx), device=device)[:num_rays]]
            
            # Convert flat indices to 2D
            v = ray_idx // W
            u = ray_idx % W
            
            # Get depths at selected pixels
            ray_depths = depth[b, v, u]  # (num_rays,)
            
            # Get camera parameters
            K_b = K[b]  # (3, 3)
            pose_b = pose[b]  # (4, 4)
            R = pose_b[:3, :3]
            t = pose_b[:3, 3]
            
            # Compute ray directions in camera frame
            fx, fy = K_b[0, 0], K_b[1, 1]
            cx, cy = K_b[0, 2], K_b[1, 2]
            
            # Ray directions in camera frame
            ray_dirs_cam = torch.stack([
                (u.float() - cx) / fx,
                (v.float() - cy) / fy,
                torch.ones(num_rays, device=device)
            ], dim=-1)  # (num_rays, 3)
            ray_dirs_cam = ray_dirs_cam / torch.norm(ray_dirs_cam, dim=-1, keepdim=True)
            
            # Transform to world frame
            ray_dirs_world = (R @ ray_dirs_cam.T).T  # (num_rays, 3)
            
            # Sample depths: from (surface_depth - truncation) to (surface_depth + truncation)
            depth_offsets = torch.linspace(-truncation, truncation, num_depth_samples, device=device)
            
            # Sample points along each ray
            # ray_depths: (num_rays,), depth_offsets: (num_depth_samples,)
            sample_depths = ray_depths.unsqueeze(1) + depth_offsets.unsqueeze(0)  # (num_rays, num_depth_samples)
            sample_depths = torch.clamp(sample_depths, min=0.1)  # Avoid negative depths
            
            # Compute 3D points: camera_pos + depth * ray_dir
            # sample_depths: (num_rays, num_depth_samples)
            # ray_dirs_world: (num_rays, 3)
            points = t.unsqueeze(0).unsqueeze(0) + \
                     sample_depths.unsqueeze(-1) * ray_dirs_world.unsqueeze(1)
            # points: (num_rays, num_depth_samples, 3)
            
            # Compute SDF targets: signed distance to surface along ray
            # Negative = inside (behind surface), Positive = outside (in front of surface)
            # Note: SDF convention - positive outside, negative inside
            sdf_targets = -depth_offsets.unsqueeze(0).expand(num_rays, -1)  # (num_rays, num_depth_samples)
            
            # Truncate
            sdf_targets = torch.clamp(sdf_targets, -truncation, truncation)
            
            # Flatten
            points = points.view(-1, 3)  # (num_rays * num_depth_samples, 3)
            sdf_targets = sdf_targets.view(-1)  # (num_rays * num_depth_samples,)
            
            all_points.append(points)
            all_targets.append(sdf_targets)
        
        # Stack batches
        all_points = torch.stack([p for p in all_points], dim=0)  # (B, N, 3)
        all_targets = torch.stack([t for t in all_targets], dim=0)  # (B, N)
        
        return all_points, all_targets
        
        return loss, components
        
    def _sample_surface_points(
        self,
        depth: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> tuple:
        """Sample 3D surface points from depth maps."""
        B, H, W = depth.shape
        device = depth.device
        N_samples = self.config.num_surface_samples
        
        # Flatten for sampling
        depth_flat = depth.view(B, -1)
        valid_flat = valid_mask.view(B, -1)
        
        all_surface_points = []
        all_camera_centers = []
        
        for b in range(B):
            # Get valid indices
            valid_idx = torch.where(valid_flat[b])[0]
            
            if len(valid_idx) == 0:
                # No valid points - generate random points near origin
                pts_world = torch.randn(N_samples, 3, device=device) * 0.1
                t = pose[b, :3, 3] if pose is not None else torch.zeros(3, device=device)
                all_surface_points.append(pts_world)
                all_camera_centers.append(t.expand(N_samples, -1))
                continue
            
            if len(valid_idx) < N_samples:
                # Pad with repetition
                idx = valid_idx[torch.randint(len(valid_idx), (N_samples,), device=device)]
            else:
                # Random sample
                perm = torch.randperm(len(valid_idx), device=device)[:N_samples]
                idx = valid_idx[perm]
                
            # Get pixel coordinates
            v = idx // W
            u = idx % W
            z = depth_flat[b, idx]
            
            # Back-project to camera coordinates
            fx, fy = K[b, 0, 0], K[b, 1, 1]
            cx, cy = K[b, 0, 2], K[b, 1, 2]
            
            x = (u.float() - cx) * z / fx
            y = (v.float() - cy) * z / fy
            
            pts_cam = torch.stack([x, y, z], dim=-1)  # (N, 3)
            
            # Transform to world coordinates
            R = pose[b, :3, :3]
            t = pose[b, :3, 3]
            
            pts_world = pts_cam @ R.T + t
            
            all_surface_points.append(pts_world)
            all_camera_centers.append(t.expand(N_samples, -1))
            
        surface_points = torch.stack(all_surface_points, dim=0)  # (B, N, 3)
        camera_centers = torch.stack(all_camera_centers, dim=0)  # (B, N, 3)
        
        return surface_points, camera_centers
        
    def _sample_freespace_points(
        self,
        surface_points: torch.Tensor,
        camera_centers: torch.Tensor
    ) -> tuple:
        """Sample freespace points between camera and surface."""
        # Ray direction (from surface to camera)
        ray_dirs = camera_centers - surface_points
        ray_lengths = torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_dirs = ray_dirs / (ray_lengths + 1e-8)
        
        # Random jitter distance
        jitter = torch.rand_like(ray_lengths)
        jitter = jitter * (self.config.freespace_jitter_max - self.config.freespace_jitter_min)
        jitter = jitter + self.config.freespace_jitter_min
        
        # Clamp to not go past camera
        jitter = torch.min(jitter, ray_lengths * 0.9)
        
        # Sample freespace points
        freespace_points = surface_points + ray_dirs * jitter
        
        # Target SDF values (distance to surface)
        freespace_targets = jitter.squeeze(-1)
        
        return freespace_points, freespace_targets
    
    def _sample_random_space_points(self, batch_size: int) -> torch.Tensor:
        """
        Sample random points throughout the scene volume.
        
        These points should mostly be in empty space (outside surfaces),
        helping the model learn positive SDF values for empty regions.
        
        Args:
            batch_size: Number of samples per batch item
            
        Returns:
            random_points: (B, N, 3) random world coordinates
        """
        N = self.config.num_random_samples
        
        # Sample uniformly in the scene bounding box (world coordinates)
        random_points = torch.rand(
            batch_size, N, 3, 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Scale to scene bounds
        scene_extent = self.scene_max - self.scene_min
        random_points = random_points * scene_extent + self.scene_min
        
        return random_points
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._to_device(batch)
            
            # Forward pass
            depth = batch['depth']
            pose = batch['pose']
            K = batch['K']
            valid_mask = batch.get('valid_mask', depth > 0)
            
            surface_points_world, camera_centers = self._sample_surface_points(
                depth, pose, K, valid_mask
            )
            
            # Normalize coordinates for model
            surface_points = self._normalize_coords(surface_points_world)
            
            surface_out = self.model(surface_points)
            sdf_surface = surface_out['sdf']
            
            loss = torch.abs(sdf_surface).mean()
            total_loss += loss.item()
            num_batches += 1
            
        return {'loss': total_loss / num_batches}
        
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
    def _log_step(self, loss: float, components: Dict[str, float]):
        """Log training step metrics."""
        if self.writer is None:
            return
            
        self.writer.add_scalar('train/loss', loss, self.global_step)
        for key, value in components.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
        self.writer.add_scalar(
            'train/lr',
            self.optimizer.param_groups[0]['lr'],
            self.global_step
        )
        
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        
        for key, value in train_metrics.items():
            if key != 'loss':
                print(f"  Train {key}: {value:.6f}")
                
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            
        if self.writer:
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.current_epoch)
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], self.current_epoch)
                
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch{self.current_epoch}.pt'
            
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def _cache_viz_batch(self):
        """Cache ground truth points from across the entire dataset."""
        try:
            # Sample GT points from multiple batches across the dataset
            print("Caching GT points from across dataset...")
            all_points = []
            max_batches = min(100, len(self.train_loader))  # Sample up to 100 batches
            
            for i, batch in enumerate(self.train_loader):
                if i >= max_batches:
                    break
                
                # Only cache the first batch for visualization
                if i == 0:
                    self._viz_batch = {k: v.clone() if torch.is_tensor(v) else v 
                                       for k, v in batch.items()}
                
                # Extract points from this batch
                depth = batch['depth']
                pose = batch['pose']
                K = batch['K']
                valid_mask = batch.get('valid_mask', depth > 0)
                
                B, H, W = depth.shape
                
                for b in range(B):
                    valid = valid_mask[b]
                    v, u = torch.where(valid)
                    
                    if len(v) < 100:
                        continue
                    
                    # Subsample each frame
                    sample_idx = torch.randperm(len(v))[:1000]
                    v, u = v[sample_idx], u[sample_idx]
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
            
            if all_points:
                self._gt_points = np.concatenate(all_points, axis=0)
                
                # Subsample to reasonable size
                max_points = 100000
                if len(self._gt_points) > max_points:
                    indices = np.random.choice(len(self._gt_points), max_points, replace=False)
                    self._gt_points = self._gt_points[indices]
                
                print(f"Cached {len(self._gt_points)} GT points from {max_batches} batches")
                print(f"GT point range: X=[{self._gt_points[:, 0].min():.1f}, {self._gt_points[:, 0].max():.1f}], "
                      f"Y=[{self._gt_points[:, 1].min():.1f}, {self._gt_points[:, 1].max():.1f}], "
                      f"Z=[{self._gt_points[:, 2].min():.1f}, {self._gt_points[:, 2].max():.1f}]")
            else:
                print("Warning: No valid GT points found")
                self._gt_points = None
                    
        except Exception as e:
            print(f"Warning: Could not cache visualization batch: {e}")
            import traceback
            traceback.print_exc()
            self._viz_batch = None
            self._gt_points = None
    
    def _generate_visualizations(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Generate visualizations for the current epoch."""
        if self.visualizer is None:
            return
            
        print(f"Generating visualizations for epoch {epoch}...")
        
        try:
            eval_metrics = self.visualizer.visualize_checkpoint(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                sample_batch=self._viz_batch,
                gt_points=self._gt_points,
            )
            
            # Log evaluation metrics
            if eval_metrics and self.writer:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'eval/{key}', value, epoch)
            
            # Print summary
            if eval_metrics:
                print(f"  Reconstruction metrics:")
                print(f"    Chamfer Distance: {eval_metrics.get('chamfer_distance', 0):.6f}")
                print(f"    Accuracy @5cm: {eval_metrics.get('accuracy_5cm', 0)*100:.1f}%")
                print(f"    Completeness @5cm: {eval_metrics.get('completeness_5cm', 0)*100:.1f}%")
                print(f"    F-score @5cm: {eval_metrics.get('fscore_5cm', 0)*100:.1f}%")
                
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
