"""
Checkpoint management utilities.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    path: str
    epoch: int
    step: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'path': self.path,
            'epoch': self.epoch,
            'step': self.step,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointInfo':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    
    Features:
    - Save/load checkpoints
    - Keep best N checkpoints
    - Track training progress
    - Resume training from checkpoint
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        metric_name: Name of metric to track for best checkpoint
        mode: 'min' or 'max' - whether lower or higher metric is better
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        metric_name: str = 'val_loss',
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        self.checkpoints: list[CheckpointInfo] = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint: Optional[CheckpointInfo] = None
        
        # Load existing checkpoint info
        self._load_checkpoint_info()
        
    def _load_checkpoint_info(self):
        """Load checkpoint info from disk."""
        info_file = self.checkpoint_dir / 'checkpoint_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                data = json.load(f)
                
            self.checkpoints = [
                CheckpointInfo.from_dict(c) for c in data.get('checkpoints', [])
            ]
            self.best_metric = data.get('best_metric', self.best_metric)
            
            if data.get('best_checkpoint'):
                self.best_checkpoint = CheckpointInfo.from_dict(data['best_checkpoint'])
                
    def _save_checkpoint_info(self):
        """Save checkpoint info to disk."""
        info_file = self.checkpoint_dir / 'checkpoint_info.json'
        
        data = {
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'best_metric': self.best_metric,
            'best_checkpoint': self.best_checkpoint.to_dict() if self.best_checkpoint else None
        }
        
        with open(info_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.mode == 'min':
            return metric < self.best_metric
        return metric > self.best_metric
        
    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            state_dict: State dictionary to save (model, optimizer, etc.)
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Import torch here to avoid circular imports
        import torch
        
        metrics = metrics or {}
        
        # Determine if this is the best checkpoint
        if self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self._is_better(metric_value):
                is_best = True
                self.best_metric = metric_value
                
        # Save checkpoint
        checkpoint_name = f'checkpoint_epoch{epoch:04d}_step{step:08d}.pt'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Add metadata to state dict
        state_dict['epoch'] = epoch
        state_dict['step'] = step
        state_dict['metrics'] = metrics
        
        torch.save(state_dict, checkpoint_path)
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            path=str(checkpoint_path),
            epoch=epoch,
            step=step,
            metrics=metrics
        )
        
        self.checkpoints.append(checkpoint_info)
        
        # Update best checkpoint
        if is_best:
            self.best_checkpoint = checkpoint_info
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            shutil.copy(checkpoint_path, best_path)
            
        # Save latest checkpoint link
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        shutil.copy(checkpoint_path, latest_path)
        
        # Cleanup old checkpoints
        self._cleanup()
        
        # Save checkpoint info
        self._save_checkpoint_info()
        
        return str(checkpoint_path)
        
    def _cleanup(self):
        """Remove old checkpoints exceeding max_checkpoints."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Sort by epoch/step (oldest first)
        self.checkpoints.sort(key=lambda c: (c.epoch, c.step))
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            
            # Don't delete if it's the best checkpoint
            if (self.best_checkpoint and 
                old_checkpoint.path == self.best_checkpoint.path):
                self.checkpoints.insert(0, old_checkpoint)
                if len(self.checkpoints) > 1:
                    old_checkpoint = self.checkpoints.pop(1)
                else:
                    break
                    
            # Delete file
            old_path = Path(old_checkpoint.path)
            if old_path.exists():
                old_path.unlink()
                
    def load(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest.
            
        Returns:
            Loaded state dictionary
        """
        import torch
        
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
            
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        return torch.load(checkpoint_path, map_location='cpu')
        
    def load_best(self) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / 'best_checkpoint.pt'
        return self.load(best_path)
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        if latest_path.exists():
            return str(latest_path)
        return None
        
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'best_checkpoint.pt'
        if best_path.exists():
            return str(best_path)
        return None
        
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return self.get_latest_checkpoint() is not None


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    path: Union[str, Path],
    scheduler=None,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs
):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        step: Current step
        path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional metrics dictionary
        **kwargs: Additional items to save
    """
    import torch
    
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {}
    }
    
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    state_dict.update(kwargs)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state_dict, path)
    

def load_checkpoint(
    path: Union[str, Path],
    model=None,
    optimizer=None,
    scheduler=None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: PyTorch optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        device: Device to load checkpoint to
        
    Returns:
        Loaded checkpoint dictionary
    """
    import torch
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint
