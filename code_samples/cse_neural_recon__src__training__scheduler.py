"""
Learning rate schedulers with warmup support.
"""

import math
from typing import Optional
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    
    Linearly increases LR from 0 to base_lr over warmup_steps,
    then applies the specified decay schedule.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        base_scheduler: Scheduler to use after warmup
        last_epoch: Last epoch index
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs
            
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        
        if epoch >= self.warmup_steps and self.base_scheduler is not None:
            self.base_scheduler.step()
            
        # Apply learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    
    Learning rate follows a cosine curve that resets to initial
    value periodically, with optional increasing period lengths.
    
    Args:
        optimizer: PyTorch optimizer
        T_0: Initial restart period
        T_mult: Period multiplier after each restart
        eta_min: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Compute learning rate using cosine annealing."""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
        
    def step(self, epoch=None):
        """Step scheduler with restart logic."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_cur = epoch
                self.T_i = self.T_0
                
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_epochs: int = 0,
    total_epochs: int = 100,
    min_lr: float = 1e-6,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'exp', 'plateau', 'warmup_cosine')
        warmup_epochs: Number of warmup epochs (for warmup variants)
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
    elif scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    elif scheduler_type == 'exp':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
        
    elif scheduler_type == 'plateau':
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        
    elif scheduler_type == 'cosine_restart':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr
        )
        
    elif scheduler_type == 'warmup_cosine':
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=warmup_epochs,
            base_scheduler=base_scheduler
        )
        return scheduler
        
    elif scheduler_type == 'none':
        return None
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    # Wrap with warmup if requested
    if warmup_epochs > 0 and scheduler_type != 'warmup_cosine':
        scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=warmup_epochs,
            base_scheduler=scheduler
        )
        
    return scheduler
