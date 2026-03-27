"""
Automatic batch size scaling based on available GPU memory.

Determines optimal batch size by:
1. Detecting available VRAM
2. Running a quick memory profiling pass
3. Binary search for max batch size that fits
"""

import gc
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """GPU memory profile results."""
    total_memory_gb: float
    available_memory_gb: float
    model_memory_mb: float
    recommended_batch_size: int
    max_tested_batch_size: int
    memory_per_sample_mb: float
    safety_margin: float


def get_gpu_memory_info(device: torch.device = None) -> Tuple[float, float]:
    """
    Get total and available GPU memory in GB.
    
    Returns:
        (total_gb, available_gb)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        return (0.0, 0.0)
    
    # Get memory from PyTorch (more reliable than nvidia-smi on some systems)
    torch.cuda.synchronize()
    
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    
    total_gb = total / (1024**3)
    available_gb = (total - reserved) / (1024**3)
    
    return total_gb, available_gb


def estimate_model_memory(model: nn.Module, device: torch.device = None) -> float:
    """
    Estimate memory used by model parameters and buffers.
    
    Returns:
        Memory in MB
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Account for gradients (same size as parameters)
    grad_memory = param_memory
    
    # Account for optimizer states (Adam uses 2 momentum buffers)
    optimizer_memory = param_memory * 2
    
    total_mb = (param_memory + buffer_memory + grad_memory + optimizer_memory) / (1024**2)
    
    return total_mb


def profile_memory_usage(
    model: nn.Module,
    sample_input_fn: callable,
    batch_size: int,
    device: torch.device = None,
) -> Tuple[bool, float]:
    """
    Profile memory usage for a given batch size.
    
    Args:
        model: The neural network model
        sample_input_fn: Function that creates sample input for given batch size
        batch_size: Batch size to test
        device: GPU device
        
    Returns:
        (success, memory_used_mb)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    initial_memory = torch.cuda.memory_allocated(device)
    
    try:
        # Create sample input
        sample_input = sample_input_fn(batch_size)
        
        # Move to device
        if isinstance(sample_input, dict):
            sample_input = {k: v.to(device) if torch.is_tensor(v) else v 
                          for k, v in sample_input.items()}
        elif torch.is_tensor(sample_input):
            sample_input = sample_input.to(device)
        
        # Forward pass
        model.train()
        
        if isinstance(sample_input, dict):
            output = model(sample_input.get('points', sample_input.get('coords')))
        else:
            output = model(sample_input)
        
        # Compute dummy loss and backward
        if isinstance(output, dict):
            loss = output['sdf'].mean()
        else:
            loss = output.mean()
        
        loss.backward()
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_used = (peak_memory - initial_memory) / (1024**2)
        
        # Clear
        del output, loss, sample_input
        gc.collect()
        torch.cuda.empty_cache()
        
        return True, memory_used
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return False, 0.0
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def find_optimal_batch_size(
    model: nn.Module,
    sample_input_fn: callable,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    target_memory_fraction: float = 0.85,
    device: torch.device = None,
    verbose: bool = True,
) -> MemoryProfile:
    """
    Find optimal batch size using binary search.
    
    Args:
        model: Neural network model
        sample_input_fn: Function to create sample input for given batch size
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        target_memory_fraction: Target fraction of GPU memory to use (0.85 = 85%)
        device: GPU device
        verbose: Print progress
        
    Returns:
        MemoryProfile with recommended batch size
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.train()
    
    # Get memory info
    total_gb, available_gb = get_gpu_memory_info(device)
    model_memory = estimate_model_memory(model, device)
    
    if verbose:
        print(f"GPU Memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        print(f"Model Memory (estimated): {model_memory:.0f} MB")
        print(f"Target memory usage: {target_memory_fraction*100:.0f}%")
        print(f"Searching batch size in range [{min_batch_size}, {max_batch_size}]...")
    
    # Binary search for max batch size
    low = min_batch_size
    high = max_batch_size
    best_batch_size = min_batch_size
    memory_per_sample = 0.0
    
    # First, check if min batch size works
    success, mem_used = profile_memory_usage(model, sample_input_fn, min_batch_size, device)
    if not success:
        if verbose:
            print(f"ERROR: Even batch size {min_batch_size} doesn't fit in memory!")
        return MemoryProfile(
            total_memory_gb=total_gb,
            available_memory_gb=available_gb,
            model_memory_mb=model_memory,
            recommended_batch_size=1,
            max_tested_batch_size=0,
            memory_per_sample_mb=0,
            safety_margin=target_memory_fraction,
        )
    
    base_memory = mem_used
    
    while low <= high:
        mid = (low + high) // 2
        
        if verbose:
            print(f"  Testing batch size {mid}...", end=" ")
        
        success, mem_used = profile_memory_usage(model, sample_input_fn, mid, device)
        
        if success:
            best_batch_size = mid
            memory_per_sample = (mem_used - base_memory) / max(mid - min_batch_size, 1) if mid > min_batch_size else mem_used / mid
            
            if verbose:
                print(f"OK ({mem_used:.0f} MB)")
            
            low = mid + 1
        else:
            if verbose:
                print(f"OOM")
            high = mid - 1
    
    # Apply safety margin
    target_memory_mb = available_gb * 1024 * target_memory_fraction
    usable_memory = target_memory_mb - model_memory
    
    if memory_per_sample > 0:
        safe_batch_size = max(1, int(usable_memory / memory_per_sample))
        recommended = min(best_batch_size, safe_batch_size)
    else:
        recommended = best_batch_size
    
    # Round to nice numbers
    if recommended >= 32:
        recommended = (recommended // 8) * 8
    elif recommended >= 8:
        recommended = (recommended // 4) * 4
    elif recommended >= 4:
        recommended = (recommended // 2) * 2
    
    recommended = max(1, recommended)
    
    if verbose:
        print(f"\n  Max tested batch size: {best_batch_size}")
        print(f"  Memory per sample: {memory_per_sample:.1f} MB")
        print(f"  Recommended batch size: {recommended}")
    
    return MemoryProfile(
        total_memory_gb=total_gb,
        available_memory_gb=available_gb,
        model_memory_mb=model_memory,
        recommended_batch_size=recommended,
        max_tested_batch_size=best_batch_size,
        memory_per_sample_mb=memory_per_sample,
        safety_margin=target_memory_fraction,
    )


def auto_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1000, 3),
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    target_memory_fraction: float = 0.80,
    device: torch.device = None,
    verbose: bool = True,
) -> int:
    """
    Convenience function to find optimal batch size for a model.
    
    Args:
        model: Neural network model
        input_shape: Shape of single sample input (without batch dim)
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        target_memory_fraction: Target GPU memory usage fraction
        device: GPU device
        verbose: Print progress
        
    Returns:
        Recommended batch size
    """
    def sample_input_fn(batch_size: int):
        # Create random input tensor
        shape = (batch_size,) + tuple(input_shape[1:]) if len(input_shape) > 1 else (batch_size, input_shape[0])
        return torch.randn(shape)
    
    profile = find_optimal_batch_size(
        model=model,
        sample_input_fn=sample_input_fn,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        target_memory_fraction=target_memory_fraction,
        device=device,
        verbose=verbose,
    )
    
    return profile.recommended_batch_size


def auto_batch_size_for_sdf(
    model: nn.Module,
    num_points_per_sample: int = 8192,
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    target_memory_fraction: float = 0.80,
    device: torch.device = None,
    verbose: bool = True,
) -> MemoryProfile:
    """
    Find optimal batch size for SDF training specifically.
    
    This accounts for the specific memory patterns of SDF training:
    - Multiple point samples per batch (surface, TSDF, far-field)
    - Gradient computation for eikonal loss
    - Mixed precision training
    
    Args:
        model: SDF model
        num_points_per_sample: Total points sampled per batch item
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        target_memory_fraction: Target GPU memory usage
        device: GPU device
        verbose: Print progress
        
    Returns:
        MemoryProfile with recommended batch size
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def sample_input_fn(batch_size: int):
        # Simulate SDF training input pattern
        # Points in normalized [0, 1] space
        return torch.rand(batch_size, num_points_per_sample, 3)
    
    if verbose:
        print("=" * 60)
        print("AUTO BATCH SIZE DETECTION FOR SDF TRAINING")
        print("=" * 60)
        print(f"Points per sample: {num_points_per_sample}")
    
    profile = find_optimal_batch_size(
        model=model,
        sample_input_fn=sample_input_fn,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        target_memory_fraction=target_memory_fraction,
        device=device,
        verbose=verbose,
    )
    
    if verbose:
        print("=" * 60)
    
    return profile


if __name__ == "__main__":
    # Test with a simple model
    import sys
    sys.path.insert(0, '.')
    
    from src.models import HashGridSDF
    
    print("Testing auto batch size detection...")
    
    model = HashGridSDF(
        hidden_features=256,
        hidden_layers=6,
        encoding_config={
            'num_levels': 16,
            'base_resolution': 16,
            'max_resolution': 4096,
            'features_per_level': 2,
            'log2_hashmap_size': 19,
        },
        geometric_init=True,
    )
    
    profile = auto_batch_size_for_sdf(
        model=model,
        num_points_per_sample=8192,
        min_batch_size=1,
        max_batch_size=32,
        target_memory_fraction=0.80,
        verbose=True,
    )
    
    print(f"\nFinal recommendation: batch_size={profile.recommended_batch_size}")
