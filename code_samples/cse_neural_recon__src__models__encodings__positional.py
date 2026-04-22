"""
Positional encoding for neural implicit representations.

Implements:
- Sinusoidal positional encoding (NeRF-style)
- Fourier feature mapping (Random Fourier Features)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 3D coordinates.
    
    Maps input coordinates to higher dimensional space using
    sinusoidal functions at multiple frequencies, enabling
    neural networks to learn high-frequency functions.
    
    encoding(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^L π p), cos(2^L π p)]
    
    Args:
        num_frequencies: Number of frequency bands (L)
        include_input: Whether to include original coordinates
        log_sampling: Use log-linear frequency spacing
        periodic_fns: Custom periodic functions (default: sin, cos)
    """
    
    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
        input_dim: int = 3
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.input_dim = input_dim
        
        # Compute output dimension
        # Each frequency produces 2 values (sin, cos) for each input dimension
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim
            
        # Precompute frequency bands
        if log_sampling:
            # Log-linear spacing: 2^0, 2^1, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            # Linear spacing from 1 to 2^(L-1)
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)
            
        self.register_buffer('freq_bands', freq_bands * np.pi)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: (*, input_dim) input coordinates
            
        Returns:
            encoded: (*, output_dim) encoded features
        """
        # x: (..., input_dim)
        # Expand frequencies: (..., input_dim, num_freq)
        x_freq = x.unsqueeze(-1) * self.freq_bands
        
        # Apply sin and cos
        sin_features = torch.sin(x_freq)  # (..., input_dim, num_freq)
        cos_features = torch.cos(x_freq)  # (..., input_dim, num_freq)
        
        # Interleave sin and cos, then flatten
        # Result: (..., input_dim * num_freq * 2)
        encoded = torch.stack([sin_features, cos_features], dim=-1)
        encoded = encoded.view(*x.shape[:-1], -1)
        
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
            
        return encoded
        
    def get_output_dim(self) -> int:
        """Return output dimension of encoding."""
        return self.output_dim


class FourierFeatures(nn.Module):
    """
    Random Fourier Features encoding.
    
    Uses random frequencies instead of predetermined ones,
    which can provide better coverage of the frequency spectrum.
    
    γ(p) = [cos(2π B p), sin(2π B p)]
    
    where B is a random matrix sampled from N(0, σ²).
    
    Args:
        input_dim: Input coordinate dimension
        num_features: Number of Fourier features
        sigma: Standard deviation of random frequencies
        trainable: Whether frequencies are trainable
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        num_features: int = 256,
        sigma: float = 10.0,
        trainable: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.output_dim = num_features * 2  # sin and cos
        
        # Initialize random frequency matrix
        B = torch.randn(input_dim, num_features) * sigma
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping.
        
        Args:
            x: (*, input_dim) input coordinates
            
        Returns:
            features: (*, num_features * 2) Fourier features
        """
        # x @ B: (*, num_features)
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        
        # Concatenate sin and cos
        features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        
        return features
        
    def get_output_dim(self) -> int:
        """Return output dimension of encoding."""
        return self.output_dim


class IntegratedPositionalEncoding(nn.Module):
    """
    Integrated Positional Encoding (IPE) from Mip-NeRF.
    
    Computes expected value of positional encoding over
    a Gaussian distribution, providing anti-aliasing for
    neural radiance fields.
    
    Args:
        num_frequencies: Number of frequency bands
        include_input: Whether to include original coordinates
    """
    
    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        input_dim: int = 3
    ):
        super().__init__()
        
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.input_dim = input_dim
        
        # Output dimension
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim
            
        # Frequency bands
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands * np.pi)
        
    def forward(
        self,
        mean: torch.Tensor,
        var: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply integrated positional encoding.
        
        Args:
            mean: (*, input_dim) mean of Gaussian
            var: (*, input_dim) variance (diagonal covariance)
            
        Returns:
            encoded: (*, output_dim) IPE features
        """
        # mean_freq: (..., input_dim, num_freq)
        mean_freq = mean.unsqueeze(-1) * self.freq_bands
        var_freq = var.unsqueeze(-1) * (self.freq_bands ** 2)
        
        # Expected sin and cos under Gaussian
        # E[sin(x)] = sin(μ) * exp(-σ²/2)
        # E[cos(x)] = cos(μ) * exp(-σ²/2)
        decay = torch.exp(-0.5 * var_freq)
        
        sin_features = torch.sin(mean_freq) * decay
        cos_features = torch.cos(mean_freq) * decay
        
        # Flatten
        encoded = torch.stack([sin_features, cos_features], dim=-1)
        encoded = encoded.view(*mean.shape[:-1], -1)
        
        if self.include_input:
            encoded = torch.cat([mean, encoded], dim=-1)
            
        return encoded
        
    def get_output_dim(self) -> int:
        """Return output dimension of encoding."""
        return self.output_dim
