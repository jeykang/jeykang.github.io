"""
Score-Based Denoising for Point Clouds.

Implements score matching and Langevin dynamics for
gradient-based point cloud refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class ScoreNetwork(nn.Module):
    """
    Score network for point cloud denoising.
    
    Estimates the score function ∇_x log p(x|σ), which points
    toward higher density regions of the data distribution.
    
    Args:
        hidden_dim: Hidden feature dimension
        num_noise_levels: Number of noise levels for training
        noise_levels: List of noise standard deviations
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_noise_levels: int = 5,
        noise_levels: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_noise_levels = num_noise_levels
        
        # Default noise levels (geometric sequence)
        if noise_levels is None:
            self.noise_levels = [0.1 * (0.5 ** i) for i in range(num_noise_levels)]
        else:
            self.noise_levels = noise_levels
            
        self.register_buffer(
            'sigmas',
            torch.tensor(self.noise_levels, dtype=torch.float32)
        )
        
        # Noise level embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Point encoder (simplified PointNet)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # Global feature aggregation
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Score prediction head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Score vector (dx, dy, dz)
        )
        
    def forward(
        self,
        xyz: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate score at given noise level.
        
        Args:
            xyz: (B, N, 3) noisy point coordinates
            sigma: (B,) or (B, 1) noise standard deviation
            
        Returns:
            score: (B, N, 3) estimated score vectors
        """
        B, N, _ = xyz.shape
        
        # Ensure sigma has correct shape
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)  # (B, 1)
            
        # Embed noise level
        noise_emb = self.noise_embed(sigma)  # (B, hidden_dim)
        noise_emb = noise_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
        
        # Encode points
        point_features = self.point_encoder(xyz)  # (B, N, hidden_dim)
        
        # Global feature
        global_feature = point_features.max(dim=1)[0]  # (B, hidden_dim)
        global_feature = self.global_encoder(global_feature)
        global_feature = global_feature.unsqueeze(1).expand(-1, N, -1)
        
        # Concatenate features
        combined = torch.cat([point_features, global_feature, noise_emb], dim=-1)
        
        # Predict score
        score = self.score_head(combined)
        
        return score
        
    def denoise(
        self,
        xyz: torch.Tensor,
        sigma_idx: int
    ) -> torch.Tensor:
        """
        Single denoising step at specified noise level.
        
        Args:
            xyz: (B, N, 3) noisy points
            sigma_idx: Index of noise level
            
        Returns:
            score: (B, N, 3) denoising direction
        """
        sigma = self.sigmas[sigma_idx].expand(xyz.shape[0])
        return self.forward(xyz, sigma)


class LangevinDynamics(nn.Module):
    """
    Annealed Langevin Dynamics for score-based denoising.
    
    Iteratively refines points by following the score function
    with decreasing noise levels.
    
    Args:
        score_network: Trained score network
        noise_levels: List of noise levels (high to low)
        steps_per_level: Number of Langevin steps per noise level
        step_size: Base step size for updates
        noise_scale: Scale of injected noise (0 = deterministic)
    """
    
    def __init__(
        self,
        score_network: ScoreNetwork,
        noise_levels: Optional[List[float]] = None,
        steps_per_level: int = 10,
        step_size: float = 0.01,
        noise_scale: float = 1.0
    ):
        super().__init__()
        
        self.score_network = score_network
        self.steps_per_level = steps_per_level
        self.step_size = step_size
        self.noise_scale = noise_scale
        
        if noise_levels is None:
            self.noise_levels = score_network.noise_levels
        else:
            self.noise_levels = noise_levels
            
    def forward(
        self,
        xyz: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Run annealed Langevin dynamics.
        
        Args:
            xyz: (B, N, 3) initial noisy points
            return_trajectory: Whether to return all intermediate states
            
        Returns:
            refined: (B, N, 3) or (B, T, N, 3) refined points
        """
        current = xyz.clone()
        trajectory = [current] if return_trajectory else None
        
        # Anneal through noise levels (high to low)
        for sigma in self.noise_levels:
            sigma_tensor = torch.full(
                (xyz.shape[0],), sigma, device=xyz.device
            )
            
            # Langevin steps at this noise level
            for _ in range(self.steps_per_level):
                # Get score
                score = self.score_network(current, sigma_tensor)
                
                # Step size scaled by sigma²
                eps = self.step_size * (sigma ** 2)
                
                # Langevin update
                if self.noise_scale > 0:
                    noise = torch.randn_like(current) * np.sqrt(2 * eps) * self.noise_scale
                else:
                    noise = 0
                    
                current = current + eps * score + noise
                
                if return_trajectory:
                    trajectory.append(current.clone())
                    
        if return_trajectory:
            return torch.stack(trajectory, dim=1)
        return current


class ScoreMatchingLoss(nn.Module):
    """
    Denoising Score Matching loss for training.
    
    L = E_{σ,x,ε} [||s_θ(x + σε, σ) - (-ε/σ)||²]
    
    The optimal score at noise level σ is -ε/σ, pointing
    back toward the clean data.
    """
    
    def __init__(
        self,
        noise_levels: List[float],
        loss_weighting: str = 'uniform'
    ):
        super().__init__()
        
        self.noise_levels = noise_levels
        self.loss_weighting = loss_weighting
        
    def forward(
        self,
        score_network: ScoreNetwork,
        clean_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute denoising score matching loss.
        
        Args:
            score_network: Score network to train
            clean_points: (B, N, 3) clean point cloud
            
        Returns:
            loss: Scalar loss value
        """
        B = clean_points.shape[0]
        device = clean_points.device
        
        # Sample random noise level
        sigma_idx = torch.randint(0, len(self.noise_levels), (B,), device=device)
        sigmas = torch.tensor(self.noise_levels, device=device)[sigma_idx]
        
        # Add noise
        noise = torch.randn_like(clean_points)
        noisy_points = clean_points + sigmas.view(B, 1, 1) * noise
        
        # Predict score
        predicted_score = score_network(noisy_points, sigmas)
        
        # Target score: -noise / sigma
        target_score = -noise / sigmas.view(B, 1, 1)
        
        # L2 loss
        if self.loss_weighting == 'uniform':
            loss = F.mse_loss(predicted_score, target_score)
        elif self.loss_weighting == 'sigma':
            # Weight by σ² (helps with different noise scales)
            weights = sigmas.view(B, 1, 1) ** 2
            loss = (weights * (predicted_score - target_score) ** 2).mean()
        else:
            loss = F.mse_loss(predicted_score, target_score)
            
        return loss


class DiffusionPointDenoiser(nn.Module):
    """
    Diffusion-based point cloud denoiser.
    
    Combines score network with DDPM-style reverse diffusion
    for high-quality denoising.
    
    Args:
        hidden_dim: Network hidden dimension
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Noise schedule type ('linear', 'cosine')
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear'
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        elif beta_schedule == 'cosine':
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")
            
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Noise prediction network
        self.noise_net = nn.Sequential(
            nn.Linear(3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Point encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise at timestep t.
        
        Args:
            x_t: (B, N, 3) noisy points at timestep t
            t: (B,) timestep indices
            
        Returns:
            noise_pred: (B, N, 3) predicted noise
        """
        B, N, _ = x_t.shape
        
        # Embed timestep
        t_emb = self.time_embed(t.float().view(B, 1) / self.num_timesteps)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)
        
        # Encode points
        point_emb = self.point_encoder(x_t)
        
        # Predict noise
        combined = torch.cat([x_t, point_emb + t_emb], dim=-1)
        noise_pred = self.noise_net(combined)
        
        return noise_pred
        
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean points.
        
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self.forward(x_t, t_tensor)
        
        # Reverse step
        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # Mean
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1 - alpha_bar)
        mean = coef1 * (x_t - coef2 * noise_pred)
        
        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta)
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean
            
        return x_t_minus_1
        
    @torch.no_grad()
    def sample(
        self,
        x_T: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full reverse diffusion to denoise.
        
        Args:
            x_T: (B, N, 3) fully noised points
            num_steps: Number of denoising steps (default: all)
            
        Returns:
            x_0: (B, N, 3) denoised points
        """
        steps = num_steps or self.num_timesteps
        step_size = self.num_timesteps // steps
        
        x = x_T
        for t in range(self.num_timesteps - 1, -1, -step_size):
            x = self.p_sample(x, t)
            
        return x
