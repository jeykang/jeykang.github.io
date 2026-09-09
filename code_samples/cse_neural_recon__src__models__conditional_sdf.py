"""
Pixel-aligned conditional SDF.

This implements a conditional implicit surface model:
  SDF(x | {I_v, K_v, T_w<-c_v}_v)

where the conditioning is obtained by projecting 3D query points into each view
and sampling learned image features (pixel-aligned).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .encodings import HashGridEncoding


class SimpleConvEncoder(nn.Module):
    """
    A small CNN that outputs a feature map for pixel-aligned conditioning.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, self.out_dim, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb)


class ConditionalHashGridSDF(nn.Module):
    """
    Hash-grid SDF decoder that is conditioned on per-point features.
    """

    def __init__(
        self,
        encoding_config: Dict,
        cond_dim: int = 64,
        hidden_features: int = 256,
        hidden_layers: int = 6,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.encoding = HashGridEncoding(**encoding_config)
        self.cond_dim = int(cond_dim)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)

        input_dim = int(self.encoding.output_dim) + self.cond_dim
        self.skip_layer = self.hidden_layers // 2

        layers = []
        first = nn.Linear(input_dim, self.hidden_features)
        if use_weight_norm:
            first = nn.utils.weight_norm(first)
        layers.append(first)

        for i in range(self.hidden_layers - 1):
            if i == self.skip_layer:
                in_dim = self.hidden_features + input_dim
            else:
                in_dim = self.hidden_features
            layer = nn.Linear(in_dim, self.hidden_features)
            if use_weight_norm:
                layer = nn.utils.weight_norm(layer)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.sdf_head = nn.Linear(self.hidden_features, 1)

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        original_shape = coords.shape[:-1]
        coords_flat = coords.reshape(-1, 3)
        cond_flat = cond.reshape(-1, self.cond_dim)

        x = self.encoding(coords_flat)
        x = torch.cat([x, cond_flat], dim=-1)
        encoded = x

        for i, layer in enumerate(self.layers):
            if i == self.skip_layer + 1:
                x = torch.cat([x, encoded], dim=-1)
            x = F.relu(layer(x), inplace=True)

        sdf = self.sdf_head(x)
        return {'sdf': sdf.view(*original_shape, 1)}


@dataclass
class ViewSet:
    rgb: torch.Tensor  # (B, V, 3, H, W)
    pose: torch.Tensor  # (B, V, 4, 4) camera-to-world (T_w<-c)
    K: torch.Tensor  # (B, V, 3, 3)


class PixelAlignedConditionalSDF(nn.Module):
    """
    Full conditional model: image encoder + pixel-aligned sampling + SDF decoder.

    Forward requires:
      - coords: normalized coords in [0, 1], shape (B, N, 3)
      - world_coords: world coords in meters, shape (B, N, 3)
      - views: ViewSet with rgb/pose/K
    """

    def __init__(
        self,
        encoding_config: Dict,
        cond_dim: int = 64,
        hidden_features: int = 256,
        hidden_layers: int = 6,
        use_weight_norm: bool = True,
        encoder_out_dim: int = 64,
    ):
        super().__init__()
        self.encoder = SimpleConvEncoder(out_dim=int(encoder_out_dim))
        self.decoder = ConditionalHashGridSDF(
            encoding_config=encoding_config,
            cond_dim=int(cond_dim),
            hidden_features=int(hidden_features),
            hidden_layers=int(hidden_layers),
            use_weight_norm=bool(use_weight_norm),
        )
        if int(cond_dim) != int(encoder_out_dim):
            self.cond_proj = nn.Conv2d(int(encoder_out_dim), int(cond_dim), kernel_size=1)
        else:
            self.cond_proj = None

    @torch.no_grad()
    def _invert_pose(self, pose_c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        R = pose_c2w[..., :3, :3]
        t = pose_c2w[..., :3, 3]
        R_wc = R.transpose(-1, -2)  # world->cam rotation
        t_wc = -(R_wc @ t.unsqueeze(-1)).squeeze(-1)  # world->cam translation
        return R_wc, t_wc

    def encode_views(self, views_rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB views into feature maps.

        Args:
            views_rgb: (B, V, 3, H, W)
        Returns:
            feats: (B, V, C, Hf, Wf)
        """
        B, V, C, H, W = views_rgb.shape
        x = views_rgb.reshape(B * V, C, H, W)
        feats = self.encoder(x)
        if self.cond_proj is not None:
            feats = self.cond_proj(feats)
        Cc, Hf, Wf = feats.shape[1], feats.shape[2], feats.shape[3]
        return feats.reshape(B, V, Cc, Hf, Wf)

    def _sample_pixel_aligned_features(
        self,
        world_coords: torch.Tensor,  # (B, N, 3)
        views_pose: torch.Tensor,  # (B, V, 4, 4) c2w
        views_K: torch.Tensor,  # (B, V, 3, 3)
        feats: torch.Tensor,  # (B, V, C, Hf, Wf)
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        # Projection math is numerically sensitive (especially for random points near
        # the camera plane). Do it in FP32 even when the outer forward is under AMP.
        device_type = world_coords.device.type
        with autocast(device_type=device_type, enabled=False):
            world_coords = world_coords.float()
            views_pose = views_pose.float()
            views_K = views_K.float()
            feats = feats.float()

            B, N, _ = world_coords.shape
            V = views_pose.shape[1]
            H, W = int(image_hw[0]), int(image_hw[1])
            Hf, Wf = int(feats.shape[-2]), int(feats.shape[-1])

            # Prepare for vectorized projection: expand points across V.
            pts = world_coords[:, None, :, :].expand(B, V, N, 3)  # (B, V, N, 3)
            R_wc, t_wc = self._invert_pose(views_pose)  # (B, V, 3, 3), (B, V, 3)

            # Transform to camera coordinates.
            pts_cam = (R_wc[:, :, None, :, :] @ pts.unsqueeze(-1)).squeeze(-1) + t_wc[:, :, None, :]  # (B, V, N, 3)
            x, y, z = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]

            fx = views_K[..., 0, 0][:, :, None]
            fy = views_K[..., 1, 1][:, :, None]
            cx = views_K[..., 0, 2][:, :, None]
            cy = views_K[..., 1, 2][:, :, None]

            # Guard against points extremely close to the camera plane, which can
            # overflow under AMP and poison the whole batch with NaNs.
            z_min = 1e-2
            z_safe = z.clamp(min=z_min)
            u = fx * (x / z_safe) + cx
            v = fy * (y / z_safe) + cy

            # Points behind the camera / too close to plane / outside image are invalid.
            valid = (z > z_min) & (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))

            # Map to feature-map normalized coordinates for grid_sample.
            u_f = u * (Wf - 1) / max(W - 1, 1)
            v_f = v * (Hf - 1) / max(H - 1, 1)
            grid_x = (u_f / max(Wf - 1, 1)) * 2 - 1
            grid_y = (v_f / max(Hf - 1, 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, V, N, 2)
            # Ensure invalid points don't introduce NaNs/Infs into grid_sample.
            grid = torch.where(valid.unsqueeze(-1), grid, torch.zeros_like(grid))

            # grid_sample expects (N, H_out, W_out, 2); we sample N points as a 1D "strip".
            feats_bv = feats.reshape(B * V, feats.shape[2], Hf, Wf)
            grid_bv = grid.reshape(B * V, N, 1, 2)
            sampled = F.grid_sample(
                feats_bv,
                grid_bv,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )  # (B*V, C, N, 1)
            sampled = sampled.squeeze(-1).transpose(1, 2)  # (B*V, N, C)
            sampled = sampled.reshape(B, V, N, -1)  # (B, V, N, C)

            valid_f = valid.float().unsqueeze(-1)  # (B, V, N, 1)
            sampled = sampled * valid_f
            denom = valid_f.sum(dim=1).clamp(min=1.0)  # (B, N, 1)
            cond = sampled.sum(dim=1) / denom  # (B, N, C)
            return cond

    def forward(
        self,
        coords: torch.Tensor,  # (B, N, 3) normalized
        *,
        world_coords: torch.Tensor,
        views_rgb: torch.Tensor,
        views_pose: torch.Tensor,
        views_K: torch.Tensor,
        views_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if views_feats is None:
            views_feats = self.encode_views(views_rgb)
        H, W = int(views_rgb.shape[-2]), int(views_rgb.shape[-1])
        cond = self._sample_pixel_aligned_features(
            world_coords=world_coords,
            views_pose=views_pose,
            views_K=views_K,
            feats=views_feats,
            image_hw=(H, W),
        )
        return self.decoder(coords, cond)
