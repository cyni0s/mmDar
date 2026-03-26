"""Stage 3 residual point cloud decoder with 2D angular topology preserved.

Fixes the angular collapse bug in decoder.py where feature sampling uses a
(B, C, 1, R) feature map — height=1 makes azimuth interpolation a no-op,
so points at the same range but different azimuths get identical features.

This decoder takes a 2D feature map (B, C, H=N_az, W=N_r) and uses
grid_sample on the full 2D grid, giving each point azimuth-resolved features.

Key differences from decoder.py:
    1. sample_features_2d: operates on (B, C, H, W) not (B, C, R)
       - Uses sin_theta for azimuth mapping (matches LISTA/FFT sin-theta grid)
       - sin_theta = y / r maps directly to [-1, +1] for grid_sample
    2. PointCloudDecoder2D: Conv2d global encoder (not Conv1d)
       - AdaptiveMaxPool over both spatial dims
       - Otherwise identical architecture (polar template + 3 DensificationStages)

Architecture:
    Input:  (B, feature_ch, 256, 512) float32 feature map
    Output: (B, 8192, 3) float32 point cloud, (B, 8192, 1) float32 confidence logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.decoder import build_polar_template, DensificationStage


def sample_features_2d(
    feature_map: torch.Tensor,
    pts_xyz: torch.Tensor,
    r_max: float = 10.8,
) -> torch.Tensor:
    """Sample features from a 2D (azimuth x range) feature map at point locations.

    Feature map layout: (B, C, H=N_az, W=N_r) where H is azimuth
    in sin_theta space [-1, +1] and W is range [0, r_max].

    Uses sin_theta = y / r for azimuth coordinate, which maps directly to the
    LISTA/FFT beamformer's sin-theta grid: sin_theta[k] = -1 + 2k/(N_az-1).

    Args:
        feature_map: (B, C, H=N_az, W=N_r) float32 feature map
        pts_xyz:     (B, N, 3) float32 point coordinates [x, y, z]
        r_max:       Maximum range for normalization (default 10.8 m)

    Returns:
        local_feats: (B, N, C) float32 interpolated features at each point
    """
    B, C, H, W = feature_map.shape
    N = pts_xyz.shape[1]

    x = pts_xyz[..., 0]  # (B, N)
    y = pts_xyz[..., 1]  # (B, N)
    r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)  # (B, N)

    # Range: [0, r_max] -> [-1, +1] for grid_sample x (W dimension)
    grid_r = (r / r_max) * 2.0 - 1.0
    grid_r = grid_r.clamp(-1.0, 1.0)

    # Azimuth: sin_theta directly maps to [-1, +1] for grid_sample y (H dimension)
    # This matches LISTA's sin_theta grid: sin_theta[k] = -1 + 2k/(N_az-1)
    sin_theta = y / (r + 1e-8)
    grid_az = sin_theta.clamp(-1.0, 1.0)

    # grid_sample convention: grid[..., 0] = x (width = range), grid[..., 1] = y (height = azimuth)
    grid = torch.stack([grid_r, grid_az], dim=-1).unsqueeze(2)  # (B, N, 1, 2)

    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, N, 1)

    # Reshape to (B, N, C)
    return sampled.squeeze(-1).permute(0, 2, 1)


class PointCloudDecoder2D(nn.Module):
    """Stage 3 residual point cloud decoder with 2D angular topology.

    Same architecture as PointCloudDecoder but takes (B, feature_ch, H=256, W=512)
    instead of (B, feature_ch, R=512). Uses Conv2d(kernel_size=1) for the global
    encoder and sample_features_2d for azimuth-resolved feature sampling.

    Pipeline:
        1. Global encoder: Conv2d stack + global max pool over both spatial dims -> (B, global_dim)
        2. Template: 32x32 polar grid registered as buffer -> (B, 1024, 3)
        3. 3x DensificationStage with 2D local feature sampling at each step

    Args:
        feature_ch: Number of channels in input feature map (default 128)
        global_dim: Dimension of global scene descriptor (default 1024)
        r_max:      Maximum range in meters for template + feature sampling
    """

    def __init__(
        self,
        feature_ch: int = 128,
        global_dim: int = 1024,
        r_max: float = 10.8,
    ) -> None:
        super().__init__()
        self.feature_ch = feature_ch
        self.global_dim = global_dim
        self.r_max = r_max

        # Polar template (1024 = 32x32 range x azimuth grid)
        template = build_polar_template(
            N_r=32, N_az=32, r_max=r_max, az_range_deg=140.0
        )  # (1024, 3)
        self.register_buffer("template", template)

        # Global encoder: pool spatial dims FIRST to avoid OOM on (B, 1024, 256, 512).
        # AdaptiveMaxPool2d(1) reduces (B, C, 256, 512) → (B, C, 1, 1) before the
        # channel-expansion layers, cutting peak activation memory by 131072×.
        self.global_pool = nn.AdaptiveMaxPool2d(1)  # (B, C, H, W) → (B, C, 1, 1)
        self.global_encoder = nn.Sequential(
            nn.Conv2d(feature_ch, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, global_dim, kernel_size=1),
        )

        # Each DensificationStage: in_dim = global_dim + feature_ch + 3
        stage_in_dim = global_dim + feature_ch + 3  # 1155

        # DensificationStage is reused as-is from decoder.py (MLP, shape-agnostic)
        # first_bias=False on stage 0 forces dependence on beamformer signal
        self.stages = nn.ModuleList([
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256, first_bias=False),
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256),
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256),
        ])

    def forward(
        self, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode 2D feature map to dense point cloud.

        Args:
            feature_map: (B, feature_ch, H=256, W=512) float32

        Returns:
            pts:  (B, 8192, 3) float32 point coordinates
            conf: (B, 8192, 1) float32 confidence logits (raw, pre-sigmoid)
        """
        B = feature_map.shape[0]

        # --- Global scene descriptor ---
        # Pool spatial dims FIRST to avoid OOM: (B, C, 256, 512) → (B, C, 1, 1)
        pooled = self.global_pool(feature_map)   # (B, feature_ch, 1, 1)
        enc = self.global_encoder(pooled)         # (B, global_dim, 1, 1)
        global_desc = enc.squeeze(-1).squeeze(-1) # (B, global_dim)

        # --- Initialize from polar template ---
        pts = self.template.unsqueeze(0).expand(B, -1, -1).clone()  # (B, 1024, 3)

        # --- Three densification stages ---
        conf = None
        for stage in self.stages:
            local_feats = sample_features_2d(
                feature_map, pts, r_max=self.r_max
            )  # (B, N, feature_ch)
            pts, conf = stage(pts, global_desc, local_feats)

        return pts, conf
