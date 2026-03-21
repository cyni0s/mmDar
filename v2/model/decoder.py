"""Stage 3 residual point cloud decoder for mmDar v2.

Implements a mmPoint-style Lift-and-Deform decoder that progressively densifies
a fixed polar template grid from 1024 to 8192 points via 3 deformation stages.

Architecture:
    Input:  (B, 128, 512) float32 feature map from Stage2Bridge
    Output: (B, 8192, 3) float32 point cloud, (B, 8192, 1) float32 confidence logits

Pipeline:
    1. Build polar template grid (1024 points, registered buffer)
    2. Global scene descriptor via 1D conv encoder + global max pool -> (B, 1024)
    3. Three DensificationStage modules (1024->2048->4096->8192 points):
       - Local feature sampling from Stage 2 feature map (grid_sample)
       - MLP: [global(1024) || local(128) || pts(3)] -> offsets + confidence
       - Doubling: predict offset_a, offset_b per point -> 2x points

Physics notes:
    - Polar template: r in [0.3, 10.8] m, az in [-70, +70] degrees
    - z=0 (flat ground prior — FMCW radar has poor elevation resolution)
    - r_max=10.8 m matches eval constants from eval/eval_pointcloud.py

References:
    - mmPoint (BMVC 2023): Lift-and-Deform Module for mmWave point cloud densification
    - Plan 03-01: Stage 3 decoder spec
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_polar_template(
    N_r: int = 32,
    N_az: int = 32,
    r_max: float = 10.8,
    az_range_deg: float = 140.0,
) -> torch.Tensor:
    """Build a uniform polar template grid in Cartesian coordinates.

    Constructs a 2D grid of N_r x N_az points in polar space (range × azimuth),
    lifted to flat 3D (z=0) to serve as the initial template for densification.

    Grid:
        range: linspace(0.3, r_max, N_r)    — avoid r=0 singularity
        azimuth: linspace(-az_range_deg/2, +az_range_deg/2, N_az) in degrees
        x = r * cos(az), y = r * sin(az), z = 0

    Args:
        N_r:          Number of range bins (default 32)
        N_az:         Number of azimuth bins (default 32)
        r_max:        Maximum range in meters (default 10.8)
        az_range_deg: Total azimuth span in degrees (default 140.0 -> ±70 deg)

    Returns:
        Template points, shape (N_r * N_az, 3) float32.
        Column order: [x, y, z] where z=0 everywhere.
    """
    r_vals = torch.linspace(0.3, r_max, N_r)               # (N_r,)
    az_vals = torch.linspace(
        -az_range_deg / 2.0,
        az_range_deg / 2.0,
        N_az
    ) * (math.pi / 180.0)                                   # (N_az,) in radians

    # Meshgrid: (N_r, N_az)
    r_grid, az_grid = torch.meshgrid(r_vals, az_vals, indexing="ij")

    x = r_grid * torch.cos(az_grid)  # (N_r, N_az)
    y = r_grid * torch.sin(az_grid)  # (N_r, N_az)
    z = torch.zeros_like(x)          # (N_r, N_az)

    # Flatten to (N_r*N_az, 3) = (1024, 3) for default 32x32 grid
    pts = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1)
    return pts.float()


def sample_features_from_range_map(
    feature_map: torch.Tensor,
    pts_xyz: torch.Tensor,
    r_max: float = 10.8,
) -> torch.Tensor:
    """Sample local features from a 1D range-bin feature map at point locations.

    Converts points to normalized range coordinates and uses grid_sample for
    bilinear (here: 1D linear) interpolation along the range axis.

    Args:
        feature_map: (B, C, 512) float32 range-bin feature map from Stage2Bridge
        pts_xyz:     (B, N, 3) float32 point coordinates [x, y, z]
        r_max:       Maximum range for normalization (default 10.8 m)

    Returns:
        local_feats: (B, N, C) float32 interpolated features at each point
    """
    B, C, R = feature_map.shape
    N = pts_xyz.shape[1]

    # Compute range r = sqrt(x^2 + y^2) for each point
    r = torch.sqrt(pts_xyz[..., 0] ** 2 + pts_xyz[..., 1] ** 2)  # (B, N)

    # Normalize to [-1, +1] for grid_sample (r in [0, r_max] -> [-1, +1])
    # grid_sample convention: -1 = leftmost pixel, +1 = rightmost pixel
    grid_x = (r / r_max) * 2.0 - 1.0  # (B, N), maps [0, r_max] -> [-1, +1]
    grid_x = grid_x.clamp(-1.0, 1.0)

    # grid_sample expects (B, C, H_out, W_out) with grid (B, H_out, W_out, 2)
    # For 1D lookup: treat feature_map as (B, C, 1, R) and grid as (B, N, 1, 2)
    feat_4d = feature_map.unsqueeze(2)  # (B, C, 1, R)

    # grid: (B, N, 1, 2) with (x_coord, y_coord) — y_coord fixed at 0 (single row)
    grid = torch.stack(
        [grid_x, torch.zeros_like(grid_x)], dim=-1
    ).unsqueeze(2)  # (B, N, 1, 2)

    # Sample features: output (B, C, N, 1)
    sampled = F.grid_sample(
        feat_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, N, 1)

    # Reshape to (B, N, C)
    local_feats = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)
    return local_feats


class DensificationStage(nn.Module):
    """One point cloud densification step (doubles point count).

    Given a set of N points conditioned on global + local features,
    predicts two sets of residual offsets (offset_a, offset_b) per point,
    creating 2N deformed points plus a per-point confidence logit.

    Input:
        pts:         (B, N, 3) float32 current points
        global_desc: (B, global_dim) float32 global scene descriptor
        local_feats: (B, N, C_local) float32 local features sampled from feature map

    Output:
        pts_doubled:  (B, 2N, 3) float32 deformed points
        conf_doubled: (B, 2N, 1) float32 confidence logits

    MLP output: 7 channels = offset_a(3) + offset_b(3) + conf(1)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 7),  # offset_a(3) + offset_b(3) + conf(1)
        )

    def forward(
        self,
        pts: torch.Tensor,
        global_desc: torch.Tensor,
        local_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Densify points by predicting two residual offsets per point.

        Args:
            pts:         (B, N, 3) current point coordinates
            global_desc: (B, global_dim) global scene descriptor
            local_feats: (B, N, C_local) locally sampled features

        Returns:
            pts_doubled:  (B, 2N, 3) two deformed copies of each input point
            conf_doubled: (B, 2N, 1) confidence logits (raw, pre-sigmoid)
        """
        B, N, _ = pts.shape

        # Expand global descriptor to per-point: (B, 1024) -> (B, N, 1024)
        g_exp = global_desc.unsqueeze(1).expand(B, N, -1)

        # Concatenate: [global(1024) || local(C_local) || pts(3)]
        feat = torch.cat([g_exp, local_feats, pts], dim=-1)  # (B, N, in_dim)

        # MLP -> 7 channels
        out = self.mlp(feat)  # (B, N, 7)

        offset_a = out[..., :3]    # (B, N, 3) residuals for copy A
        offset_b = out[..., 3:6]   # (B, N, 3) residuals for copy B
        conf_raw = out[..., 6:7]   # (B, N, 1) shared confidence logit

        # Apply residual offsets
        pts_a = pts + offset_a  # (B, N, 3)
        pts_b = pts + offset_b  # (B, N, 3)

        # Concatenate along point dimension -> (B, 2N, 3)
        pts_doubled = torch.cat([pts_a, pts_b], dim=1)

        # Same confidence for both copies: (B, 2N, 1)
        conf_doubled = torch.cat([conf_raw, conf_raw], dim=1)

        return pts_doubled, conf_doubled


class PointCloudDecoder(nn.Module):
    """Stage 3 residual point cloud decoder.

    Maps Stage2Bridge output (B, feature_ch, 512) float32 to a dense
    point cloud (B, 8192, 3) float32 via 3 densification stages:
        Template (1024) -> Stage 3a (2048) -> Stage 3b (4096) -> Stage 3c (8192)

    Architecture:
        1. Global encoder: Conv1d stack + global max pool -> (B, global_dim)
        2. Template: 32x32 polar grid registered as buffer -> (B, 1024, 3)
        3. 3x DensificationStage with local feature sampling at each step

    Stage input dimensions (feature_ch=128, global_dim=1024):
        in_dim = global_dim + feature_ch + 3 = 1024 + 128 + 3 = 1155

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

        # Global encoder: (B, feature_ch, 512) -> (B, global_dim)
        # Conv1d layers process per-range-bin features and aggregate globally
        self.global_encoder = nn.Sequential(
            nn.Conv1d(feature_ch, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, global_dim, kernel_size=1),
        )
        # Global max pool applied in forward after encoder

        # Each DensificationStage: in_dim = global_dim + feature_ch + 3
        stage_in_dim = global_dim + feature_ch + 3  # 1155

        self.stages = nn.ModuleList([
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256),  # 1024 -> 2048
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256),  # 2048 -> 4096
            DensificationStage(in_dim=stage_in_dim, hidden_dim=256),  # 4096 -> 8192
        ])

    def forward(
        self, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode feature map to dense point cloud.

        Args:
            feature_map: (B, feature_ch, 512) float32 from Stage2Bridge

        Returns:
            pts:  (B, 8192, 3) float32 point coordinates
            conf: (B, 8192, 1) float32 confidence logits (raw, pre-sigmoid)
        """
        B = feature_map.shape[0]

        # --- Global scene descriptor ---
        enc = self.global_encoder(feature_map)  # (B, global_dim, 512)
        # Global max pool over range dimension
        global_desc = enc.max(dim=-1).values    # (B, global_dim)

        # --- Initialize from polar template ---
        pts = self.template.unsqueeze(0).expand(B, -1, -1).clone()  # (B, 1024, 3)
        # clone() needed so autograd has a leaf-compatible tensor

        # --- Three densification stages ---
        conf = None
        for stage in self.stages:
            local_feats = sample_features_from_range_map(
                feature_map, pts, r_max=self.r_max
            )  # (B, N, feature_ch)
            pts, conf = stage(pts, global_desc, local_feats)
            # pts: (B, 2N, 3), conf: (B, 2N, 1)

        return pts, conf
