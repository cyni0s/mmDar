# 1D azimuth super-resolution model for radar-to-lidar translation.
#
# Tests hypothesis: is 64→512 azimuth sharpening fundamentally a per-range-bin
# 1D problem, or does cross-range 2D context matter?
#
# The model reshapes (B, C, 256, 64) → (B*256, C, 64), processes each range bin
# independently with 1D convolutions, upsamples 64→512, and reshapes back.

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """Residual block with two 1D convolutions and optional channel change."""

    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Azimuth1DNet(nn.Module):
    """Per-range-bin 1D azimuth super-resolution: (B, C_in, 256, 64) → (B, 1, 256, 512).

    Processes each of 256 range bins independently via 1D convolutions along azimuth.
    Three 2× upsample stages (64 → 128 → 256 → 512) using sub-pixel convolution.

    Args:
        n_channels: number of input channels (41 for stacked radar frames, 1 for single-frame)
        hidden: hidden channel width (default 128)
        n_blocks: number of residual blocks (default 6)
    """

    def __init__(self, n_channels=41, hidden=128, n_blocks=6):
        super().__init__()
        self.n_channels = n_channels

        # Input projection: (n_channels, 64) → (hidden, 64)
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing dilation to cover full 64-bin receptive field
        blocks = []
        for i in range(n_blocks):
            dilation = min(2 ** i, 8)  # 1, 2, 4, 8, 8, 8
            blocks.append(ResBlock1D(hidden, kernel_size=7, dilation=dilation))
        self.res_blocks = nn.Sequential(*blocks)

        # Three 2× upsample stages: 64 → 128 → 256 → 512
        # Sub-pixel convolution: conv to 2× channels, then PixelShuffle-1D
        self.up1 = self._make_upsample(hidden, hidden)
        self.up2 = self._make_upsample(hidden, hidden)
        self.up3 = self._make_upsample(hidden, hidden)

        # Output projection: (hidden, 512) → (1, 512)
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _make_upsample(in_ch, out_ch):
        """Sub-pixel 1D upsampling: conv to 2× channels, then reshape."""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch * 2),
            nn.ReLU(inplace=True),
            PixelShuffle1D(upscale_factor=2),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C_in, 256, 64) — stacked radar input (standard baseline format)

        Returns:
            out: (B, 1, 256, 512) — predicted lidar occupancy
        """
        B, C, R, A = x.shape  # B, 41, 256, 64

        # Reshape: treat each range bin as an independent sample
        # (B, C, 256, 64) → (B, 256, C, 64) → (B*256, C, 64)
        x = x.permute(0, 2, 1, 3).reshape(B * R, C, A)

        # 1D processing along azimuth
        x = self.input_proj(x)       # (B*256, hidden, 64)
        x = self.res_blocks(x)       # (B*256, hidden, 64)
        x = self.up1(x)              # (B*256, hidden, 128)
        x = self.up2(x)              # (B*256, hidden, 256)
        x = self.up3(x)              # (B*256, hidden, 512)
        x = self.output_proj(x)      # (B*256, 1, 512)

        # Reshape back: (B*256, 1, 512) → (B, 1, 256, 512)
        out = x.reshape(B, R, 1, 512).permute(0, 2, 1, 3)
        return out


class PixelShuffle1D(nn.Module):
    """1D pixel shuffle (sub-pixel convolution): rearrange channels into length.

    Input:  (N, C*r, L)
    Output: (N, C, L*r)
    """

    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        N, C_r, L = x.shape
        C = C_r // self.r
        # (N, C*r, L) → (N, C, r, L) → (N, C, L*r)
        x = x.reshape(N, C, self.r, L)
        x = x.permute(0, 1, 3, 2).reshape(N, C, L * self.r)
        return x
