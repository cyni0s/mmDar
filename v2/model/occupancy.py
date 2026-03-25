"""Polar occupancy model for mmDar v2.

Maps raw radar IQ (B, 8, 512) complex64 -> occupancy logits (B, 1, 256, 512)
in polar coordinates (azimuth x range).

Pipeline:
    1. Beamformer  (FFT or LISTA): (B, 8, 512) complex64 -> (B, 256, 512) complex64
    2. Channelizer:                (B, 256, 512) complex64 -> (B, 3, 256, 512) float32
    3. DilatedResHead:             (B, 3, 256, 512) float32 -> (B, 1, 256, 512) float32

Modules:
    Channelizer     — Re/Im/log-power extraction + InstanceNorm2d
    DilatedResBlock — Residual block with dilated Conv2d + GroupNorm
    DilatedResHead  — Sequence of DilatedResBlocks with cycling dilations
    OccupancyModel  — Full pipeline assembly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.lista import FFTBeamformer, LISTABeamformer


class Channelizer(nn.Module):
    """Convert complex angular spectrum to normalized real feature channels.

    Takes (B, A, R) complex64 and produces (B, 3, A, R) float32 where the
    three channels are [Re, Im, log(Re^2 + Im^2 + eps)].

    InstanceNorm2d(3, affine=True) normalizes per-channel across the spatial
    (A x R) dimensions, producing zero-mean unit-variance features per sample.
    Affine=True allows the network to rescale and shift after normalization.

    Args: none (stateless except the InstanceNorm2d affine parameters)
    """

    _EPS: float = 1e-6  # epsilon for log-power to avoid log(0)

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and normalize real/imaginary/log-power channels.

        Args:
            x: Complex angular spectrum, shape (B, A, R) complex64

        Returns:
            Normalized feature map, shape (B, 3, A, R) float32
        """
        re = x.real                              # (B, A, R) float32
        im = x.imag                              # (B, A, R) float32
        log_pwr = torch.log(re ** 2 + im ** 2 + self._EPS)  # (B, A, R) float32

        # Stack to (B, 3, A, R)
        out = torch.stack([re, im, log_pwr], dim=1)  # (B, 3, A, R) float32

        # Per-sample, per-channel spatial normalization
        out = self.norm(out)                     # (B, 3, A, R) float32
        return out


class DilatedResBlock(nn.Module):
    """Residual block with a dilated depthwise-friendly Conv2d followed by a
    1x1 Conv2d to close the block.

    Architecture:
        input -> Conv2d(d=dilation, k=3, p=dilation) -> GroupNorm(8) -> ReLU
              -> Conv2d(d=1, k=3, p=1)               -> GroupNorm(8)
              -> add residual -> ReLU

    All convolutions preserve spatial dimensions (same padding).

    Args:
        ch:      Number of input and output channels (residual block preserves ch)
        dilation: Dilation factor for the first conv (int >= 1)
    """

    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        # GroupNorm groups: min(8, ch) to handle small ch gracefully
        n_groups = min(8, ch)

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(n_groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.norm2 = nn.GroupNorm(n_groups, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        x = F.relu(x + residual, inplace=True)
        return x


class DilatedResHead(nn.Module):
    """Dilated residual convolutional head for polar occupancy prediction.

    Architecture:
        Input proj: Conv2d(in_ch -> mid_ch, k=3, p=1) -> GroupNorm -> ReLU
        n_blocks x DilatedResBlock with cycling dilations [1, 2, 4]
        Output:     Conv2d(mid_ch -> 1, k=1)  — raw logits, no sigmoid

    Args:
        in_ch:   Number of input channels (default 3 from Channelizer)
        mid_ch:  Internal channel width (default 32)
        n_blocks: Number of DilatedResBlocks (default 4)
    """

    _DILATIONS = [1, 2, 4]

    def __init__(self, in_ch: int = 3, mid_ch: int = 32, n_blocks: int = 4) -> None:
        super().__init__()
        n_groups = min(8, mid_ch)

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(n_groups, mid_ch),
            nn.ReLU(inplace=True),
        )

        dilations = [self._DILATIONS[i % len(self._DILATIONS)] for i in range(n_blocks)]
        self.blocks = nn.ModuleList([DilatedResBlock(mid_ch, d) for d in dilations])

        # Output: logits, no activation
        self.out_conv = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Occupancy head forward pass.

        Args:
            x: Feature map, shape (B, in_ch, A, R) float32

        Returns:
            Logits, shape (B, 1, A, R) float32 (pre-sigmoid)
        """
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(x)
        return x


class OccupancyModel(nn.Module):
    """Full polar occupancy model: beamformer -> channelizer -> head -> logits.

    Pipeline:
        1. FFTBeamformer or LISTABeamformer: (B, 8, 512) complex64 -> (B, N_az, 512) complex64
        2. Channelizer:                       (B, N_az, 512) complex64 -> (B, 3, N_az, 512) float32
        3. DilatedResHead:                    (B, 3, N_az, 512) float32 -> (B, 1, N_az, 512) float32

    Args:
        beamformer: "fft" or "lista" (case-insensitive)
        K:          LISTA unrolling layers (only used when beamformer="lista", default 5)
        N_az:       Number of azimuth bins (default 256)
        mid_ch:     DilatedResHead internal width (default 32)
        n_blocks:   Number of DilatedResBlocks (default 4)
    """

    def __init__(
        self,
        beamformer: str = "fft",
        K: int = 5,
        N_az: int = 256,
        mid_ch: int = 32,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()

        bf = beamformer.lower()
        if bf == "fft":
            self.beamformer = FFTBeamformer(N_az=N_az)
        elif bf == "lista":
            self.beamformer = LISTABeamformer(K=K, N_az=N_az)
        else:
            raise ValueError(f"beamformer must be 'fft' or 'lista', got '{beamformer}'")

        self.channelizer = Channelizer()
        self.head = DilatedResHead(in_ch=3, mid_ch=mid_ch, n_blocks=n_blocks)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """End-to-end polar occupancy forward pass.

        Args:
            y: Raw radar input, shape (B, 8, R) complex64

        Returns:
            logits: Polar occupancy logits, shape (B, 1, N_az, R) float32 (pre-sigmoid)
        """
        spec = self.beamformer(y)         # (B, N_az, R) complex64
        feats = self.channelizer(spec)    # (B, 3, N_az, R) float32
        logits = self.head(feats)         # (B, 1, N_az, R) float32
        return logits
