"""Physics-first 2D frontend: classical FFT beamformer + light 2D encoder.

Uses classical signal processing (FFT) to produce a 2D range-azimuth map,
then a light 2D convolutional encoder extracts spatial features while
preserving the 2D angular structure. Finally collapses to per-range-bin
tokens for the DETR Gaussian decoder.

The network learns the DELTA on top of classical physics, not physics itself.

Input:  (B, T, 8, R) complex64 — T frames of 8 antennas × R range bins
Output: (B, out_ch, R) float32 — per-range-bin features for Gaussian decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalFFTFrontend(nn.Module):
    """Fixed FFT beamformer — no trainable parameters.

    Per frame: (8, R) complex → FFT along antennas → (N_az, R) complex
    → [magnitude, log_power] → (2, N_az, R) float

    Args:
        N_az: number of azimuth bins (default 64 — matched to sensor resolution)
    """

    def __init__(self, N_az: int = 64):
        super().__init__()
        self.N_az = N_az

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFT beamform a single frame.

        Args:
            x: (B, 8, R) complex64

        Returns:
            (B, 2, N_az, R) float32 — [magnitude, log_power]
        """
        # FFT along antenna dim, zero-pad to N_az
        spectrum = torch.fft.fft(x, n=self.N_az, dim=1)  # (B, N_az, R) complex
        spectrum = torch.fft.fftshift(spectrum, dim=1)     # DC at center

        mag = spectrum.abs().float()                        # (B, N_az, R)
        log_pow = torch.log(mag ** 2 + 1e-6)              # (B, N_az, R)

        return torch.stack([mag, log_pow], dim=1)          # (B, 2, N_az, R)


class Light2DEncoder(nn.Module):
    """Light 2D convolutional encoder on range-azimuth maps.

    Preserves the 2D spatial structure (azimuth × range) through several
    conv layers, then collapses azimuth to produce per-range-bin features.

    Args:
        in_ch: input channels (T * 2 for mag + logpow per frame)
        mid_ch: intermediate 2D conv channels (default 64)
        out_ch: output per-range-bin features after azimuth collapse (default 128)
        N_az: azimuth bins (default 64)
    """

    def __init__(self, in_ch: int, mid_ch: int = 64, out_ch: int = 128,
                 N_az: int = 64):
        super().__init__()
        self.N_az = N_az

        # 2D conv blocks on (azimuth × range) — preserves both dimensions
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.GELU(),
        )

        # Collapse azimuth to per-range features
        # After 2D conv: (B, mid_ch, N_az, R)
        # Collapse: (B, mid_ch * N_az, R) via reshape, then project down
        # But mid_ch * N_az = 64 * 64 = 4096 — too large for direct projection
        # Instead: use adaptive pooling along azimuth → (B, mid_ch, pool_az, R)
        # then flatten
        self.pool_az = 8  # pool 64 azimuth bins down to 8
        self.collapse = nn.Sequential(
            nn.Conv1d(mid_ch * self.pool_az, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(16, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 2D range-azimuth features to per-range tokens.

        Args:
            x: (B, C, N_az, R) float32

        Returns:
            (B, out_ch, R) float32 — per-range-bin features
        """
        B, C, A, R = x.shape

        # 2D spatial processing
        feat_2d = self.conv2d(x)  # (B, mid_ch, N_az, R)

        # Adaptive pool along azimuth: (B, mid_ch, pool_az, R)
        feat_pooled = F.adaptive_avg_pool2d(feat_2d, (self.pool_az, R))

        # Reshape: (B, mid_ch * pool_az, R)
        feat_1d = feat_pooled.view(B, -1, R)

        # Project to output dim
        return self.collapse(feat_1d)  # (B, out_ch, R)


class PhysicsFirstEncoder(nn.Module):
    """Full physics-first frontend: classical FFT → 2D encoder → 1D deep encoder.

    Combines:
    1. Classical FFT beamformer (fixed, no learning) → 2D range-azimuth map
    2. Light 2D conv encoder → preserves angular patterns
    3. Azimuth collapse → per-range-bin tokens
    4. Deep 1D dilated residual encoder → range context

    Args:
        N_az: FFT azimuth bins (default 64)
        T: temporal frames (default 41)
        mid_2d_ch: 2D encoder channels (default 64)
        hidden_1d_ch: 1D encoder channels (default 192)
        out_ch: final output channels for DETR decoder (default 128)
        n_blocks_1d: number of 1D dilated residual blocks (default 8)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, N_az: int = 64, T: int = 41, mid_2d_ch: int = 64,
                 hidden_1d_ch: int = 192, out_ch: int = 128,
                 n_blocks_1d: int = 8, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.fft = ClassicalFFTFrontend(N_az=N_az)

        # 2D encoder: T*2 input channels (mag + logpow per frame)
        in_2d_ch = T * 2
        self.encoder_2d = Light2DEncoder(
            in_ch=in_2d_ch, mid_ch=mid_2d_ch, out_ch=out_ch, N_az=N_az,
        )

        # Deep 1D encoder on top (reuse DilatedResBlock1d from beamspace.py)
        from v2.model.beamspace import DilatedResBlock1d

        n_groups = min(32, hidden_1d_ch)
        self.input_proj = nn.Sequential(
            nn.Conv1d(out_ch, hidden_1d_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(n_groups, hidden_1d_ch),
            nn.GELU(),
        )

        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:n_blocks_1d]
        self.blocks_1d = nn.ModuleList([
            DilatedResBlock1d(hidden_1d_ch, d, kernel_size=5, dropout=dropout)
            for d in dilations
        ])

        self.output_proj = nn.Conv1d(hidden_1d_ch, out_ch, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode multi-frame raw IQ to per-range-bin features.

        Args:
            x_seq: (B, T, 8, R) complex64

        Returns:
            (B, out_ch, R) float32
        """
        B, T, A, R = x_seq.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # Step 1: classical FFT per frame → stack as channels
        frame_feats = []
        for t in range(T):
            frame_feats.append(self.fft(x_seq[:, t]))  # (B, 2, N_az, R)
        stacked_2d = torch.cat(frame_feats, dim=1)     # (B, T*2, N_az, R)

        # Step 2: 2D encoder → collapse azimuth → per-range tokens
        range_tokens = self.encoder_2d(stacked_2d)      # (B, out_ch, R)

        # Step 3: deep 1D encoder for range context
        x = self.input_proj(range_tokens)
        for block in self.blocks_1d:
            x = block(x)
        return self.output_proj(x)                       # (B, out_ch, R)


class PhysicsGaussianModel(nn.Module):
    """Full model: PhysicsFirstEncoder → GaussianSetDecoder.

    Classical FFT (fixed) → 2D conv (light) → 1D encoder (deep) → DETR → Gaussians.

    Args:
        N_az: FFT azimuth bins (default 64)
        T: temporal frames (default 41)
        K: Gaussian queries (default 96)
        out_ch: encoder output / decoder input channels (default 128)
    """

    def __init__(self, N_az=64, T=41, K=96, out_ch=128):
        super().__init__()
        self.encoder = PhysicsFirstEncoder(N_az=N_az, T=T, out_ch=out_ch)
        from v2.model.gaussian_head import GaussianSetDecoder
        self.decoder = GaussianSetDecoder(K=K, feat_ch=out_ch)

    def forward(self, x_seq: torch.Tensor) -> dict:
        features = self.encoder(x_seq)
        return self.decoder(features)

    def predict_points(self, x_seq: torch.Tensor,
                       threshold: float = 0.0) -> list[torch.Tensor]:
        out = self.forward(x_seq)
        existence_prob = torch.sigmoid(out['existence'])
        mu_xy = out['mu_xy']
        points = []
        for b in range(mu_xy.shape[0]):
            mask = existence_prob[b] > threshold
            points.append(mu_xy[b, mask])
        return points
