"""Physics-first 2D frontend: classical FFT beamformer + deep 2D encoder.

Uses classical signal processing (FFT) to produce a 2D range-azimuth map,
then a deep 2D convolutional encoder extracts spatial features while
PRESERVING the 2D angular structure all the way to the DETR decoder.

No 1D collapse — the DETR queries attend to 2D spatial tokens.

Input:  (B, T, 8, R) complex64 — T frames of 8 antennas × R range bins
Output: (B, N_tokens, C) float32 — 2D spatial tokens for DETR cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalFFTFrontend(nn.Module):
    """Fixed FFT beamformer — no trainable parameters.

    Args:
        N_az: number of azimuth bins (default 64)
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
        spectrum = torch.fft.fft(x, n=self.N_az, dim=1)
        spectrum = torch.fft.fftshift(spectrum, dim=1)
        mag = spectrum.abs().float()
        log_pow = torch.log(mag ** 2 + 1e-6)
        return torch.stack([mag, log_pow], dim=1)


class ResBlock2d(nn.Module):
    """Residual 2D conv block with GroupNorm + GELU."""

    def __init__(self, ch: int, dropout: float = 0.1):
        super().__init__()
        n_groups = min(16, ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock2d(nn.Module):
    """Downsample 2x in both dims + ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        n_groups = min(16, out_ch)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.GELU(),
        )
        self.res = ResBlock2d(out_ch, dropout)

    def forward(self, x):
        return self.res(self.down(x))


class Deep2DEncoder(nn.Module):
    """Deep 2D encoder that preserves spatial structure.

    Progressively downsamples (az, range) while increasing channels:
      (T*2, 64, 512) → (64, 32, 256) → (128, 16, 128) → (192, 8, 64)

    Output: flattened spatial tokens (B, 8*64, 192) = (B, 512, 192)
    Each token retains its 2D position via learned positional encoding.

    Args:
        in_ch: input channels (T * 2)
        channels: list of channel widths per level (default [64, 128, 192])
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, in_ch: int, channels: list = None,
                 dropout: float = 0.1):
        super().__init__()
        if channels is None:
            channels = [64, 128, 192]

        # Input projection (no downsampling)
        n_groups = min(16, channels[0])
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(n_groups, channels[0]),
            nn.GELU(),
        )
        self.input_res = ResBlock2d(channels[0], dropout)

        # Downsampling levels
        self.downs = nn.ModuleList()
        for i in range(1, len(channels)):
            self.downs.append(DownBlock2d(channels[i-1], channels[i], dropout))

        self.out_ch = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 2D feature map to spatial tokens.

        Args:
            x: (B, in_ch, H, W) float32

        Returns:
            (B, N_tokens, out_ch) float32 — flattened 2D spatial tokens
        """
        x = self.input_res(self.input_proj(x))
        for down in self.downs:
            x = down(x)
        # x: (B, out_ch, H', W')
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, C, H*W) → (B, H*W, C)
        return x.view(B, C, H * W).permute(0, 2, 1)


class PhysicsFirstEncoder(nn.Module):
    """Classical FFT → stack frames → deep 2D encoder → 2D spatial tokens.

    No 1D collapse. The DETR decoder attends to 2D spatial tokens that
    preserve both azimuth and range structure.

    Args:
        N_az: FFT azimuth bins (default 64)
        T: temporal frames (default 41)
        channels_2d: channel widths per encoder level (default [64, 128, 192])
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, N_az: int = 64, T: int = 41,
                 channels_2d: list = None, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.N_az = N_az
        self.fft = ClassicalFFTFrontend(N_az=N_az)

        if channels_2d is None:
            channels_2d = [64, 128, 192]

        in_ch = T * 2  # mag + logpow per frame
        self.encoder_2d = Deep2DEncoder(in_ch, channels_2d, dropout)
        self.out_ch = self.encoder_2d.out_ch

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode multi-frame raw IQ to 2D spatial tokens.

        Args:
            x_seq: (B, T, 8, R) complex64

        Returns:
            (B, N_tokens, out_ch) float32 — 2D spatial tokens
            With default settings: (B, 512, 192) tokens
            (from 64az×512r → 32×256 → 16×128 → 8×64 = 512 tokens)
        """
        B, T, A, R = x_seq.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # Classical FFT per frame → stack as channels
        frame_feats = []
        for t in range(T):
            frame_feats.append(self.fft(x_seq[:, t]))  # (B, 2, N_az, R)
        stacked_2d = torch.cat(frame_feats, dim=1)     # (B, T*2, N_az, R)

        # Deep 2D encoder → spatial tokens
        return self.encoder_2d(stacked_2d)              # (B, N_tokens, out_ch)


class PhysicsGaussianModel(nn.Module):
    """Full model: PhysicsFirstEncoder → GaussianSetDecoder.

    Classical FFT (fixed) → 2D encoder (deep, preserves structure) → DETR → Gaussians.

    The DETR decoder attends to 2D spatial tokens, not 1D range tokens.

    Args:
        N_az: FFT azimuth bins (default 64)
        T: temporal frames (default 41)
        K: Gaussian queries (default 96)
    """

    def __init__(self, N_az=64, T=41, K=96):
        super().__init__()
        self.encoder = PhysicsFirstEncoder(N_az=N_az, T=T)
        from v2.model.gaussian_head import GaussianSetDecoder
        self.decoder = GaussianSetDecoder(K=K, feat_ch=self.encoder.out_ch)

    def forward(self, x_seq: torch.Tensor) -> dict:
        tokens = self.encoder(x_seq)        # (B, N_tokens, C)
        return self.decoder(tokens)

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
