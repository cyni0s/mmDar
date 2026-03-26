# v2/model — physics-informed complex-valued beamformer modules
"""Full model assembly for mmDar v2.

Combines the three-stage pipeline into two top-level model classes:

    RadarPointCloudModel:
        Stage 1 (LISTABeamformer) -> Stage 2 (Stage2Bridge) -> Stage 3 (PointCloudDecoder)
        Input:  (B, 8, 512) complex64
        Output: (B, 8192, 3) float32 pts, (B, 8192, 1) float32 conf logits

    MagnitudeBaseline:
        Stage 1 (FFTBeamformer) -> real Conv1d bridge -> Stage 3 (PointCloudDecoder)
        Same I/O shape; no learned beamforming, no complex ops.
        Use as fair single-frame comparison (same decoder, no phase).

Helper:
    set_stage1_frozen(model, frozen: bool):
        Freeze/unfreeze Stage 1 (LISTA beamformer) parameters for staged training.
"""

import torch
import torch.nn as nn

from v2.model.cvnn import safe_modulus
from v2.model.lista import FFTBeamformer, LISTABeamformer, Stage2Bridge
from v2.model.decoder import PointCloudDecoder
from v2.model.decoder_2d import PointCloudDecoder2D
from v2.model.occupancy import OccupancyModel, Channelizer, DilatedResHead
from v2.model.temporal import TemporalMagPhaseFusion, TemporalCrossAttention


class RadarPointCloudModel(nn.Module):
    """Full 3-stage model: LISTA beamformer -> Stage2Bridge -> PointCloudDecoder.

    Pipeline:
        1. LISTABeamformer: (B, 8, 512) complex64 -> (B, N_az, 512) complex64
        2. Stage2Bridge:    (B, N_az, 512) complex64 -> (B, bridge_out_ch, 512) float32
        3. PointCloudDecoder: (B, bridge_out_ch, 512) float32 ->
                              pts  (B, 8192, 3) float32
                              conf (B, 8192, 1) float32 logits

    Args:
        K:            Number of LISTA unrolling layers (default 5)
        N_az:         Number of angular bins in beamformer output (default 256)
        bridge_out_ch: Number of output channels from Stage2Bridge (default 128)
    """

    def __init__(self, K: int = 5, N_az: int = 256, bridge_out_ch: int = 128) -> None:
        super().__init__()
        self.beamformer = LISTABeamformer(K=K, N_az=N_az)
        self.bridge = Stage2Bridge(in_ch=N_az, out_ch=bridge_out_ch)
        self.decoder = PointCloudDecoder(feature_ch=bridge_out_ch)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass.

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:  (B, 8192, 3) float32 predicted point cloud
            conf: (B, 8192, 1) float32 per-point confidence logits (pre-sigmoid)
        """
        # Stage 1: LISTA angular super-resolution beamforming
        angular_spec = self.beamformer(y)    # (B, N_az, 512) complex64

        # Stage 2: Complex feature extraction + modulus (complex -> real)
        features = self.bridge(angular_spec)  # (B, bridge_out_ch, 512) float32

        # Stage 3: Residual point cloud decoder
        pts, conf = self.decoder(features)    # (B, 8192, 3), (B, 8192, 1)

        return pts, conf

    def forward_with_intermediates(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning intermediate LISTA output for measurement consistency.

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:          (B, 8192, 3) float32 predicted point cloud
            conf:         (B, 8192, 1) float32 per-point confidence logits
            angular_spec: (B, N_az, 512) complex64 LISTA beamformer output
        """
        angular_spec = self.beamformer(y)
        features = self.bridge(angular_spec)
        pts, conf = self.decoder(features)
        return pts, conf, angular_spec


class MagnitudeBaseline(nn.Module):
    """Single-frame magnitude-only baseline for fair comparison.

    Uses FFTBeamformer (no learned Stage 1) instead of LISTA. The complex
    FFT output is converted to magnitude via safe_modulus before the bridge,
    which uses a real-valued Conv1d (not complex ops).

    This provides a fair comparison:
        - Same decoder architecture as RadarPointCloudModel
        - Same number of angular bins (N_az)
        - No learned beamforming, no phase information
        - No temporal stacking (single-frame, unlike the 41-frame 0.295m baseline)

    Args:
        N_az:         Number of angular bins (default 256); matches LISTABeamformer
        bridge_out_ch: Number of output channels from bridge (default 128)
    """

    def __init__(self, N_az: int = 256, bridge_out_ch: int = 128) -> None:
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        # Real-valued bridge: magnitude input -> feature map
        # N_az channels -> bridge_out_ch channels, GroupNorm + ReLU
        self.bridge = nn.Sequential(
            nn.Conv1d(N_az, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = PointCloudDecoder(feature_ch=bridge_out_ch)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass (magnitude-only, no phase).

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:  (B, 8192, 3) float32 predicted point cloud
            conf: (B, 8192, 1) float32 per-point confidence logits (pre-sigmoid)
        """
        # Stage 1: FFT beamforming (no learning)
        spec = self.beamformer(y)              # (B, N_az, 512) complex64

        # Discard phase — take magnitude only
        spec_mag = safe_modulus(spec)          # (B, N_az, 512) float32, non-negative

        # Real-valued bridge: Conv1d + GroupNorm + ReLU
        features = self.bridge(spec_mag)       # (B, bridge_out_ch, 512) float32

        # Stage 3: Same decoder as RadarPointCloudModel
        pts, conf = self.decoder(features)

        return pts, conf


class MagnitudePhaseFusion(nn.Module):
    """Magnitude + phase channels baseline for phase ablation.

    Same as MagnitudeBaseline but feeds 3 real channels per angular bin:
        - magnitude: |FFT|
        - sin(angle(FFT))
        - cos(angle(FFT))

    This tests whether phase helps in the simplest possible way — no CVNN,
    no LISTA, no complex arithmetic. Just phase as extra real features.

    Input channels: N_az * 3 (magnitude + sin_phase + cos_phase)
    Bridge reduces to bridge_out_ch before the same decoder.

    Args:
        N_az:         Number of angular bins (default 256)
        bridge_out_ch: Number of output channels from bridge (default 128)
    """

    def __init__(self, N_az: int = 256, bridge_out_ch: int = 128) -> None:
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        # 3x channels: mag + sin(phase) + cos(phase)
        # Use two Conv1d layers to reduce 3*N_az -> bridge_out_ch
        self.bridge = nn.Sequential(
            nn.Conv1d(N_az * 3, N_az, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(N_az, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = PointCloudDecoder(feature_ch=bridge_out_ch)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: FFT -> magnitude + phase channels -> decoder.

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:  (B, 8192, 3) float32
            conf: (B, 8192, 1) float32
        """
        spec = self.beamformer(y)                  # (B, N_az, 512) complex64
        mag = safe_modulus(spec)                   # (B, N_az, 512) float32
        phase = torch.angle(spec)                  # (B, N_az, 512) float32
        sin_ph = torch.sin(phase)                  # (B, N_az, 512)
        cos_ph = torch.cos(phase)                  # (B, N_az, 512)

        # Gate phase by magnitude — low-SNR bins get zero phase contribution
        gate = (mag > mag.mean(dim=1, keepdim=True) * 0.1).float()
        sin_ph = sin_ph * gate
        cos_ph = cos_ph * gate

        fused = torch.cat([mag, sin_ph, cos_ph], dim=1)  # (B, 3*N_az, 512)
        features = self.bridge(fused)              # (B, bridge_out_ch, 512)
        pts, conf = self.decoder(features)
        return pts, conf


class MagnitudeBaseline2D(nn.Module):
    """Single-frame magnitude baseline with 2D angular topology preserved.

    Fixes the angular collapse bug: bridge uses Conv2d to maintain (azimuth x range)
    spatial layout. The decoder can now distinguish features at different azimuths.

    Pipeline:
        1. FFTBeamformer: (B, 8, 512) complex64 -> (B, 256, 512) complex64
        2. safe_modulus -> (B, 256, 512) float32
        3. Conv2d bridge: (B, 1, 256, 512) -> (B, bridge_out_ch, 256, 512)
        4. PointCloudDecoder2D: (B, bridge_out_ch, 256, 512) -> pts + conf

    Args:
        N_az:          Number of angular bins (default 256)
        bridge_out_ch: Number of output channels from bridge (default 128)
    """

    def __init__(self, N_az: int = 256, bridge_out_ch: int = 128) -> None:
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        # 2D bridge: (B, 1, 256, 512) -> (B, bridge_out_ch, 256, 512)
        self.bridge = nn.Sequential(
            nn.Conv2d(1, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = PointCloudDecoder2D(feature_ch=bridge_out_ch)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass (magnitude-only, 2D angular topology).

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:  (B, 8192, 3) float32 predicted point cloud
            conf: (B, 8192, 1) float32 per-point confidence logits (pre-sigmoid)
        """
        spec = self.beamformer(y)              # (B, 256, 512) complex64
        mag = safe_modulus(spec)               # (B, 256, 512) float32, non-negative
        mag_2d = mag.unsqueeze(1)              # (B, 1, 256, 512) — explicit 2D
        features = self.bridge(mag_2d)         # (B, bridge_out_ch, 256, 512) — 2D preserved!
        pts, conf = self.decoder(features)
        return pts, conf


class MagnitudePhaseFusion2D(nn.Module):
    """Magnitude + phase with 2D angular topology preserved.

    Same as MagnitudeBaseline2D but feeds 3 real channels per angular position:
        - magnitude: |FFT|
        - sin(angle(FFT)), gated by magnitude
        - cos(angle(FFT)), gated by magnitude

    Pipeline:
        1. FFTBeamformer: (B, 8, 512) complex64 -> (B, 256, 512) complex64
        2. Extract mag, sin(phase), cos(phase) -> stack -> (B, 3, 256, 512)
        3. Conv2d bridge: (B, 3, 256, 512) -> (B, bridge_out_ch, 256, 512)
        4. PointCloudDecoder2D: (B, bridge_out_ch, 256, 512) -> pts + conf

    Args:
        N_az:          Number of angular bins (default 256)
        bridge_out_ch: Number of output channels from bridge (default 128)
    """

    def __init__(self, N_az: int = 256, bridge_out_ch: int = 128) -> None:
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        # 3 channels: mag + sin_phase + cos_phase, all in 2D
        self.bridge = nn.Sequential(
            nn.Conv2d(3, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = PointCloudDecoder2D(feature_ch=bridge_out_ch)

    def forward(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward: FFT -> magnitude + phase channels (2D) -> decoder.

        Args:
            y: Raw radar input, shape (B, 8, 512) complex64

        Returns:
            pts:  (B, 8192, 3) float32
            conf: (B, 8192, 1) float32
        """
        spec = self.beamformer(y)                  # (B, 256, 512) complex64
        mag = safe_modulus(spec)                    # (B, 256, 512) float32
        phase = torch.angle(spec)                  # (B, 256, 512) float32
        sin_ph = torch.sin(phase)                  # (B, 256, 512)
        cos_ph = torch.cos(phase)                  # (B, 256, 512)

        # Gate phase by magnitude — low-SNR bins get zero phase contribution
        gate = (mag > mag.mean(dim=1, keepdim=True) * 0.1).float()
        sin_ph = sin_ph * gate
        cos_ph = cos_ph * gate

        fused = torch.stack([mag, sin_ph, cos_ph], dim=1)  # (B, 3, 256, 512)
        features = self.bridge(fused)              # (B, bridge_out_ch, 256, 512)
        pts, conf = self.decoder(features)
        return pts, conf


def set_stage1_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze or unfreeze Stage 1 (LISTA beamformer) parameters.

    During staged training: freeze Stage 1 for the first N epochs so the
    decoder learns from a fixed beamformer, then unfreeze for joint fine-tuning.

    For RadarPointCloudModel: freezes model.beamformer parameters.
    For MagnitudeBaseline: no-op (FFTBeamformer has no learnable parameters).

    Args:
        model:  A RadarPointCloudModel or MagnitudeBaseline instance
        frozen: If True, beamformer params are frozen (requires_grad=False).
                If False, beamformer params are unfrozen (requires_grad=True).
    """
    if not hasattr(model, "beamformer"):
        return
    for p in model.beamformer.parameters():
        p.requires_grad = not frozen
