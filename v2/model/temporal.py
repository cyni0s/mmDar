"""Temporal cross-attention for multi-frame radar fusion.

Per-range-bin cross-attention: current frame queries past frames.
Residual design: fused = current + delta, so N=1 is identity.

Architecture:
    Per-frame: FFT -> mag/sin/cos -> Conv1d bridge -> (128, 512)
    Temporal: per-range-bin cross-attention with lag encoding
    Decoder: unchanged PointCloudDecoder

Key design decisions:
    1. N=1 returns current unchanged (identity by construction)
    2. CrossAttnBlock returns DELTA ONLY (no internal residual)
       Outer module does current + delta to avoid double-add.
    3. Learnable relative lag encoding added to KV
    4. 1 attention layer, d_model=128, n_heads=4, ff_dim=256
    5. Per-range-bin: (B*R, 1, C) query vs (B*R, N-1, C) key-values
"""

import torch
import torch.nn as nn

from v2.model.cvnn import safe_modulus
from v2.model.lista import FFTBeamformer
from v2.model.decoder import PointCloudDecoder


class CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention block returning DELTA only.

    No internal residual -- the outer TemporalCrossAttention does current + delta.
    This prevents double-add of the current frame.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 ff_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Returns delta only (no residual).

        Args:
            q:  (B*R, 1, C) query from current frame
            kv: (B*R, N-1, C) key-values from history frames

        Returns:
            delta: (B*R, 1, C) attention-weighted update
        """
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        delta = self.ffn(self.ln_ff(attn_out))
        return delta


class TemporalCrossAttention(nn.Module):
    """Per-range-bin temporal cross-attention with residual design.

    N=1: returns current unchanged (identity by construction).
    N>1: current + delta where delta comes from cross-attending to history.

    Args:
        d_model: feature dim (128)
        n_heads: attention heads (4)
        ff_dim: FFN hidden dim (256)
        max_lag: max history length (16)
        dropout: dropout rate (0.1)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 ff_dim: int = 256, max_lag: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.block = CrossAttnBlock(d_model, n_heads, ff_dim, dropout)
        self.lag_embed = nn.Embedding(max_lag, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-range-bin temporal fusion.

        Args:
            x: (B, N, C, R) -- N frames, last = current

        Returns:
            (B, C, R) -- fused features
        """
        B, N, C, R = x.shape
        current = x[:, -1]  # (B, C, R)

        if N == 1:
            return current

        history = x[:, :-1]  # (B, N-1, C, R)
        assert N - 1 <= self.lag_embed.num_embeddings, \
            f"History length {N-1} exceeds max_lag {self.lag_embed.num_embeddings}"

        # Per-range-bin: each range bin is independent
        # q: current frame features per range bin
        q = current.permute(0, 2, 1).reshape(B * R, 1, C)        # (B*R, 1, C)
        # kv: history frames per range bin
        kv = history.permute(0, 3, 1, 2).reshape(B * R, N - 1, C)  # (B*R, N-1, C)

        # Learnable lag encoding: most recent history = lag 0, oldest = lag N-2
        lag_indices = torch.arange(N - 1, device=x.device).flip(0)
        lag_enc = self.lag_embed(lag_indices)  # (N-1, C)
        kv = kv + lag_enc.unsqueeze(0)

        # Cross-attention -> delta
        delta = self.block(q, kv)  # (B*R, 1, C)

        # Reshape and add residual
        delta = delta.squeeze(1).reshape(B, R, C).permute(0, 2, 1)  # (B, C, R)
        return current + delta


class TemporalMagPhaseFusion(nn.Module):
    """MagnitudePhaseFusion + temporal cross-attention.

    Per-frame: FFT -> mag/sin/cos -> Conv1d bridge -> (128, 512)
    Temporal: cross-attention across N frames -> (128, 512)
    Decoder: unchanged PointCloudDecoder -> (8192, 3)

    Args:
        N_az: Number of angular bins for FFT beamformer (default 256)
        bridge_out_ch: Output channels from bridge / d_model (default 128)
        max_lag: Maximum history length for lag encoding (default 16)
    """

    def __init__(self, N_az: int = 256, bridge_out_ch: int = 128,
                 max_lag: int = 16) -> None:
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        self.bridge = nn.Sequential(
            nn.Conv1d(N_az * 3, N_az, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(N_az, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = TemporalCrossAttention(
            d_model=bridge_out_ch, n_heads=4, ff_dim=256, max_lag=max_lag
        )
        self.decoder = PointCloudDecoder(feature_ch=bridge_out_ch)

    def forward(
        self, y_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end temporal fusion forward pass.

        Args:
            y_seq: (B, N, 8, 512) complex64 -- N frames of raw radar

        Returns:
            pts:  (B, 8192, 3) float32 predicted point cloud
            conf: (B, 8192, 1) float32 per-point confidence logits
        """
        B, N, A, R = y_seq.shape
        y_flat = y_seq.reshape(B * N, A, R)

        # Per-frame beamforming + feature extraction
        spec = self.beamformer(y_flat)             # (B*N, N_az, 512) complex64
        mag = safe_modulus(spec)                   # (B*N, N_az, 512) float32
        phase = torch.angle(spec)                  # (B*N, N_az, 512) float32
        sin_ph = torch.sin(phase)
        cos_ph = torch.cos(phase)

        # Gate phase by magnitude -- low-SNR bins get zero phase contribution
        gate = (mag > mag.mean(dim=1, keepdim=True) * 0.1).float()
        fused_input = torch.cat([mag, sin_ph * gate, cos_ph * gate], dim=1)
        features = self.bridge(fused_input)        # (B*N, 128, 512)

        # Reshape for temporal fusion
        features = features.reshape(B, N, -1, R)  # (B, N, 128, 512)
        fused = self.temporal(features)            # (B, 128, 512)

        # Decode to point cloud
        pts, conf = self.decoder(fused)
        return pts, conf

    def load_single_frame_weights(self, checkpoint_path: str) -> None:
        """Load from single-frame MagnitudePhaseFusion checkpoint.

        Handles: ckpt["model_state_dict"], ckpt["state_dict"], raw state_dict.
        Strips "module." prefix. temporal.* stays at random init.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            src_state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            src_state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(
            k.startswith(("beamformer", "bridge", "decoder")) for k in ckpt
        ):
            src_state = ckpt
        else:
            raise ValueError(
                f"Cannot find state dict. Keys: {list(ckpt.keys())[:10]}"
            )

        # Strip DataParallel "module." prefix if present
        src_state = {k.removeprefix("module."): v for k, v in src_state.items()}
        own_state = self.state_dict()

        loaded, skipped, mismatched = 0, 0, 0
        for k, v in src_state.items():
            if k not in own_state:
                skipped += 1
                continue
            if own_state[k].shape != v.shape:
                print(f"  [WARN] Shape mismatch: {k}: "
                      f"ckpt={v.shape} vs model={own_state[k].shape}")
                mismatched += 1
                continue
            own_state[k] = v
            loaded += 1

        self.load_state_dict(own_state)
        temporal_params = sum(1 for k in own_state if k.startswith("temporal."))
        print(
            f"[load_single_frame_weights] Loaded {loaded}, skipped {skipped}, "
            f"mismatched {mismatched}, temporal (random init): {temporal_params}"
        )
