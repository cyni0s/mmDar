"""DETR-style Gaussian set decoder for radar point cloud prediction.

Predicts K Gaussian primitives from encoded radar features.
Each Gaussian: (μ_r, μ_φ, σ_r, σ_perp, existence_logit)

Internal Gaussians, external point cloud: at inference, extract centers
as the output point cloud.

Input:  (B, C, R) spatial features from BeamspaceEncoder
Output: (B, K, 5) Gaussian parameters + (B, K) existence logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSetDecoder(nn.Module):
    """Cross-attention decoder that predicts K Gaussians from spatial features.

    Architecture:
        K learnable queries → N_layers × (self-attn + cross-attn + FFN) → per-query MLP
        → (μ_r, μ_φ, raw_σ_r, raw_σ_perp, existence_logit)

    Args:
        K: number of Gaussian queries (default 96)
        d_model: query/key/value dimension (default 128)
        n_heads: attention heads (default 4)
        n_layers: decoder layers (default 3)
        feat_ch: input feature channels from encoder (default 128)
        r_max: maximum range in meters (default 10.8)
        sigma_r_min: minimum range uncertainty in meters (default 0.02)
        sigma_perp_min_base: minimum perpendicular uncertainty at r=1m (default 0.05)
    """

    def __init__(
        self,
        K: int = 96,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        feat_ch: int = 128,
        r_max: float = 10.8,
        sigma_r_min: float = 0.02,
        sigma_perp_min_base: float = 0.05,
    ):
        super().__init__()
        self.K = K
        self.r_max = r_max
        self.sigma_r_min = sigma_r_min
        self.sigma_perp_min_base = sigma_perp_min_base

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # Project spatial features to d_model
        self.feat_proj = nn.Linear(feat_ch, d_model)

        # Positional encoding for range bins
        self.range_pe = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])

        # Per-query prediction head
        # Output: (μ_r, μ_φ, raw_σ_r, raw_σ_perp, existence_logit)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 5),
        )

    def forward(self, features: torch.Tensor) -> dict:
        """Predict Gaussian set from spatial features.

        Args:
            features: (B, C, R) float32 from BeamspaceEncoder

        Returns:
            dict with:
                'mu_r':     (B, K) range in meters [0, r_max]
                'mu_phi':   (B, K) azimuth in radians [-π/2, π/2]
                'sigma_r':  (B, K) range uncertainty in meters (≥ sigma_r_min)
                'sigma_perp': (B, K) perpendicular uncertainty in meters (≥ sigma_perp_min)
                'existence': (B, K) existence logits (pre-sigmoid)
                'mu_xy':    (B, K, 2) Cartesian centers for eval
        """
        B, C, R = features.shape

        # Project features: (B, R, d_model)
        feat_tokens = self.feat_proj(features.permute(0, 2, 1))  # (B, R, d)
        feat_tokens = feat_tokens + self.range_pe[:, :R, :]

        # Expand queries: (B, K, d_model)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention decoding
        for layer in self.layers:
            queries = layer(queries, feat_tokens)

        # Predict per-query parameters
        raw = self.head(queries)  # (B, K, 5)

        # Decode parameters with physical constraints
        mu_r = torch.sigmoid(raw[:, :, 0]) * self.r_max  # [0, r_max] meters
        mu_phi = torch.tanh(raw[:, :, 1]) * (math.pi / 2)  # [-π/2, π/2] radians
        sigma_r = self.sigma_r_min + F.softplus(raw[:, :, 2])
        # σ_perp scales with range: farther = more angular uncertainty
        sigma_perp_min = self.sigma_perp_min_base * (mu_r.detach() / 1.0).clamp(min=0.5)
        sigma_perp = sigma_perp_min + F.softplus(raw[:, :, 3])
        existence = raw[:, :, 4]  # logits

        # Convert to Cartesian for eval
        mu_x = mu_r * torch.cos(mu_phi)
        mu_y = mu_r * torch.sin(mu_phi)
        mu_xy = torch.stack([mu_x, mu_y], dim=-1)  # (B, K, 2)

        return {
            'mu_r': mu_r,
            'mu_phi': mu_phi,
            'sigma_r': sigma_r,
            'sigma_perp': sigma_perp,
            'existence': existence,
            'mu_xy': mu_xy,
        }


class DecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention among queries
        q = self.ln1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # Cross-attention to spatial features
        q = self.ln2(queries)
        queries = queries + self.cross_attn(q, memory, memory)[0]

        # FFN
        queries = queries + self.ffn(self.ln3(queries))

        return queries


class GaussianRadarModel(nn.Module):
    """Full model: BeamspaceEncoder → GaussianSetDecoder.

    Args:
        N_beam: beamspace bins (default 32)
        T: temporal frames (default 8)
        K: Gaussian queries (default 96)
        hidden_ch: encoder channels (default 128)
    """

    def __init__(self, N_beam=32, T=8, K=96, hidden_ch=128):
        super().__init__()
        from v2.model.beamspace import BeamspaceEncoder
        self.encoder = BeamspaceEncoder(N_beam=N_beam, T=T, hidden_ch=hidden_ch)
        self.decoder = GaussianSetDecoder(K=K, feat_ch=hidden_ch)

    def forward(self, x_seq: torch.Tensor) -> dict:
        """End-to-end: raw IQ → Gaussian parameters.

        Args:
            x_seq: (B, T, 8, R) complex64

        Returns:
            dict with mu_r, mu_phi, sigma_r, sigma_perp, existence, mu_xy
        """
        features = self.encoder(x_seq)  # (B, hidden_ch, R)
        return self.decoder(features)

    def predict_points(self, x_seq: torch.Tensor,
                       threshold: float = 0.0) -> list[torch.Tensor]:
        """Inference: predict point cloud from raw IQ.

        Args:
            x_seq: (B, T, 8, R) complex64
            threshold: existence threshold (sigmoid space)

        Returns:
            list of (N_i, 2) tensors — variable-size point clouds per batch element
        """
        out = self.forward(x_seq)
        existence_prob = torch.sigmoid(out['existence'])  # (B, K)
        mu_xy = out['mu_xy']  # (B, K, 2)

        points = []
        for b in range(mu_xy.shape[0]):
            mask = existence_prob[b] > threshold
            points.append(mu_xy[b, mask])  # (N_b, 2)
        return points
