"""Learned beamspace frontend for raw IQ radar data.

Replaces fixed FFT/LISTA beamforming with a trainable complex linear layer
initialized from the ULA steering matrix. Adds phase-difference features
for inter-antenna phase relationships.

Input:  (B, T, 8, R) complex64 — T frames of 8 antennas × R range bins
Output: (B, C, R) float32 — fused spatial features for the decoder
"""

import numpy as np
import torch
import torch.nn as nn

from v2.model.lista import build_steering_matrix


class LearnedBeamspace(nn.Module):
    """Trainable complex beamspace projection + phase-difference features.

    Per range bin per frame:
      Branch 1: W·x → (N_beam,) complex → [Re, Im, log|·|²] → (3*N_beam,)
      Branch 2: adjacent antenna phase diffs → (7,) real features

    Total per-frame features: 3*N_beam + 7

    Args:
        N_beam: number of beamspace output bins (default 32)
        N_ant: number of antennas (default 8)
    """

    def __init__(self, N_beam: int = 32, N_ant: int = 8):
        super().__init__()
        self.N_beam = N_beam
        self.N_ant = N_ant

        # Initialize W from first N_beam columns of steering matrix
        A = build_steering_matrix(N_az=max(N_beam * 4, 64))  # (8, N_az)
        # Select N_beam evenly-spaced columns
        indices = torch.linspace(0, A.shape[1] - 1, N_beam).long()
        W_init = A[:, indices].T.contiguous()  # (N_beam, 8) complex
        self.W_real = nn.Parameter(W_init.real.float())
        self.W_imag = nn.Parameter(W_init.imag.float())

        self.feat_dim = 3 * N_beam + (N_ant - 1)  # Re, Im, logpow + phase diffs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw antenna data to beamspace features.

        Args:
            x: (B, N_ant, R) complex64 — single frame

        Returns:
            (B, feat_dim, R) float32
        """
        # Complex matrix multiply: W @ x
        # W: (N_beam, N_ant), x: (B, N_ant, R)
        x_real = x.real.float()  # (B, 8, R)
        x_imag = x.imag.float()  # (B, 8, R)

        # (N_beam, 8) × (B, 8, R) → (B, N_beam, R)
        bf_real = torch.einsum('ba,bar->br', self.W_real, x_real) - \
                  torch.einsum('ba,bar->br', self.W_imag, x_imag)
        bf_imag = torch.einsum('ba,bar->br', self.W_real, x_imag) + \
                  torch.einsum('ba,bar->br', self.W_imag, x_real)

        # Reshape to (B, N_beam, R)
        B, R = x_real.shape[0], x_real.shape[2]
        bf_real = bf_real.view(B, self.N_beam, R)
        bf_imag = bf_imag.view(B, self.N_beam, R)

        log_pow = torch.log(bf_real ** 2 + bf_imag ** 2 + 1e-6)

        # Phase differences between adjacent antennas: angle(x[n+1] * conj(x[n]))
        phase_diffs = torch.angle(
            x[:, 1:, :] * x[:, :-1, :].conj()
        ).float()  # (B, 7, R)

        # Concatenate: (B, 3*N_beam + 7, R)
        features = torch.cat([bf_real, bf_imag, log_pow, phase_diffs], dim=1)
        return features


class BeamspaceEncoder(nn.Module):
    """Full frontend: beamspace + temporal stacking + 1D conv encoder.

    Processes T frames through LearnedBeamspace, stacks temporally,
    then applies 1D convolutions across range for spatial context.

    Args:
        N_beam: beamspace bins (default 32)
        T: number of temporal frames (default 8)
        hidden_ch: encoder hidden channels (default 128)
    """

    def __init__(self, N_beam: int = 32, T: int = 8, hidden_ch: int = 128):
        super().__init__()
        self.beamspace = LearnedBeamspace(N_beam=N_beam)
        self.T = T

        in_ch = self.beamspace.feat_dim * T  # all frames stacked as channels

        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encode multi-frame raw IQ to spatial features.

        Args:
            x_seq: (B, T, 8, R) complex64

        Returns:
            (B, hidden_ch, R) float32 — per-range-bin features
        """
        B, T, A, R = x_seq.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # Process each frame through beamspace
        frame_feats = []
        for t in range(T):
            frame_feats.append(self.beamspace(x_seq[:, t]))  # (B, feat_dim, R)

        # Stack temporally: (B, T * feat_dim, R)
        stacked = torch.cat(frame_feats, dim=1)

        # Encode with 1D convs across range
        return self.encoder(stacked)
