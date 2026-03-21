"""LISTA beamformer modules for mmDar v2.

Implements physics-informed deep unrolling of ISTA for mmWave angular
super-resolution from raw per-antenna complex IQ data.

Architecture:
    Input:  (B, 8, 512) complex64  — 8 virtual antennas × 512 range bins
    Output: (B, N_az, 512) complex64 — N_az-bin angular spectrum

Modules:
    build_steering_matrix  — IWR1443 uniform λ/2 array steering matrix
    sin_theta_to_bin       — inverse of the sine grid formula (for tests)
    LISTALayer             — one unrolled ISTA proximal-gradient step
    LISTABeamformer        — K stacked LISTA layers with per-element calibration
    FFTBeamformer          — FFT baseline (same I/O shape, fftshift included)

Physics notes:
    - IWR1443: 3TX × 4RX; selecting TX0+TX2 gives 8 virtual azimuth elements
      at λ/2 spacing. Element indices n=0..7; d/λ = 0.5.
    - Steering formula: A[n, k] = exp(j·π·n·sin(θ_k)), where
      sin(θ_k) = -1 + 2k/(N_az - 1) for k=0..N_az-1
    - LISTA update: x_{t+1} = S_λ(x_t - α·A_eff^H·(A_eff·x_t - y))
      where S_λ is complex soft-thresholding.

References:
    - Gregor & LeCun 2010 (LISTA)
    - RCI-DUNet (complex deep unrolling for MIMO radar)
    - TI AN swra554a (TDM-MIMO Doppler phase compensation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.cvnn import complex_soft_threshold


def build_steering_matrix(N_az: int = 256) -> torch.Tensor:
    """Build the IWR1443 uniform λ/2 array steering matrix.

    Element positions: n = 0, 1, ..., 7 (8 virtual azimuth elements)
    d/λ = 0.5, so the phase increment per element is π·sin(θ).

    Sine grid: sin(θ_k) = -1 + 2k/(N_az - 1) for k = 0 .. N_az-1
    This grid spans [-1, +1] uniformly with N_az points.

    Steering matrix: A[n, k] = exp(j·π·n·sin(θ_k))
    Shape: (8, N_az), dtype: complex64

    Note: sin(θ) = 0 (broadside) corresponds to the grid point nearest 0,
    i.e. bin = round((0+1)*(N_az-1)/2). For N_az=256, bin=128.
    Due to grid discretization, this is not exactly 0 — max phase is
    π·7·(2·128/255 - 1) ≈ 0.086 rad, which is near-broadside.

    Args:
        N_az: Number of angular bins (default 256).

    Returns:
        Steering matrix A as torch.Tensor, shape (8, N_az), dtype complex64.
        Registered as a non-trainable buffer in LISTABeamformer.
    """
    n = np.arange(8)  # element indices 0..7
    sin_theta = -1.0 + 2.0 * np.arange(N_az) / (N_az - 1)  # shape (N_az,)
    phase = np.pi * np.outer(n, sin_theta)  # shape (8, N_az)
    A = np.exp(1j * phase).astype(np.complex64)
    return torch.from_numpy(A)


def sin_theta_to_bin(sin_val: float, N_az: int = 256) -> int:
    """Convert a sine-space angle to the nearest angular bin index.

    Inverse of the steering matrix sine grid formula:
        sin(θ_k) = -1 + 2k/(N_az - 1)
    Solving for k:
        k = round((sin_val + 1.0) * (N_az - 1) / 2.0)

    Use this function everywhere bins are needed from known angles —
    never hardcode bin numbers in tests or production code.

    REVIEW FIX #1: formula-derived bin indices, never hardcoded.

    Args:
        sin_val: Sine of the target angle (range [-1.0, +1.0]).
        N_az:    Number of angular bins (must match steering matrix).

    Returns:
        Nearest integer bin index in [0, N_az-1].
    """
    return int(round((sin_val + 1.0) * (N_az - 1) / 2.0))


class LISTALayer(nn.Module):
    """One unrolled proximal-gradient (ISTA) step for complex sparse recovery.

    Implements:
        residual = A_eff · x_k - y           (shape B,8,512)
        grad     = A_eff^H · residual          (shape B,N_az,512)
        x_hat    = x_k - α · grad
        x_{k+1}  = S_λ(x_hat)

    where α = softplus(alpha_raw) > 0 and λ = softplus(lam_raw) > 0.

    REVIEW FIX #4: A_eff_H is passed in already conjugate-transposed.
    REVIEW FIX #5: alpha_raw / lam_raw are raw parameters; positivity enforced
                   by softplus in forward — never store softplus result as parameter.

    No normalization layers (GroupNorm/BatchNorm) — see CONTEXT.md locked decisions.
    """

    def __init__(self) -> None:
        super().__init__()
        # Raw parameters — softplus applied in forward to ensure positivity
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.lam_raw = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        x_k: torch.Tensor,
        y: torch.Tensor,
        A_eff: torch.Tensor,
        A_eff_H: torch.Tensor,
    ) -> torch.Tensor:
        """One LISTA proximal-gradient step.

        Args:
            x_k:    Current estimate, shape (B, N_az, R) complex64
            y:      Measurement, shape (B, 8, R) complex64
            A_eff:  Effective steering matrix, shape (8, N_az) complex64
            A_eff_H: Conjugate-transpose of A_eff, shape (N_az, 8) complex64
                     Must be A_eff.conj().T — NOT just A_eff.T

        Returns:
            x_{k+1}: Updated estimate, shape (B, N_az, R) complex64
        """
        # Step size — always positive
        alpha = F.softplus(self.alpha_raw)

        # Forward model residual: A_eff @ x_k - y
        # A_eff: (8, N_az); x_k: (B, N_az, R) -> residual: (B, 8, R)
        residual = torch.einsum("mn,bnr->bmr", A_eff, x_k) - y

        # Gradient: A_eff^H @ residual
        # A_eff_H: (N_az, 8); residual: (B, 8, R) -> grad: (B, N_az, R)
        grad = torch.einsum("nm,bmr->bnr", A_eff_H, residual)

        # Gradient step
        x_hat = x_k - alpha * grad

        # Proximal operator (complex soft-thresholding)
        x_next = complex_soft_threshold(x_hat, self.lam_raw)
        return x_next


class LISTABeamformer(nn.Module):
    """Physics-informed unrolled LISTA beamformer for mmWave angular SR.

    Maps raw per-antenna IQ measurements (B, 8, 512) complex64 to a
    high-resolution angular spectrum (B, N_az, 512) complex64 using
    K stacked proximal-gradient ISTA steps with learned step sizes and
    thresholds.

    Architecture:
        1. A_eff = g[:, None] * A   (per-element calibration, not diag matmul)
        2. A_eff_H = A_eff.conj().T  (REVIEW FIX #4 — explicit conjugate-transpose)
        3. x_0 = A_eff^H @ y         (matched-filter initialization)
        4. x_k = LISTALayer_k(x_{k-1}, y, A_eff, A_eff_H) for k=1..K
        5. return x_K

    Initialization:
        - A registered as non-trainable buffer from build_steering_matrix()
        - g (calibration) initialized to 1+0j (identity: A_eff = A at init)
        - alpha_raw initialized so softplus(alpha_raw) = 1/||A||^2_F (ISTA step)
          via inverse softplus: raw = log(exp(alpha_init) - 1)
        - lam_raw initialized so softplus(lam_raw) = 0.1

    No GroupNorm/BatchNorm inside this module (CONTEXT.md locked decision).
    """

    def __init__(self, K: int = 5, N_az: int = 256) -> None:
        super().__init__()
        self.K = K
        self.N_az = N_az

        # Steering matrix — fixed, non-trainable
        A = build_steering_matrix(N_az)
        self.register_buffer("A", A)

        # Per-element complex gain/phase calibration vector
        # REVIEW FIX: g is explicitly trainable, per-element diagonal (not dense ΔA)
        self.g = nn.Parameter(torch.ones(8, dtype=torch.complex64))

        # K independent LISTA layers (untied weights across layers)
        self.lista_layers = nn.ModuleList([LISTALayer() for _ in range(K)])

        # ISTA-form initialization: alpha = 1 / ||A||^2_2 (spectral norm)
        # Using matrix_norm(A, ord=2) = largest singular value
        with torch.no_grad():
            sigma_max = torch.linalg.matrix_norm(A, ord=2)
            alpha_init = 1.0 / (sigma_max ** 2).item()

            # Inverse softplus: raw = log(exp(val) - 1)
            # Numerically stable version: for small val, use val directly
            def inv_softplus(val: float) -> float:
                t = torch.tensor(val, dtype=torch.float64)
                return torch.log(torch.exp(t) - 1.0).item()

            raw_alpha = inv_softplus(alpha_init)
            # lam_init=0.01: smaller threshold prevents over-thresholding at init.
            # lam=0.1 was empirically too aggressive — after 2 LISTA steps the signal
            # collapses to zero even for a high-SNR matched-filter input.
            raw_lam = inv_softplus(0.01)

            for layer in self.lista_layers:
                layer.alpha_raw.fill_(raw_alpha)
                layer.lam_raw.fill_(raw_lam)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """LISTA beamforming forward pass.

        Args:
            y: Input measurements, shape (B, 8, R) complex64
               where B=batch, 8=virtual antennas, R=range bins (512)

        Returns:
            x: Angular spectrum, shape (B, N_az, R) complex64
        """
        # Effective steering matrix with per-element calibration
        # g[:, None] broadcasts over columns (N_az dimension)
        # REVIEW FIX: element-wise scaling, NOT diag @ A
        A_eff = self.g.unsqueeze(-1) * self.A  # (8, N_az)

        # Explicit conjugate-transpose — REVIEW FIX #4
        A_eff_H = A_eff.conj().T  # (N_az, 8)

        # Matched-filter initial estimate: x_0 = A_eff^H @ y
        x = torch.einsum("nm,bmr->bnr", A_eff_H, y)  # (B, N_az, R)

        # K unrolled LISTA steps
        for layer in self.lista_layers:
            x = layer(x, y, A_eff, A_eff_H)

        return x


class FFTBeamformer(nn.Module):
    """FFT baseline beamformer with fftshift for correct centering.

    Drop-in replacement for LISTABeamformer with identical I/O shapes.
    Uses DFT along the antenna dimension (dim=1) padded to N_az bins,
    then fftshift to place DC (broadside) at bin N_az//2.

    REVIEW FIX #7: fftshift ensures broadside (all-ones input, sin=0)
    peaks at center bin ~N_az//2, matching the LISTA convention where
    the sine grid is centered at N_az//2.

    Usage: ablation baseline to verify LISTA adds value beyond having
    more angular bins.
    """

    def __init__(self, N_az: int = 256) -> None:
        super().__init__()
        self.N_az = N_az

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """FFT beamforming forward pass.

        Args:
            y: Input measurements, shape (B, 8, R) complex64

        Returns:
            x_fft: Angular spectrum, shape (B, N_az, R) complex64
                   DC (broadside) at bin N_az//2 after fftshift.
        """
        # DFT along antenna dimension (dim=1), zero-padded to N_az
        x_fft = torch.fft.fft(y, n=self.N_az, dim=1)

        # Center DC at N_az//2 (matches LISTA sine grid centering)
        x_fft = torch.fft.fftshift(x_fft, dim=1)

        return x_fft
