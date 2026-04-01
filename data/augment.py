"""Physically valid data augmentation for FMCW radar IQ.

Three augmentations (physics-validated):

1. Horizontal flip: reverse antenna order → mirror scene in azimuth
   - Exact for symmetric ULA at λ/2 spacing
   - Also flips lidar GT y-coordinate

2. Complex Gaussian noise: simulate lower SNR
   - σ scaled from per-sample RMS, SNR drawn from Uniform(15, 25) dB
   - Applied to raw IQ before any processing

3. Temporal frame masking: zero out random past frames
   - Never masks the current (last) frame
   - Drops 2-6 random past frames
   - Forces robustness to missing temporal context

Usage:
    from data.augment import augment_sample
    radar_aug, lidar_aug, proto_aug = augment_sample(radar, lidar, protos)
"""

import torch
import math


def horizontal_flip(radar: torch.Tensor, lidar: torch.Tensor,
                    protos: torch.Tensor) -> tuple:
    """Mirror the scene in azimuth by reversing antenna order.

    For a ULA with λ/2 spacing, reversing antenna indices 0↔7, 1↔6, etc.
    is equivalent to negating the azimuth angle. No phase correction needed.

    Args:
        radar: (T, 8, R) complex — multi-frame raw IQ
        lidar: (8192, 3) float — GT point cloud (x, y, z)
        protos: (K, 2) float — GT prototype centers (x, y)

    Returns:
        Flipped (radar, lidar, protos)
    """
    radar_flip = radar.flip(dims=[1])          # reverse antenna dim
    lidar_flip = lidar.clone()
    lidar_flip[:, 1] = -lidar_flip[:, 1]      # negate y (azimuth)
    protos_flip = protos.clone()
    protos_flip[:, 1] = -protos_flip[:, 1]    # negate y
    return radar_flip, lidar_flip, protos_flip


def add_complex_noise(radar: torch.Tensor, snr_db_min: float = 15.0,
                      snr_db_max: float = 25.0) -> torch.Tensor:
    """Add complex Gaussian noise at a random SNR level.

    σ is computed from per-sample RMS magnitude:
        A_rms = sqrt(mean(|x|²))
        σ = A_rms / sqrt(2 × 10^(SNR_dB/10))

    Args:
        radar: (T, 8, R) complex
        snr_db_min: minimum SNR in dB
        snr_db_max: maximum SNR in dB

    Returns:
        Noisy radar (same shape)
    """
    snr_db = torch.empty(1).uniform_(snr_db_min, snr_db_max).item()
    a_rms = (radar.abs() ** 2).mean().sqrt()
    sigma = a_rms / math.sqrt(2 * 10 ** (snr_db / 10))

    noise = sigma * torch.complex(
        torch.randn_like(radar.real),
        torch.randn_like(radar.imag),
    )
    return radar + noise


def temporal_mask(radar: torch.Tensor, n_drop_min: int = 2,
                  n_drop_max: int = 6) -> torch.Tensor:
    """Zero out random past frames (never the current/last frame).

    Args:
        radar: (T, 8, R) complex
        n_drop_min: minimum frames to drop
        n_drop_max: maximum frames to drop

    Returns:
        Masked radar (same shape)
    """
    T = radar.shape[0]
    if T <= 2:
        return radar

    max_droppable = min(n_drop_max, T - 1)
    if max_droppable < n_drop_min:
        return radar

    n_drop = torch.randint(n_drop_min, max_droppable + 1, (1,)).item()
    # Select random past frames to drop (not the last frame)
    drop_idx = torch.randperm(T - 1)[:n_drop]

    radar_masked = radar.clone()
    radar_masked[drop_idx] = 0
    return radar_masked


def augment_sample(
    radar: torch.Tensor,
    lidar: torch.Tensor,
    protos: torch.Tensor,
    p_flip: float = 0.5,
    p_noise: float = 0.5,
    p_mask: float = 0.3,
) -> tuple:
    """Apply random augmentations to a single training sample.

    Args:
        radar: (T, 8, R) complex
        lidar: (8192, 3) float
        protos: (K, 2) float
        p_flip: probability of horizontal flip
        p_noise: probability of noise addition
        p_mask: probability of temporal masking

    Returns:
        Augmented (radar, lidar, protos)
    """
    if torch.rand(1).item() < p_flip:
        radar, lidar, protos = horizontal_flip(radar, lidar, protos)

    if torch.rand(1).item() < p_noise:
        radar = add_complex_noise(radar)

    if torch.rand(1).item() < p_mask:
        radar = temporal_mask(radar)

    return radar, lidar, protos
