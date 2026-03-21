"""
Shared fixtures for v2/data preprocessing tests.

All fixtures use synthetic data — no zip files, no Docker, no real trajectories.
"""

import numpy as np
import pytest

RNG_SEED = 42


@pytest.fixture
def synthetic_frame():
    """
    A single raw TDM-MIMO radar frame with random complex128 data.

    Shape: (192, 4, 512) complex128 — matches raw pkl format.
    Seeded for reproducibility.
    """
    rng = np.random.default_rng(RNG_SEED)
    real_part = rng.standard_normal((192, 4, 512))
    imag_part = rng.standard_normal((192, 4, 512))
    return (real_part + 1j * imag_part).astype(np.complex128)


@pytest.fixture
def synthetic_lidar_large():
    """
    A large lidar point cloud with >8192 points, all within the scene volume.

    Shape: (15000, 3) float64.
    Scene volume: x in [0,10], y in [-10,10], z in [-0.3,0.3].
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    pts = rng.random((15000, 3))
    # Scale to scene volume
    pts[:, 0] = pts[:, 0] * 10.0          # x in [0, 10]
    pts[:, 1] = pts[:, 1] * 20.0 - 10.0  # y in [-10, 10]
    pts[:, 2] = pts[:, 2] * 0.6 - 0.3    # z in [-0.3, 0.3]
    return pts.astype(np.float64)


@pytest.fixture
def synthetic_lidar_small():
    """
    A small lidar point cloud with <8192 points, all within the scene volume.

    Shape: (5000, 3) float64.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    pts = rng.random((5000, 3))
    pts[:, 0] = pts[:, 0] * 10.0
    pts[:, 1] = pts[:, 1] * 20.0 - 10.0
    pts[:, 2] = pts[:, 2] * 0.6 - 0.3
    return pts.astype(np.float64)
