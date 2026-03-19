"""Shared pytest fixtures for mmDar unit tests."""
import pytest
import torch


@pytest.fixture
def dummy_radar_single():
    """Single-frame radar input: (B=2, C=1, H=256, W=64)."""
    return torch.randn(2, 1, 256, 64)


@pytest.fixture
def dummy_radar_seq():
    """5-frame radar sequence: (B=2, T=5, C=1, H=256, W=64)."""
    return torch.randn(2, 5, 1, 256, 64)


@pytest.fixture
def dummy_lidar_seq():
    """5-frame lidar sequence: (B=2, T=5, C=1, H=256, W=512)."""
    return torch.randn(2, 5, 1, 256, 512)


@pytest.fixture
def device():
    """Return the best available compute device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
