import torch
import pytest


def test_unet_occ_forward_shape():
    """U-Net output matches input spatial dims with 1 output channel."""
    from v2.model.unet_occupancy import UNetOcc
    model = UNetOcc(n_channels=41, n_classes=1)
    x = torch.randn(2, 41, 256, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 512), f"Expected (2, 1, 256, 512), got {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output should be sigmoid-bounded"


def test_unet_occ_param_count():
    """U-Net should have ~17M params (4-level symmetric, 512-channel bottleneck)."""
    from v2.model.unet_occupancy import UNetOcc
    model = UNetOcc(n_channels=41, n_classes=1)
    n_params = sum(p.numel() for p in model.parameters())
    assert 15_000_000 < n_params < 20_000_000, f"Expected 15-20M params, got {n_params:,}"
