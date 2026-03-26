"""Unit tests for v2/model/decoder_2d.py — 2D angular-topology-preserving decoder.

The CRITICAL test: points at same range but different azimuths must get DIFFERENT
features from the 2D feature map. This is the exact bug that decoder.py has —
its feature map is (B, C, 1, R) so azimuth interpolation is a no-op.

All tests run on CPU with synthetic random data.
"""

import math
import torch
import pytest


# ---------------------------------------------------------------------------
# Test 1: THE CRITICAL TEST — different azimuths get different features
# ---------------------------------------------------------------------------

def test_sample_features_2d_different_azimuths():
    """Points at same range but different azimuths must get DIFFERENT features.

    This is the exact angular collapse bug. The 1D decoder produces identical
    features for all azimuths at the same range because the feature map has
    height=1 in the azimuth dimension.
    """
    from v2.model.decoder_2d import sample_features_2d

    B, C, H, W = 1, 16, 256, 512
    feat = torch.randn(B, C, H, W)

    # Two points at same range (5m), different azimuths (0 deg and 45 deg)
    r = 5.0
    pts = torch.tensor([[[r, 0.0, 0.0],  # broadside (az=0 deg)
                          [r * math.cos(math.radians(45)),
                           r * math.sin(math.radians(45)),
                           0.0]]])        # az=45 deg

    feats = sample_features_2d(feat, pts)  # (1, 2, C)
    f0 = feats[0, 0]  # features at az=0 deg
    f1 = feats[0, 1]  # features at az=45 deg

    assert not torch.allclose(f0, f1, atol=1e-3), \
        f"Features at different azimuths should DIFFER! Max diff: {(f0 - f1).abs().max()}"


# ---------------------------------------------------------------------------
# Test 2: different ranges get different features
# ---------------------------------------------------------------------------

def test_sample_features_2d_different_ranges():
    """Points at different ranges must get different features."""
    from v2.model.decoder_2d import sample_features_2d

    feat = torch.randn(1, 16, 256, 512)
    pts = torch.tensor([[[3.0, 0.0, 0.0],
                          [7.0, 0.0, 0.0]]])

    feats = sample_features_2d(feat, pts)
    assert not torch.allclose(feats[0, 0], feats[0, 1], atol=1e-3)


# ---------------------------------------------------------------------------
# Test 3: output shape
# ---------------------------------------------------------------------------

def test_sample_features_2d_output_shape():
    """sample_features_2d returns (B, N, C) for (B, C, H, W) input."""
    from v2.model.decoder_2d import sample_features_2d

    B, C, H, W = 2, 32, 256, 512
    feat = torch.randn(B, C, H, W)
    pts = torch.randn(B, 100, 3) * 5.0
    pts[..., 2] = 0.0

    out = sample_features_2d(feat, pts)
    assert out.shape == (B, 100, C), f"Expected (2, 100, 32), got {out.shape}"
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 4: PointCloudDecoder2D output shape
# ---------------------------------------------------------------------------

def test_decoder_2d_output_shape():
    """Decoder2D maps (B, 128, 256, 512) -> (B, 8192, 3) pts + (B, 8192, 1) conf."""
    from v2.model.decoder_2d import PointCloudDecoder2D

    dec = PointCloudDecoder2D(feature_ch=128, global_dim=1024)
    dec.eval()
    feat = torch.randn(2, 128, 256, 512)

    with torch.no_grad():
        pts, conf = dec(feat)

    assert pts.shape == (2, 8192, 3), f"Expected (2, 8192, 3), got {pts.shape}"
    assert conf.shape == (2, 8192, 1), f"Expected (2, 8192, 1), got {conf.shape}"


# ---------------------------------------------------------------------------
# Test 5: MagnitudeBaseline2D end-to-end
# ---------------------------------------------------------------------------

def test_magnitude_baseline_2d_end_to_end():
    """MagnitudeBaseline2D: (B, 8, 512) complex -> (B, 8192, 3) + (B, 8192, 1)."""
    from v2.model import MagnitudeBaseline2D

    model = MagnitudeBaseline2D(N_az=256, bridge_out_ch=128)
    model.eval()
    x = torch.randn(2, 8, 512, dtype=torch.complex64)

    with torch.no_grad():
        pts, conf = model(x)

    assert pts.shape == (2, 8192, 3)
    assert conf.shape == (2, 8192, 1)


# ---------------------------------------------------------------------------
# Test 6: MagnitudePhaseFusion2D end-to-end
# ---------------------------------------------------------------------------

def test_mag_phase_fusion_2d_end_to_end():
    """MagnitudePhaseFusion2D: (B, 8, 512) complex -> (B, 8192, 3) + (B, 8192, 1)."""
    from v2.model import MagnitudePhaseFusion2D

    model = MagnitudePhaseFusion2D(N_az=256, bridge_out_ch=128)
    model.eval()
    x = torch.randn(2, 8, 512, dtype=torch.complex64)

    with torch.no_grad():
        pts, conf = model(x)

    assert pts.shape == (2, 8192, 3)
    assert conf.shape == (2, 8192, 1)


# ---------------------------------------------------------------------------
# Test 7: gradient flow through decoder_2d
# ---------------------------------------------------------------------------

def test_gradient_flow_decoder_2d():
    """Backward produces finite gradients from output back to 2D feature_map."""
    from v2.model.decoder_2d import PointCloudDecoder2D

    dec = PointCloudDecoder2D(feature_ch=128, global_dim=1024)
    dec.train()
    feat = torch.randn(1, 128, 256, 512, requires_grad=True)

    pts, conf = dec(feat)
    loss = pts.sum() + conf.sum()
    loss.backward()

    assert feat.grad is not None, "No gradient on feature_map"
    assert torch.isfinite(feat.grad).all(), "Non-finite gradient on feature_map"
