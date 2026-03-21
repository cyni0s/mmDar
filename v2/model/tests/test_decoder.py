"""Unit tests for v2/model/decoder.py.

All tests run on CPU with synthetic random data — no Docker, no GPU needed.

Tests:
    1. PointCloudDecoder forward produces (B, 8192, 3) float32 points and
       (B, 8192, 1) float32 confidence for input (B, 128, 512) float32
    2. Template grid is (1024, 3) float32, registered as buffer (moves with
       .cuda() call but accessible on CPU), z=0 everywhere
    3. Intermediate densification produces correct counts: 1024->2048->4096->8192
    4. sample_features_from_range_map returns (B, N, C) for (B, C, 512) feature
       map and (B, N, 3) points
    5. Gradient flows from output points back through to feature map input
       (backward produces finite grads)
    6. build_polar_template produces points within expected range bounds
       (r_max=10.8, az_range=140 deg)
"""

import math
import torch
import pytest

from v2.model.decoder import (
    PointCloudDecoder,
    build_polar_template,
    sample_features_from_range_map,
)


# ---------------------------------------------------------------------------
# Test 1: forward output shapes
# ---------------------------------------------------------------------------

def test_forward_output_shapes():
    """Decoder maps (B, 128, 512) -> (B, 8192, 3) pts + (B, 8192, 1) conf."""
    B = 2
    model = PointCloudDecoder(feature_ch=128, global_dim=1024, r_max=10.8)
    model.eval()
    feat = torch.randn(B, 128, 512)
    with torch.no_grad():
        pts, conf = model(feat)
    assert pts.shape == (B, 8192, 3), f"Expected (B, 8192, 3), got {pts.shape}"
    assert conf.shape == (B, 8192, 1), f"Expected (B, 8192, 1), got {conf.shape}"
    assert pts.dtype == torch.float32, f"Expected float32, got {pts.dtype}"
    assert conf.dtype == torch.float32, f"Expected float32, got {conf.dtype}"


# ---------------------------------------------------------------------------
# Test 2: template buffer properties
# ---------------------------------------------------------------------------

def test_template_buffer():
    """Template is (1024, 3) float32 buffer with z=0 everywhere."""
    model = PointCloudDecoder(feature_ch=128, global_dim=1024, r_max=10.8)
    # Check it's registered as a buffer (not a parameter)
    assert "template" in dict(model.named_buffers()), \
        "template must be a registered buffer"
    tmpl = model.template
    assert tmpl.shape == (1024, 3), f"Expected (1024, 3), got {tmpl.shape}"
    assert tmpl.dtype == torch.float32
    # z-column (index 2) must be exactly 0
    assert tmpl[:, 2].abs().max() == 0.0, "Template z-column must be zero"


# ---------------------------------------------------------------------------
# Test 3: densification step counts
# ---------------------------------------------------------------------------

def test_densification_counts():
    """Densification doubles point count: 1024->2048->4096->8192."""
    B = 1
    model = PointCloudDecoder(feature_ch=128, global_dim=1024, r_max=10.8)
    model.eval()

    feat = torch.randn(B, 128, 512)
    with torch.no_grad():
        # Run forward but capture intermediate sizes via hooks
        counts = []

        original_forward = model.forward

        def patched_forward(feature_map):
            # Replicate forward step by step to capture intermediate counts
            B_ = feature_map.shape[0]
            global_desc = model.global_encoder(feature_map)  # (B, global_dim)
            pts = model.template.unsqueeze(0).expand(B_, -1, -1)  # (B, 1024, 3)
            counts.append(pts.shape[1])
            for stage in model.stages:
                local_feats = sample_features_from_range_map(
                    feature_map, pts, r_max=model.r_max
                )
                pts, _ = stage(pts, global_desc, local_feats)
                counts.append(pts.shape[1])
            return model.forward(feature_map)  # avoid re-using patched

        # Just call forward normally and check final output
        pts, conf = model(feat)

    # Check final shape proves 3 doublings: 1024->2048->4096->8192
    assert pts.shape[1] == 8192

    # Also test the progression by running internal stages manually
    model.eval()
    feat = torch.randn(B, 128, 512)
    with torch.no_grad():
        # Must apply global max pool after encoder (matches forward())
        enc_out = model.global_encoder(feat)         # (B, global_dim, 512)
        global_desc = enc_out.max(dim=-1).values     # (B, global_dim)
        pts_cur = model.template.unsqueeze(0).expand(B, -1, -1).clone()
        assert pts_cur.shape[1] == 1024

        for i, stage in enumerate(model.stages):
            local_feats = sample_features_from_range_map(feat, pts_cur, r_max=model.r_max)
            pts_cur, _ = stage(pts_cur, global_desc, local_feats)
            expected = 1024 * (2 ** (i + 1))
            assert pts_cur.shape[1] == expected, \
                f"Stage {i+1}: expected {expected} points, got {pts_cur.shape[1]}"


# ---------------------------------------------------------------------------
# Test 4: sample_features_from_range_map output shape
# ---------------------------------------------------------------------------

def test_sample_features_shape():
    """sample_features_from_range_map returns (B, N, C) correctly."""
    B, N, C = 3, 128, 64
    feature_map = torch.randn(B, C, 512)
    # Create points within r_max=10.8
    pts = torch.randn(B, N, 3) * 5.0  # moderate range
    pts[..., 2] = 0.0  # z=0
    out = sample_features_from_range_map(feature_map, pts, r_max=10.8)
    assert out.shape == (B, N, C), f"Expected ({B}, {N}, {C}), got {out.shape}"
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 5: gradients flow through decoder
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Backward produces finite gradients from output back to feature_map input."""
    B = 1
    model = PointCloudDecoder(feature_ch=128, global_dim=1024, r_max=10.8)
    model.train()
    feat = torch.randn(B, 128, 512, requires_grad=True)
    pts, conf = model(feat)
    loss = pts.sum() + conf.sum()
    loss.backward()
    assert feat.grad is not None, "No gradient on feature_map"
    assert torch.isfinite(feat.grad).all(), "Non-finite gradient on feature_map"


# ---------------------------------------------------------------------------
# Test 6: build_polar_template range bounds
# ---------------------------------------------------------------------------

def test_polar_template_bounds():
    """build_polar_template points lie within expected range and azimuth."""
    r_max = 10.8
    az_deg = 140.0  # +/- 70 degrees
    tmpl = build_polar_template(N_r=32, N_az=32, r_max=r_max, az_range_deg=az_deg)

    assert tmpl.shape == (1024, 3), f"Expected (1024, 3), got {tmpl.shape}"
    assert tmpl.dtype == torch.float32

    # Compute range for each point: r = sqrt(x^2 + y^2)
    r_vals = tmpl[:, :2].norm(dim=1)
    assert r_vals.max() <= r_max + 1e-4, f"Max range {r_vals.max():.4f} exceeds r_max={r_max}"
    assert r_vals.min() >= 0.0, "Negative range found"

    # z should be zero
    assert tmpl[:, 2].abs().max() < 1e-6, "z values should be zero"

    # Check azimuth bounds: max |az| should be within 70 degrees
    az_rad = math.radians(az_deg / 2)
    y_from_x = torch.atan2(tmpl[:, 1], tmpl[:, 0])
    assert y_from_x.abs().max() <= az_rad + 1e-4, \
        f"Max azimuth {y_from_x.abs().max():.4f} rad exceeds {az_rad:.4f} rad"
