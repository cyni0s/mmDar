"""Unit tests for v2/train/loss.py.

Requires pytorch3d (available in Docker). Tests are skipped on host if
pytorch3d is not installed.

Tests:
    1. chamfer_loss returns finite positive scalar for random (B, N, 3) pred vs (B, M, 3) gt
    2. chamfer_loss with confidence weights produces different (lower) loss than without
    3. dcd_loss returns finite positive scalar for random inputs
    4. coverage_loss returns 0.0 when pred == gt (perfect coverage)
    5. coverage_loss returns positive scalar when pred is far from gt
    6. dcd_weight_schedule returns 0.0 for epoch < 5, ramps linearly 0->0.1 for epochs 5-14,
       returns 0.1 for epoch >= 15
    7. composite_loss returns dict with 'total', 'chamfer', 'dcd', 'coverage', 'confidence' keys
    8. composite_loss backward produces finite gradients on pred_pts
"""

import pytest

pytorch3d = pytest.importorskip("pytorch3d", reason="pytorch3d not installed")

import torch

from v2.train.loss import (
    chamfer_loss,
    dcd_loss,
    coverage_loss,
    confidence_loss,
    composite_loss,
    dcd_weight_schedule,
    measurement_consistency_loss,
)


# ---------------------------------------------------------------------------
# Test 1: chamfer_loss finite positive scalar
# ---------------------------------------------------------------------------

def test_chamfer_loss_basic():
    """chamfer_loss returns finite positive scalar for random inputs."""
    B, N, M = 2, 128, 256
    pred = torch.randn(B, N, 3)
    gt = torch.randn(B, M, 3)
    loss = chamfer_loss(pred, gt)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() > 0, "chamfer_loss should be positive"
    assert torch.isfinite(loss), f"Non-finite chamfer_loss: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 2: chamfer_loss with confidence weights changes the loss
# ---------------------------------------------------------------------------

def test_chamfer_loss_with_weights():
    """chamfer_loss with confidence weights produces different result than without."""
    B, N, M = 2, 64, 64
    torch.manual_seed(42)
    pred = torch.randn(B, N, 3)
    gt = torch.randn(B, M, 3)

    loss_unweighted = chamfer_loss(pred, gt, weights_x=None)

    # Strongly down-weight some predictions (weights near 0 for first half)
    weights = torch.ones(B, N)
    weights[:, :N // 2] = 0.01  # near-zero weight for first half
    loss_weighted = chamfer_loss(pred, gt, weights_x=weights)

    assert torch.isfinite(loss_weighted), "Weighted chamfer loss is non-finite"
    # Weighted and unweighted should differ
    assert abs(loss_unweighted.item() - loss_weighted.item()) > 1e-6, \
        f"Weighted and unweighted chamfer losses should differ: " \
        f"unweighted={loss_unweighted.item():.6f}, weighted={loss_weighted.item():.6f}"


# ---------------------------------------------------------------------------
# Test 3: dcd_loss finite positive scalar
# ---------------------------------------------------------------------------

def test_dcd_loss_basic():
    """dcd_loss returns finite positive scalar for random inputs."""
    B, N, M = 2, 64, 64
    pred = torch.randn(B, N, 3)
    gt = torch.randn(B, M, 3)
    loss = dcd_loss(pred, gt)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.item() >= 0, "dcd_loss should be non-negative"
    assert torch.isfinite(loss), f"Non-finite dcd_loss: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 4: coverage_loss is 0 when pred == gt
# ---------------------------------------------------------------------------

def test_coverage_loss_perfect():
    """coverage_loss returns 0.0 when pred exactly equals gt."""
    B, N = 2, 64
    pts = torch.randn(B, N, 3)
    loss = coverage_loss(pts, pts, threshold=0.25)
    assert loss.item() < 1e-6, f"coverage_loss should be ~0 for pred==gt, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 5: coverage_loss positive when pred is far from gt
# ---------------------------------------------------------------------------

def test_coverage_loss_far():
    """coverage_loss returns positive scalar when pred is far from gt."""
    B, N, M = 2, 64, 64
    # Pred points far from gt (offset by 10m >> threshold=0.25m)
    pred = torch.randn(B, N, 3)
    gt = pred + 10.0  # Guaranteed miss
    loss = coverage_loss(pred, gt, threshold=0.25)
    assert loss.item() > 0, f"coverage_loss should be positive when pred is far, got {loss.item()}"
    assert torch.isfinite(loss), f"Non-finite coverage_loss: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 6: dcd_weight_schedule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("epoch,expected_zero", [
    (0, True),
    (4, True),
])
def test_dcd_weight_schedule_warmup(epoch, expected_zero):
    """dcd_weight_schedule returns 0.0 for epoch < 5."""
    w = dcd_weight_schedule(epoch)
    assert w == 0.0, f"Expected 0.0 at epoch {epoch}, got {w}"


@pytest.mark.parametrize("epoch,lo,hi", [
    (5,  0.0,  0.0),   # start of ramp: exactly 0.0
    (10, 0.04, 0.06),  # midpoint: ~0.05
    (14, 0.09, 0.10),  # end of ramp: ~0.09..0.1
])
def test_dcd_weight_schedule_ramp(epoch, lo, hi):
    """dcd_weight_schedule ramps linearly from 0->0.1 for epochs 5-14."""
    w = dcd_weight_schedule(epoch)
    assert lo <= w <= hi, f"Expected w in [{lo},{hi}] at epoch {epoch}, got {w}"


def test_dcd_weight_schedule_plateau():
    """dcd_weight_schedule returns 0.1 for epoch >= 15."""
    for epoch in [15, 20, 50, 100]:
        w = dcd_weight_schedule(epoch)
        assert abs(w - 0.1) < 1e-6, f"Expected 0.1 at epoch {epoch}, got {w}"


# ---------------------------------------------------------------------------
# Test 7: composite_loss returns correct keys
# ---------------------------------------------------------------------------

def test_composite_loss_keys():
    """composite_loss returns dict with all required keys."""
    B, N, M = 2, 64, 64
    pred = torch.randn(B, N, 3, requires_grad=True)
    gt = torch.randn(B, M, 3)
    conf = torch.randn(B, N, 1)
    losses = composite_loss(pred, gt, conf, epoch=10)
    required_keys = {"total", "chamfer", "dcd", "coverage", "confidence", "measurement_consistency"}
    assert required_keys.issubset(set(losses.keys())), \
        f"Missing keys: {required_keys - set(losses.keys())}"
    for key in required_keys:
        assert torch.isfinite(losses[key]), f"Non-finite loss['{key}']: {losses[key].item()}"


# ---------------------------------------------------------------------------
# Test 8: composite_loss backward produces finite gradients
# ---------------------------------------------------------------------------

def test_composite_loss_backward():
    """composite_loss backward produces finite gradients on pred_pts."""
    B, N, M = 2, 64, 64
    pred = torch.randn(B, N, 3, requires_grad=True)
    gt = torch.randn(B, M, 3)
    conf = torch.randn(B, N, 1)
    losses = composite_loss(pred, gt, conf, epoch=10)
    losses["total"].backward()
    assert pred.grad is not None, "No gradient on pred_pts"
    assert torch.isfinite(pred.grad).all(), \
        f"Non-finite gradient on pred_pts: {pred.grad.abs().max().item()}"
