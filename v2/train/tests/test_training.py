"""Integration tests for v2 full model assembly and training script.

Tests:
    1. RadarPointCloudModel forward pass: (B, 8, 512) complex64 -> (B, 8192, 3) + (B, 8192, 1)
    2. RadarPointCloudModel backward: loss.backward() produces finite gradients on all parameters
    3. set_stage1_frozen(model, True) makes beamformer params require_grad=False,
       bridge + decoder still True
    4. set_stage1_frozen(model, False) restores beamformer params require_grad=True
    5. MagnitudeBaseline forward pass: same I/O shapes as RadarPointCloudModel

All tests run on CPU with tiny synthetic data (B=2). No real data required.
pytorch3d must be importable for tests that trigger the loss; marked with
pytest.importorskip where applicable.
"""

import pytest
import torch

from v2.model import MagnitudeBaseline, RadarPointCloudModel, set_stage1_frozen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_cpu() -> RadarPointCloudModel:
    """Small RadarPointCloudModel on CPU (K=2 to keep tests fast)."""
    return RadarPointCloudModel(K=2, N_az=64, bridge_out_ch=32).eval()


@pytest.fixture(scope="module")
def radar_input() -> torch.Tensor:
    """Synthetic radar input (B=2, 8 antennas, 512 range bins), complex64."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 512, dtype=torch.complex64)


# ---------------------------------------------------------------------------
# Test 1: Forward pass shape
# ---------------------------------------------------------------------------

def test_forward_pass_shapes(model_cpu, radar_input):
    """RadarPointCloudModel forward: (B, 8, 512) complex64 -> (B, 8192, 3) + (B, 8192, 1)."""
    with torch.no_grad():
        pts, conf = model_cpu(radar_input)

    B = radar_input.shape[0]  # 2

    assert pts.shape == (B, 8192, 3), (
        f"Expected pts shape ({B}, 8192, 3), got {pts.shape}"
    )
    assert conf.shape == (B, 8192, 1), (
        f"Expected conf shape ({B}, 8192, 1), got {conf.shape}"
    )
    assert pts.dtype == torch.float32, f"pts dtype must be float32, got {pts.dtype}"
    assert conf.dtype == torch.float32, f"conf dtype must be float32, got {conf.dtype}"


# ---------------------------------------------------------------------------
# Test 2: Backward pass gradient flow
# ---------------------------------------------------------------------------

def test_backward_finite_gradients(radar_input):
    """loss.backward() produces finite (non-NaN, non-Inf) gradients on all parameters."""
    pytorch3d = pytest.importorskip(
        "pytorch3d", reason="pytorch3d not installed — skipping backward test"
    )

    model = RadarPointCloudModel(K=2, N_az=64, bridge_out_ch=32)
    model.train()

    pts, conf = model(radar_input)

    # Synthetic GT point cloud (same shape as decoder output)
    gt = torch.randn(radar_input.shape[0], 8192, 3)

    # Use a simple MSE loss on pts to avoid needing pytorch3d chamfer in this test
    # (chamfer_loss is tested separately in test_loss.py)
    loss = (pts - gt).pow(2).mean() + conf.sigmoid().mean()
    loss.backward()

    nan_params = []
    inf_params = []
    none_grad_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            none_grad_params.append(name)
            continue
        if torch.isnan(param.grad).any():
            nan_params.append(name)
        if torch.isinf(param.grad).any():
            inf_params.append(name)

    assert not nan_params, f"NaN gradients found in: {nan_params}"
    assert not inf_params, f"Inf gradients found in: {inf_params}"
    assert not none_grad_params, (
        f"None gradients (no grad path) found in: {none_grad_params}"
    )


# ---------------------------------------------------------------------------
# Test 3: Freeze Stage 1
# ---------------------------------------------------------------------------

def test_set_stage1_frozen_true(radar_input):
    """set_stage1_frozen(model, True) makes beamformer params require_grad=False,
    but bridge and decoder params remain require_grad=True."""
    model = RadarPointCloudModel(K=2, N_az=64, bridge_out_ch=32)

    set_stage1_frozen(model, frozen=True)

    # All beamformer params should be frozen
    beamformer_params = list(model.beamformer.parameters())
    assert len(beamformer_params) > 0, "Beamformer has no parameters"
    for p in beamformer_params:
        assert not p.requires_grad, (
            "Expected beamformer parameter requires_grad=False after freeze"
        )

    # Bridge params should remain trainable
    bridge_params = list(model.bridge.parameters())
    assert len(bridge_params) > 0, "Bridge has no parameters"
    for p in bridge_params:
        assert p.requires_grad, (
            "Expected bridge parameter requires_grad=True after freezing only Stage 1"
        )

    # Decoder params should remain trainable
    decoder_params = list(model.decoder.parameters())
    assert len(decoder_params) > 0, "Decoder has no parameters"
    for p in decoder_params:
        assert p.requires_grad, (
            "Expected decoder parameter requires_grad=True after freezing only Stage 1"
        )


# ---------------------------------------------------------------------------
# Test 4: Unfreeze Stage 1
# ---------------------------------------------------------------------------

def test_set_stage1_frozen_false(radar_input):
    """set_stage1_frozen(model, False) restores beamformer params to require_grad=True."""
    model = RadarPointCloudModel(K=2, N_az=64, bridge_out_ch=32)

    # First freeze, then unfreeze
    set_stage1_frozen(model, frozen=True)
    set_stage1_frozen(model, frozen=False)

    # After unfreeze, all params should be trainable again
    beamformer_params = list(model.beamformer.parameters())
    assert len(beamformer_params) > 0, "Beamformer has no parameters"
    for p in beamformer_params:
        assert p.requires_grad, (
            "Expected beamformer parameter requires_grad=True after unfreeze"
        )


# ---------------------------------------------------------------------------
# Test 5: MagnitudeBaseline forward pass
# ---------------------------------------------------------------------------

def test_magnitude_baseline_forward_shapes(radar_input):
    """MagnitudeBaseline forward pass produces same output shapes as RadarPointCloudModel."""
    model = MagnitudeBaseline(N_az=64, bridge_out_ch=32).eval()

    with torch.no_grad():
        pts, conf = model(radar_input)

    B = radar_input.shape[0]  # 2

    assert pts.shape == (B, 8192, 3), (
        f"Expected pts shape ({B}, 8192, 3), got {pts.shape}"
    )
    assert conf.shape == (B, 8192, 1), (
        f"Expected conf shape ({B}, 8192, 1), got {conf.shape}"
    )
    assert pts.dtype == torch.float32, f"pts dtype must be float32, got {pts.dtype}"
    assert conf.dtype == torch.float32, f"conf dtype must be float32, got {conf.dtype}"
