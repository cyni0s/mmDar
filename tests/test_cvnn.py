"""Tests for v2/model/cvnn.py — complex-valued neural network primitives.

Covers:
- complex_soft_threshold: magnitude shrink, phase preservation, zero-input safety
- complex_soft_threshold_gradient_near_zero: finite gradients (eps-safe modulus)
- test_soft_threshold_gradcheck: Wirtinger gradcheck using cdouble (complex128)
- ComplexConv1d: shape, Wirtinger gradcheck
- ComplexGroupNorm: shape, phase-rotated finite output/gradient [REVIEW FIX #8]
- rayleigh_init_: deterministic statistical check [REVIEW FIX #9]
- ComplexLinear: shape
- safe_modulus: gradient at zero [REVIEW FIX #6], non-negativity

REVIEW FIX #6: eps-safe modulus ensures finite gradients at near-zero inputs.
All gradcheck tests use cdouble (float64 complex), not complex64, as required by
torch.autograd.gradcheck for numerical precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from v2.model.cvnn import (
    complex_soft_threshold,
    ComplexConv1d,
    ComplexGroupNorm,
    ComplexLinear,
    safe_modulus,
)


def test_complex_soft_threshold():
    """Magnitude shrinks by softplus(lam), phase is preserved."""
    x = torch.tensor([3.0 + 4.0j, 1.0 + 0.0j, 0.0 + 0.5j], dtype=torch.complex64)
    lam_raw = torch.tensor(0.0)  # softplus(0) = ln(2) ~ 0.693
    out = complex_soft_threshold(x, lam_raw)
    effective_lam = F.softplus(lam_raw).item()  # ~0.693

    assert out.shape == (3,)

    # |3+4j| = 5; shrunk to 5 - 0.693 = 4.307; phase preserved
    expected_mag = 5.0 - effective_lam
    assert abs(torch.abs(out[0]).item() - expected_mag) < 0.05, (
        f"magnitude {torch.abs(out[0]).item():.4f}, expected ~{expected_mag:.4f}"
    )

    # Phase of 3+4j should be preserved
    assert abs(torch.angle(out[0]).item() - torch.angle(x[0]).item()) < 1e-4, (
        f"phase {torch.angle(out[0]).item():.6f}, expected {torch.angle(x[0]).item():.6f}"
    )


def test_complex_soft_threshold_zero():
    """Zero-magnitude input returns zero with no NaN."""
    x = torch.tensor([0.0 + 0.0j], dtype=torch.complex64)
    lam_raw = torch.tensor(0.0)
    out = complex_soft_threshold(x, lam_raw)
    assert not torch.isnan(out).any(), "NaN in output for zero input"
    assert torch.abs(out[0]).item() < 1e-4, (
        f"|out| = {torch.abs(out[0]).item():.2e}, expected ~0"
    )


def test_complex_soft_threshold_gradient_near_zero():
    """[REVIEW FIX #6] Gradient is finite for near-zero magnitude inputs."""
    x = torch.tensor([1e-10 + 1e-10j], dtype=torch.complex128, requires_grad=True)
    lam_raw = torch.tensor(-5.0, dtype=torch.float64, requires_grad=True)
    # softplus(-5) ~ 0.007, so threshold < |x| — output near zero but nonzero

    out = complex_soft_threshold(x, lam_raw)
    loss = out.abs().sum()
    loss.backward()

    assert torch.isfinite(x.grad).all(), (
        f"x.grad has non-finite values: {x.grad}"
    )
    assert torch.isfinite(lam_raw.grad).all(), (
        f"lam.grad has non-finite values: {lam_raw.grad}"
    )


def test_soft_threshold_gradcheck():
    """Wirtinger gradcheck on complex soft-threshold using cdouble (complex128).

    gradcheck requires float64 precision. The test input is shifted away from
    zero to avoid stressing the eps boundary in the modulus computation.

    REVIEW: Must use cdouble, NOT complex64 — gradcheck needs double precision.
    """
    torch.manual_seed(42)
    real_part = torch.randn(3, 16, 8, dtype=torch.float64)
    imag_part = torch.randn(3, 16, 8, dtype=torch.float64)
    x_base = torch.complex(real_part, imag_part)
    # Shift away from near-zero magnitude
    x_base = x_base + torch.complex(
        torch.full_like(real_part, 0.5),
        torch.full_like(imag_part, 0.5),
    )
    x = x_base.clone().detach().requires_grad_(True)

    lam_raw = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    result = torch.autograd.gradcheck(
        complex_soft_threshold,
        (x, lam_raw),
        eps=1e-4,
        atol=1e-3,
        rtol=1e-3,
    )
    assert result, "gradcheck failed on complex_soft_threshold"


# ---------------------------------------------------------------------------
# Task 2 (Plan 02-02): ComplexConv1d, ComplexGroupNorm, rayleigh_init_,
#                      ComplexLinear, safe_modulus tests
# ---------------------------------------------------------------------------


def test_complex_conv1d_shape():
    """ComplexConv1d(256,128,3,padding=1) maps (2,256,512) complex64 to (2,128,512) complex64."""
    conv = ComplexConv1d(256, 128, kernel_size=3, padding=1)
    x = torch.randn(2, 256, 512, dtype=torch.complex64)
    out = conv(x)
    assert out.shape == (2, 128, 512), f"shape {out.shape}, expected (2,128,512)"
    assert out.dtype == torch.complex64, f"dtype {out.dtype}, expected complex64"


def test_complex_conv1d_gradcheck():
    """Wirtinger gradcheck on ComplexConv1d with cdouble inputs."""
    conv = ComplexConv1d(4, 3, kernel_size=3, padding=1).double()
    x = torch.complex(
        torch.randn(1, 4, 8, dtype=torch.float64),
        torch.randn(1, 4, 8, dtype=torch.float64),
    ).requires_grad_(True)
    result = torch.autograd.gradcheck(conv, (x,), eps=1e-4, atol=1e-3, rtol=1e-3)
    assert result, "gradcheck failed on ComplexConv1d"


def test_complex_group_norm_shape():
    """ComplexGroupNorm(16,128) preserves (4,128,64) shape and dtype."""
    cgn = ComplexGroupNorm(16, 128)
    x = torch.randn(4, 128, 64, dtype=torch.complex64)
    out = cgn(x)
    assert out.shape == (4, 128, 64), f"shape {out.shape}, expected (4,128,64)"
    assert out.dtype == torch.complex64, f"dtype {out.dtype}, expected complex64"


def test_complex_group_norm_phase_rotated():
    """[REVIEW FIX #8] ComplexGroupNorm produces finite outputs/gradients on phase-rotated inputs."""
    cgn = ComplexGroupNorm(4, 16)
    # Create input and rotate by random phase
    base = torch.randn(2, 16, 32, dtype=torch.complex64)
    phase = torch.exp(1j * torch.tensor(1.23))  # arbitrary rotation
    x = (base * phase).requires_grad_(True)
    out = cgn(x)
    # Check finite outputs
    assert torch.isfinite(out.real).all(), "Non-finite real output on phase-rotated input"
    assert torch.isfinite(out.imag).all(), "Non-finite imag output on phase-rotated input"
    # Check finite gradients
    loss = out.abs().sum()
    loss.backward()
    assert torch.isfinite(x.grad).all(), "Non-finite gradient on phase-rotated input"


def test_rayleigh_init_statistical():
    """[REVIEW FIX #9] Deterministic statistical check on Rayleigh init — NOT histogram, NOT flaky."""
    cl = ComplexLinear(256, 128)
    wr = cl.fc_r.weight.data
    wi = cl.fc_i.weight.data
    mag = (wr ** 2 + wi ** 2).sqrt()

    # 1. All magnitudes positive
    assert (mag > 0).all(), "Some magnitudes are zero"

    # 2. Re and Im should be zero-mean Gaussian (not Rayleigh-sampled)
    assert wr.mean().abs().item() < 0.05, f"Re mean {wr.mean().item():.4f} not near zero"
    assert wi.mean().abs().item() < 0.05, f"Im mean {wi.mean().item():.4f} not near zero"

    # 3. Expected sigma = 1/sqrt(fan_in + fan_out) = 1/sqrt(256+128) = 1/sqrt(384)
    expected_sigma = 1.0 / (384 ** 0.5)
    # Variance of Re should be ~ sigma^2
    actual_var_r = wr.var().item()
    expected_var = expected_sigma ** 2
    assert abs(actual_var_r - expected_var) / expected_var < 0.3, (
        f"Re variance {actual_var_r:.6f} vs expected {expected_var:.6f}"
    )

    # 4. Mean of Rayleigh(sigma) = sigma * sqrt(pi/2)
    expected_mag_mean = expected_sigma * (3.14159265 / 2) ** 0.5
    actual_mag_mean = mag.mean().item()
    assert abs(actual_mag_mean - expected_mag_mean) / expected_mag_mean < 0.2, (
        f"Mag mean {actual_mag_mean:.6f} vs expected Rayleigh mean {expected_mag_mean:.6f}"
    )


def test_complex_linear_shape():
    """ComplexLinear(16,8) maps (2,16) complex64 to (2,8) complex64."""
    cl = ComplexLinear(16, 8)
    x = torch.randn(2, 16, dtype=torch.complex64)
    out = cl(x)
    assert out.shape == (2, 8), f"shape {out.shape}, expected (2,8)"
    assert out.dtype == torch.complex64, f"dtype {out.dtype}, expected complex64"


def test_safe_modulus_gradient_at_zero():
    """[REVIEW FIX #6] safe_modulus has finite gradient even at exactly zero input."""
    x = torch.zeros(4, dtype=torch.complex128, requires_grad=True)
    out = safe_modulus(x)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all(), f"Gradient non-finite at zero: {x.grad}"


def test_safe_modulus_nonnegative():
    """safe_modulus output is always non-negative and real-valued."""
    x = torch.randn(100, dtype=torch.complex64)
    out = safe_modulus(x)
    assert (out >= 0).all(), "safe_modulus output contains negative values"
    assert out.dtype in (torch.float32, torch.float64), f"dtype {out.dtype}, expected float"
