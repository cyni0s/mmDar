"""Tests for v2/model/cvnn.py — complex-valued neural network primitives.

Covers:
- complex_soft_threshold: magnitude shrink, phase preservation, zero-input safety
- complex_soft_threshold_gradient_near_zero: finite gradients (eps-safe modulus)
- test_soft_threshold_gradcheck: Wirtinger gradcheck using cdouble (complex128)

REVIEW FIX #6: eps-safe modulus ensures finite gradients at near-zero inputs.
All gradcheck tests use cdouble (float64 complex), not complex64, as required by
torch.autograd.gradcheck for numerical precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from v2.model.cvnn import complex_soft_threshold


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
