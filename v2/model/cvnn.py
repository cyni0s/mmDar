"""Complex-valued neural network primitives for mmDar v2.

Implements the apply_complex pattern and complex soft-thresholding
with Wirtinger-correct autograd. No external CVNN library dependency.

All ops verified with torch.autograd.gradcheck (see tests/test_cvnn.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_complex(fr: nn.Module, fi: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a complex-valued linear map using two real modules.

    Implements the Wirtinger-correct split-real/imag computation:
        out_real = fr(x.real) - fi(x.imag)
        out_imag = fr(x.imag) + fi(x.real)   <-- PLUS sign (Wirtinger correct)

    This is the apply_complex pattern from CNeRF (cyni0s/CNeRF/model.py),
    adapted for native PyTorch complex tensors.

    Args:
        fr: Real-part module (e.g. nn.Linear or nn.Conv1d)
        fi: Imaginary-part module (same shape as fr)
        x: Complex input tensor (dtype=torch.complex64 or complex128)

    Returns:
        Complex output tensor with the same batch/spatial dimensions as fr output.
    """
    real_out = fr(x.real) - fi(x.imag)
    imag_out = fr(x.imag) + fi(x.real)
    return torch.complex(real_out, imag_out)


def complex_soft_threshold(
    x: torch.Tensor,
    lam: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Complex soft-thresholding (shrinkage) with eps-safe modulus.

    Shrinks the magnitude of x by softplus(lam) while preserving phase.
    Handles zero-magnitude inputs gracefully (no NaN gradient at 0).

    Implementation:
        mag = sqrt(x.real^2 + x.imag^2 + eps)   # eps-safe, finite gradient at 0
        shrunk = relu(mag - softplus(lam))         # non-negative residual magnitude
        out = (shrunk / mag) * x                   # phase preserved, magnitude shrunk

    REVIEW FIX #5: softplus enforces non-negative threshold (lam is raw learned value)
    REVIEW FIX #6: eps in modulus prevents NaN gradient at zero-magnitude inputs

    Args:
        x:   Complex input tensor (any shape, complex64 or complex128)
        lam: Raw threshold parameter (real scalar or tensor); effective threshold
             is F.softplus(lam), which is always positive.
        eps: Small constant for numerical stability in modulus (default 1e-8)

    Returns:
        Complex tensor with same shape as x; zero where |x| <= softplus(lam).
    """
    # eps-safe modulus — avoids undefined gradient at |x|=0
    mag = (x.real ** 2 + x.imag ** 2 + eps).sqrt()

    # Effective threshold is always positive via softplus
    effective_lam = F.softplus(lam)

    # Shrink magnitude, clamp to zero (relu), preserve phase
    shrunk = F.relu(mag - effective_lam)

    # Divide by mag (eps already added, so no div-by-zero)
    return (shrunk / mag) * x
