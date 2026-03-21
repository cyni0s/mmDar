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


def safe_modulus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eps-safe modulus (complex absolute value) with finite gradient at zero.

    Returns the element-wise magnitude of a complex tensor as a real (float32/64)
    tensor. Using sqrt(|x|^2 + eps) instead of torch.abs avoids an undefined
    gradient at exactly zero magnitude, which can silently produce NaN during
    training when inputs are near zero.

    Args:
        x:   Complex input tensor (complex64 or complex128, any shape)
        eps: Small constant for numerical stability (default 1e-8)

    Returns:
        Real-valued tensor with the same shape and the corresponding float dtype.
        All values are >= sqrt(eps) > 0.
    """
    return (x.real ** 2 + x.imag ** 2 + eps).sqrt()


def rayleigh_init_(weight_r: torch.Tensor, weight_i: torch.Tensor) -> None:
    """Rayleigh-distributed complex weight initialization (He-style variance).

    Samples Re(W) and Im(W) independently from N(0, sigma^2), which produces
    magnitudes |W| = sqrt(Re^2 + Im^2) that follow a Rayleigh(sigma)
    distribution — i.e. E[|W|] = sigma * sqrt(pi/2) and Var(|W|) = (2-pi/2)*sigma^2.

    sigma is chosen so that the total variance Var(Re) + Var(Im) = 2*sigma^2
    equals 2 / (fan_in + fan_out), analogous to Glorot/Xavier init for real weights.

    IMPORTANT: weight_r and weight_i must be sampled from N(0, sigma^2), NOT from
    a Rayleigh distribution directly. Sampling components from Rayleigh gives
    non-Gaussian components and incorrect magnitude distribution.

    Args:
        weight_r: Real part weight tensor, shape (out_f, in_f) or (out_ch, -1)
        weight_i: Imaginary part weight tensor, same shape as weight_r

    Returns:
        None — modifies weight_r and weight_i in-place.
    """
    fan_in = weight_r.shape[1]
    fan_out = weight_r.shape[0]
    sigma = (1.0 / (fan_in + fan_out)) ** 0.5

    with torch.no_grad():
        weight_r.normal_(0.0, sigma)  # Re ~ N(0, sigma^2)
        weight_i.normal_(0.0, sigma)  # Im ~ N(0, sigma^2)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer using the apply_complex pattern.

    Implements W*x for complex weight W and complex input x using two real
    linear layers (fr, fi) combined via Wirtinger algebra:
        out_real = fr(x.real) - fi(x.imag)
        out_imag = fr(x.imag) + fi(x.real)

    Weights are initialized with rayleigh_init_ (He-style Rayleigh magnitude
    distribution). This module is used in the Phase 3 decoder but not inside
    the LISTA beamformer (LISTA uses the ISTA-form initialization).

    Args:
        in_f:  Number of input complex features
        out_f: Number of output complex features
        bias:  Whether to include bias (default True)
    """

    def __init__(self, in_f: int, out_f: int, bias: bool = True) -> None:
        super().__init__()
        self.fc_r = nn.Linear(in_f, out_f, bias=bias)
        self.fc_i = nn.Linear(in_f, out_f, bias=bias)
        rayleigh_init_(self.fc_r.weight, self.fc_i.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_complex(self.fc_r, self.fc_i, x)


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution using the apply_complex pattern.

    Axis semantics (ARCH-03):
        Input layout: (B, C_in, L) where C_in = azimuth channels, L = range bins.
        Conv1d slides along L (range), treating C_in azimuth bins as channels.
        This means the convolution MIXES azimuth bins (via channel mixing) while
        sliding along range. The intent of ARCH-03 "along azimuth" is implemented
        here as "azimuth bins are the channel dimension" — azimuth information is
        aggregated by the depthwise-like channel mixing, not by kernel stride.
        If a true azimuth-sliding convolution is needed, transpose the input to
        (B, L, C_in) before passing to this module.

    Weights are initialized with rayleigh_init_ (Glorot-style variance on the
    flattened weight tensor: out_ch x (in_ch * kernel_size)).

    Args:
        in_ch:       Number of input channels (azimuth bins)
        out_ch:      Number of output channels
        kernel_size: Kernel size for the 1D convolution (along range)
        padding:     Padding for the convolution (default 0)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv_r = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.conv_i = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        # Flatten weight to (out_ch, in_ch*kernel_size) for rayleigh_init_
        rayleigh_init_(
            self.conv_r.weight.view(out_ch, -1),
            self.conv_i.weight.view(out_ch, -1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_complex(self.conv_r, self.conv_i, x)


class ComplexGroupNorm(nn.Module):
    """Complex GroupNorm — applies GroupNorm independently to real and imaginary parts.

    APPROXIMATION: This module does NOT perform full complex whitening (which would
    require a 2x2 covariance matrix per group and is computationally expensive).
    Instead, it applies standard real-valued GroupNorm to Re(x) and Im(x)
    separately. This is a common CVNN approximation used in CNeRF and other
    phase-aware networks. It normalizes the magnitude of each part independently
    but does not whiten the complex covariance structure.

    Validated by testing finite outputs/gradients on phase-rotated inputs
    (see test_complex_group_norm_phase_rotated in tests/test_cvnn.py).

    Placement: ComplexGroupNorm is used ONLY in Stage2Bridge (before the modulus
    complex-to-real transition). It must NOT be used inside LISTABeamformer, as
    normalization breaks the optimization interpretation of unrolled ISTA.

    Args:
        num_groups:   Number of groups for GroupNorm
        num_channels: Number of channels (must be divisible by num_groups)
    """

    def __init__(self, num_groups: int, num_channels: int) -> None:
        super().__init__()
        self.gn_r = nn.GroupNorm(num_groups, num_channels)
        self.gn_i = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.gn_r(x.real), self.gn_i(x.imag))
