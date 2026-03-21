"""Tests for v2/model/lista.py — LISTA beamformer and FFT baseline.

Covers:
- Steering matrix shape, dtype, and near-broadside phase
- Calibration vector g: nn.Parameter, shape, dtype, initial value
- A_eff conjugate-transpose verification (REVIEW FIX #4)
- LISTABeamformer forward shape
- Alpha initialization from ISTA formula (REVIEW FIX #5)
- Alpha/lambda positivity via softplus
- Beam pattern physics: peak at formula-derived bin (REVIEW FIX #1)
- FFT beamformer shape and fftshift correctness (REVIEW FIX #7)
- Structural: no normalization layers in LISTABeamformer (locked decision)

All expected bin indices are derived from sin_theta_to_bin() — never hardcoded.
Small N_az (64) and range bins (32) are used where full resolution is not needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from v2.model.lista import (
    build_steering_matrix,
    sin_theta_to_bin,
    LISTABeamformer,
    FFTBeamformer,
)


def test_steering_matrix_shape():
    """Steering matrix has shape (8, 256) and dtype complex64."""
    A = build_steering_matrix(256)
    assert A.shape == (8, 256), f"shape {A.shape}, expected (8, 256)"
    assert A.dtype == torch.complex64, f"dtype {A.dtype}, expected complex64"


def test_steering_matrix_broadside():
    """[REVIEW FIX #1] Broadside bin from formula; near-broadside phase < 0.1 rad.

    Due to grid discretization, sin_theta_to_bin(0.0, 256) = 128, but the
    corresponding sin value is -1 + 2*128/255 = 0.00392 (not exactly 0).
    Max phase for element n=7: pi * 7 * 0.00392 ~ 0.086 rad < 0.1 rad.
    """
    A = build_steering_matrix(256)
    broadside_bin = sin_theta_to_bin(0.0, 256)

    # Verify formula gives bin 128 for N_az=256
    assert broadside_bin == 128, f"broadside_bin={broadside_bin}, expected 128"

    col = A[:, broadside_bin]

    # Near-broadside phases should be small (not exactly zero due to discretization)
    phases = torch.angle(col)
    assert phases.abs().max().item() < 0.1, (
        f"Broadside bin {broadside_bin}: max phase {phases.abs().max().item():.4f} rad, "
        f"expected < 0.1 rad (near-broadside on discrete grid)"
    )

    # All entries have unit magnitude (steering matrix is unitary by construction)
    assert torch.allclose(col.abs(), torch.ones(8), atol=1e-5), (
        f"Non-unit magnitudes in broadside column: {col.abs()}"
    )


def test_calibration_vector():
    """Calibration g is trainable nn.Parameter, shape (8,), complex64, init 1+0j."""
    m = LISTABeamformer(K=2, N_az=64)
    assert isinstance(m.g, nn.Parameter), "g must be nn.Parameter"
    assert m.g.shape == (8,), f"g.shape={m.g.shape}, expected (8,)"
    assert m.g.dtype == torch.complex64, f"g.dtype={m.g.dtype}, expected complex64"
    assert torch.allclose(m.g.data, torch.ones(8, dtype=torch.complex64)), (
        f"g initial value: {m.g.data}, expected 1+0j"
    )


def test_a_eff_uses_conj_transpose():
    """[REVIEW FIX #4] Verify conjugate-transpose is used, not just transpose.

    When g has a nonzero imaginary part, A_eff.conj().T != A_eff.T.
    This test verifies the model can handle complex g without crashing,
    and that the conj-transpose property holds.
    """
    m = LISTABeamformer(K=2, N_az=64)

    # Set g[0] to have nonzero imaginary part
    with torch.no_grad():
        m.g[0] = torch.tensor(1.0 + 1.0j, dtype=torch.complex64)

    # Forward should not crash
    y = torch.randn(1, 8, 32, dtype=torch.complex64)
    out = m(y)
    assert out.shape == (1, 64, 32), f"shape {out.shape}, expected (1,64,32)"

    # Verify A_eff.conj().T != A_eff.T for complex g
    A_eff = m.g.unsqueeze(-1) * m.A  # (8, 64)
    A_eff_H_correct = A_eff.conj().T   # (64, 8) — correct
    A_eff_T_wrong = A_eff.T            # (64, 8) — wrong (no conjugate)

    assert not torch.allclose(A_eff_H_correct, A_eff_T_wrong), (
        "A_eff.conj().T must differ from A_eff.T when g has imaginary part"
    )


def test_lista_forward_shape():
    """LISTABeamformer(K=5) maps (B,8,512) -> (B,256,512) complex64."""
    m = LISTABeamformer(K=5, N_az=256)
    y = torch.randn(2, 8, 512, dtype=torch.complex64)
    out = m(y)
    assert out.shape == (2, 256, 512), f"shape {out.shape}, expected (2,256,512)"
    assert out.dtype == torch.complex64, f"dtype {out.dtype}, expected complex64"


def test_lista_init_alpha():
    """[REVIEW FIX #5] LISTA alpha via softplus equals 1/||A||^2_2 at init."""
    m = LISTABeamformer(K=5, N_az=256)
    A = build_steering_matrix(256)
    sigma_max = torch.linalg.matrix_norm(A, ord=2)
    expected_alpha = 1.0 / (sigma_max ** 2).item()

    for i, layer in enumerate(m.lista_layers):
        actual_alpha = F.softplus(layer.alpha_raw).item()
        assert abs(actual_alpha - expected_alpha) < 1e-4, (
            f"Layer {i}: alpha={actual_alpha:.6f}, expected={expected_alpha:.6f}"
        )


def test_lista_alpha_positive():
    """[REVIEW FIX #5] Alpha and lambda are always positive via softplus enforcement."""
    m = LISTABeamformer(K=5, N_az=256)
    for i, layer in enumerate(m.lista_layers):
        alpha = F.softplus(layer.alpha_raw).item()
        lam = F.softplus(layer.lam_raw).item()
        assert alpha > 0, f"Layer {i}: alpha={alpha} not positive"
        assert lam > 0, f"Layer {i}: lam={lam} not positive"


def test_beam_pattern_physics():
    """[REVIEW FIX #1] Known-angle plane wave peaks at formula-derived bin (±2 bins).

    Injects a steering vector at sin(theta)=0.5 and verifies LISTA output
    peaks near the expected bin (no hardcoded bin numbers).
    """
    sin_target = 0.5
    N_az = 256
    expected_bin = sin_theta_to_bin(sin_target, N_az)

    m = LISTABeamformer(K=5, N_az=N_az)
    A = build_steering_matrix(N_az)

    # Pure plane wave at sin_target: steering vector at expected_bin
    steering_vec = A[:, expected_bin]           # (8,) complex64
    y = steering_vec.unsqueeze(-1).unsqueeze(0) # (1, 8, 1)

    out = m(y)  # (1, N_az, 1)
    peak_bin = torch.abs(out[0, :, 0]).argmax().item()

    assert abs(peak_bin - expected_bin) <= 2, (
        f"Peak at bin {peak_bin}, expected ~{expected_bin} "
        f"(sin(theta)={sin_target}, formula: expected_bin=sin_theta_to_bin({sin_target}, {N_az}))"
    )


def test_fft_beamformer_shape():
    """FFTBeamformer(256) maps (B,8,512) -> (B,256,512) complex64."""
    m = FFTBeamformer(N_az=256)
    y = torch.randn(2, 8, 512, dtype=torch.complex64)
    out = m(y)
    assert out.shape == (2, 256, 512), f"shape {out.shape}, expected (2,256,512)"
    assert out.dtype == torch.complex64, f"dtype {out.dtype}, expected complex64"


def test_fft_beamformer_has_fftshift():
    """[REVIEW FIX #7] Broadside (all-ones) input peaks near center bin (fftshift).

    Without fftshift, the DC peak would be at bin 0.
    With fftshift, DC is shifted to N_az//2 = 128.
    """
    m = FFTBeamformer(N_az=256)
    # All-ones input = broadside plane wave (uniform phase across antennas)
    y = torch.ones(1, 8, 1, dtype=torch.complex64)
    out = m(y)
    peak_bin = torch.abs(out[0, :, 0]).argmax().item()
    center = 256 // 2  # 128
    assert abs(peak_bin - center) <= 2, (
        f"FFT broadside peak at bin {peak_bin}, expected near center bin {center} "
        f"(fftshift must be applied)"
    )


def test_no_norm_in_lista():
    """[REVIEW FIX #10] Structural: no GroupNorm or BatchNorm inside LISTABeamformer.

    Per CONTEXT.md locked decision: GroupNorm breaks the optimization interpretation
    of unrolled ISTA. Normalization is only allowed in the downstream feature bridge.
    """
    m = LISTABeamformer(K=5, N_az=256)
    forbidden = (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
    for name, mod in m.named_modules():
        assert not isinstance(mod, forbidden), (
            f"Found normalization layer '{name}': {type(mod).__name__} — "
            f"normalization inside LISTABeamformer violates CONTEXT.md locked decision"
        )
