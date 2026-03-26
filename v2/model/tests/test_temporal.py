"""Tests for temporal cross-attention module.

TDD: written before implementation. All tests must fail initially,
then pass after temporal.py is implemented.
"""

import torch
import pytest


def test_temporal_identity_single_frame():
    """N=1 should return input unchanged."""
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 1, 128, 512)
    out = module(x)
    assert out.shape == (2, 128, 512)
    assert torch.allclose(out, x[:, 0], atol=1e-6), "N=1 must be identity"


def test_temporal_multi_frame_shape():
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 5, 128, 512)
    out = module(x)
    assert out.shape == (2, 128, 512)


def test_temporal_zero_init_then_differs_after_step():
    """At init, delta ≈ 0 (preserves pretrained single-frame behavior).
    After one gradient step, multi-frame output should differ from single-frame.
    """
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 5, 128, 512)

    # Before training: zero-init FFN → delta ≈ 0 → multi ≈ single
    out_multi_pre = module(x).detach()
    out_single_pre = module(x[:, -1:, :, :]).detach()
    assert torch.allclose(out_multi_pre, out_single_pre, atol=1e-4), \
        "At init, multi and single should match (zero-init FFN)"

    # One gradient step
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    loss = module(x).sum()
    loss.backward()
    optimizer.step()

    # After training: outputs should now differ
    with torch.no_grad():
        out_multi_post = module(x)
        out_single_post = module(x[:, -1:, :, :])
    assert not torch.allclose(out_multi_post, out_single_post, atol=1e-3), \
        "After training, multi-frame should differ from single-frame"


def test_temporal_param_count():
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4, ff_dim=256, max_lag=16)
    n_params = sum(p.numel() for p in module.parameters())
    assert n_params < 200_000, f"Too many params: {n_params}"
    print(f"Temporal module params: {n_params:,}")


def test_temporal_variable_n():
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    for N in [1, 3, 5, 8]:
        x = torch.randn(2, N, 128, 512)
        out = module(x)
        assert out.shape == (2, 128, 512), f"Failed for N={N}"


def test_temporal_gradient_flows():
    """Gradients should flow through the temporal module."""
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 5, 128, 512, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_full_model_end_to_end():
    from v2.model.temporal import TemporalMagPhaseFusion
    model = TemporalMagPhaseFusion()
    x = torch.randn(2, 5, 8, 512, dtype=torch.complex64)
    pts, conf = model(x)
    assert pts.shape == (2, 8192, 3)
    assert conf.shape == (2, 8192, 1)


def test_full_model_single_frame():
    from v2.model.temporal import TemporalMagPhaseFusion
    model = TemporalMagPhaseFusion()
    x = torch.randn(2, 1, 8, 512, dtype=torch.complex64)
    pts, conf = model(x)
    assert pts.shape == (2, 8192, 3)
