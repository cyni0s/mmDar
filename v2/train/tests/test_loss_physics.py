"""Tests for differentiable soft-splatting (physics-informed losses)."""
import torch
import pytest


def test_soft_splat_output_shape():
    from v2.train.loss_physics import SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0], [3.0, 2.0, 0.0]]])
    occ = splat(pts)
    assert occ.shape == (1, 1, 256, 512)


def test_soft_splat_bounded():
    """Output must be in [0, 1) due to 1 - exp(-I)."""
    from v2.train.loss_physics import SoftSplat
    splat = SoftSplat()
    pts = torch.randn(2, 8192, 3) * 3 + torch.tensor([5.0, 0.0, 0.0])
    pts[..., 0] = pts[..., 0].abs()
    occ = splat(pts)
    assert occ.min() >= 0.0
    assert occ.max() < 1.0


def test_soft_splat_empty():
    from v2.train.loss_physics import SoftSplat
    splat = SoftSplat()
    pts = torch.zeros(1, 0, 3)
    occ = splat(pts)
    assert occ.sum() == 0


def test_soft_splat_gradient_flows():
    from v2.train.loss_physics import SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0]]], requires_grad=True)
    occ = splat(pts)
    occ.sum().backward()
    assert pts.grad is not None
    assert pts.grad.abs().sum() > 0


def test_soft_splat_broadside_peak():
    """A point at (5, 0, 0) should peak near center azimuth and correct range bin."""
    from v2.train.loss_physics import SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0]]])
    occ = splat(pts)
    occ_2d = occ[0, 0]
    peak_az, peak_r = divmod(occ_2d.argmax().item(), 512)
    assert 120 < peak_az < 136, f"Peak az {peak_az}"
    assert 230 < peak_r < 242, f"Peak r {peak_r}"
