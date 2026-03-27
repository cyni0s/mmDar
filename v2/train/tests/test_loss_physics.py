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


def test_ra_recall_perfect_overlap():
    """Perfect overlap should give near-zero loss."""
    from v2.train.loss_physics import ra_recall_loss, SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0], [3.0, 2.0, 0.0]]])
    O = splat(pts)
    loss = ra_recall_loss(O, O.detach())
    assert loss.item() < 0.05, f"Perfect overlap loss should be ~0: {loss}"

def test_ra_recall_no_overlap():
    from v2.train.loss_physics import ra_recall_loss, SoftSplat
    splat = SoftSplat()
    O_pred = splat(torch.tensor([[[2.0, 0.0, 0.0]]]))
    O_gt = splat(torch.tensor([[[8.0, 5.0, 0.0]]]))
    loss = ra_recall_loss(O_pred, O_gt.detach())
    assert loss.item() > 0.45, f"No overlap should have high loss: {loss}"

def test_ra_recall_empty_both():
    """Both empty should give zero loss, not NaN."""
    from v2.train.loss_physics import ra_recall_loss
    O_pred = torch.zeros(1, 1, 256, 512)
    O_gt = torch.zeros(1, 1, 256, 512)
    loss = ra_recall_loss(O_pred, O_gt)
    assert not torch.isnan(loss), "Empty should not be NaN"
    assert loss.item() < 0.05

def test_ra_recall_gradient():
    from v2.train.loss_physics import ra_recall_loss, SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0]]], requires_grad=True)
    gt = torch.tensor([[[5.0, 1.0, 0.0]]])
    O_pred = splat(pts)
    O_gt = splat(gt).detach()
    loss = ra_recall_loss(O_pred, O_gt)
    loss.backward()
    assert pts.grad is not None and pts.grad.abs().sum() > 0

def test_radar_support_loss_shape():
    from v2.train.loss_physics import radar_support_loss, SoftSplat
    splat = SoftSplat()
    pts = torch.tensor([[[5.0, 0.0, 0.0]]])
    O_pred = splat(pts)
    power = torch.rand(1, 256, 512)
    loss = radar_support_loss(O_pred, power)
    assert loss.shape == ()
    assert loss.item() >= 0

def test_radar_support_zero_power():
    """Zero power = no radar-positive cells = zero loss."""
    from v2.train.loss_physics import radar_support_loss, SoftSplat
    splat = SoftSplat()
    O_pred = splat(torch.tensor([[[5.0, 0.0, 0.0]]]))
    power = torch.zeros(1, 256, 512)
    loss = radar_support_loss(O_pred, power)
    assert loss.item() == 0.0
