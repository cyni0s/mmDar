import torch
import pytest

def test_focal_bce_zero_on_perfect():
    from v2.train.loss_occupancy import focal_bce_loss
    logits = torch.tensor([[[[10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = focal_bce_loss(logits, target)
    assert loss.item() < 0.01

def test_focal_bce_high_on_wrong():
    from v2.train.loss_occupancy import focal_bce_loss
    logits = torch.tensor([[[[-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = focal_bce_loss(logits, target)
    assert loss.item() > 1.0

def test_dice_loss_range():
    from v2.train.loss_occupancy import dice_loss
    logits = torch.randn(2, 1, 256, 512)
    target = (torch.rand(2, 1, 256, 512) > 0.99).float()
    loss = dice_loss(logits, target)
    assert 0.0 <= loss.item() <= 1.0

def test_occupancy_loss_composite():
    from v2.train.loss_occupancy import occupancy_loss
    logits = torch.randn(2, 1, 256, 512, requires_grad=True)
    target = (torch.rand(2, 1, 256, 512) > 0.99).float()
    losses = occupancy_loss(logits, target)
    assert "total" in losses
    assert "focal_bce" in losses
    assert "dice" in losses
    assert losses["total"].requires_grad
