"""Occupancy losses: focal BCE (handles class imbalance) + Dice (overlap).
total = focal_bce + dice_weight * dice
"""
import torch
import torch.nn.functional as F


def focal_bce_loss(logits, target, alpha=0.25, gamma=2.0):
    """Focal BCE on logits. alpha=positive class weight."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_weight = alpha_t * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def dice_loss(logits, target, smooth=1.0):
    """Soft Dice loss. Returns 1 - Dice coefficient."""
    pred = torch.sigmoid(logits)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def occupancy_loss(logits, target, dice_weight=1.0, focal_alpha=0.25, focal_gamma=2.0):
    """Composite: focal BCE + Dice. Returns dict with total, focal_bce, dice."""
    f_bce = focal_bce_loss(logits, target, focal_alpha, focal_gamma)
    d_loss = dice_loss(logits, target)
    total = f_bce + dice_weight * d_loss
    return {"total": total, "focal_bce": f_bce, "dice": d_loss}
