"""Retrain baseline UNet1 with focal loss for better precision.

The threshold sweep showed the baseline has useful signal in sigmoid 0.004-0.010 range.
Focal loss (γ=2) should help the model produce sharper, more confident predictions
by downweighting easy negatives (the vast majority of empty cells).

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_baseline_focal.py
"""

import sys, os, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from train_test_utils.model import UNet1
from train_test_utils.dataloader import Dataset


def focal_bce(pred, target, gamma=2.0, alpha=0.25):
    """Focal binary cross-entropy loss.

    Focal loss = -alpha * (1-p_t)^gamma * log(p_t)
    where p_t = p if target=1, else 1-p.

    alpha < 0.5 downweights the positive (occupied) class,
    making the model more conservative (fewer false positives).
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    alpha_weight = alpha * target + (1 - alpha) * (1 - target)
    return (focal_weight * alpha_weight * bce).mean()


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def composite_loss(pred, target, gamma=2.0, alpha=0.25):
    return focal_bce(pred, target, gamma, alpha) + dice_loss(pred, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--alpha', type=float, default=0.25, help='Focal loss alpha (positive class weight)')
    parser.add_argument('--log-dir', default='logs/baseline_focal')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = UNet1(41, 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: UNet1, {n_params:,} params', flush=True)
    print(f'Focal loss: gamma={args.gamma}, alpha={args.alpha}', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Load data (same as baseline)
    orig_size = [256, 64, 512]
    train_set = Dataset('dataset_5/', 'train',
                        RBINS=orig_size[0], ABINS_RADAR=orig_size[1], ABINS_LIDAR=orig_size[2],
                        RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1],
                        ABINS_LIDAR_ORIG=orig_size[2], M=40)
    test_set = Dataset('dataset_5/', 'test',
                       RBINS=orig_size[0], ABINS_RADAR=orig_size[1], ABINS_LIDAR=orig_size[2],
                       RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1],
                       ABINS_LIDAR_ORIG=orig_size[2], M=40)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)
    print(f'Train: {len(train_set)}, Test: {len(test_set)}', flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = composite_loss(pred, label, args.gamma, args.alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Quick val loss on test set (same as baseline training pattern)
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pred = model(data)
                val_loss += composite_loss(pred, label, args.gamma, args.alpha).item()
                val_n += 1
        val_loss /= max(val_n, 1)

        elapsed = time.time() - t0
        print(f'Ep {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.0f}s', flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.log_dir, 'best.pt_gen'))

        if (epoch + 1) % 10 == 0:
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.log_dir, f'{epoch:03d}.pt_gen'))

    print(f'Best val loss: {best_loss:.4f}', flush=True)


if __name__ == '__main__':
    main()
