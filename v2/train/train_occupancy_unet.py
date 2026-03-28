# v2/train/train_occupancy_unet.py
"""Train symmetric U-Net on LISTA log_power features for polar occupancy.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_occupancy_unet.py
"""
import os, sys, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.unet_occupancy import UNetOcc
from v2.data.lista_dataset import build_lista_dataloaders


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice loss (same as baseline RadarHD)."""
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        pred = model(features)
        loss = composite_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            pred = model(features)
            loss = composite_loss(pred, labels)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log-dir', default='logs/v2_lista_unet_occ')
    parser.add_argument('--processed-dir', default='v2/data/processed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = UNetOcc(n_channels=41, n_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    loaders = build_lista_dataloaders(
        args.processed_dir, history=40,
        batch_size=args.batch_size, num_workers=4,
    )
    print(f'Train: {len(loaders["train"].dataset)} samples')
    if loaders['val']:
        print(f'Val: {len(loaders["val"].dataset)} samples')

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, loaders['train'], optimizer, device)
        val_loss = val_epoch(model, loaders['val'], device) if loaders['val'] else float('nan')
        elapsed = time.time() - t0

        print(f'Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(args.log_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(args.log_dir, f'epoch_{epoch:03d}.pt'))

    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints in: {args.log_dir}')


if __name__ == '__main__':
    main()
