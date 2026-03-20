"""Train Azimuth1DNet — per-range-bin 1D azimuth super-resolution.

Tests whether 64→512 azimuth sharpening is fundamentally a per-range-bin 1D
problem. Uses the same baseline Dataset (41-channel stacking) and loss
(BCE + Dice) for direct comparison.

Usage:
  python3 train_1d.py --batch 12 --lr 7e-5 --epochs 30
  python3 train_1d.py --batch 12 --lr 7e-5 --epochs 1 --dry_run
  python3 train_1d.py --n_channels 1 --batch 12 --lr 7e-5 --epochs 30  # single-frame control
"""

import os
import sys
import time
import json
import argparse
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_test_utils.model_1d import Azimuth1DNet
from train_test_utils.dataloader import Dataset
from train_test_utils.dice_score import dice_loss


class TensorDataset(torch.utils.data.Dataset):
    """Loads pre-computed stacked tensors from a .pt file.

    Use precompute_dataset.py to create these files once. Then training
    loads in seconds instead of hours.

    Fallback: if .pt file doesn't exist, loads from PNGs (slow).
    """

    def __init__(self, pt_path, fallback_dataset=None):
        import time
        if os.path.exists(pt_path):
            print(f'    Loading {pt_path}...', end=' ', flush=True)
            t0 = time.time()
            data = torch.load(pt_path, weights_only=True)
            self.X = data['X']
            self.Y = data['Y']
            elapsed = time.time() - t0
            print(f'{len(self.X)} samples in {elapsed:.1f}s')
        elif fallback_dataset is not None:
            print(f'    {pt_path} not found, loading from PNGs (slow)...')
            n = len(fallback_dataset)
            x0, y0 = fallback_dataset[0]
            self.X = torch.empty(n, *x0.shape)
            self.Y = torch.empty(n, *y0.shape)
            self.X[0] = x0
            self.Y[0] = y0
            for i in range(1, n):
                self.X[i], self.Y[i] = fallback_dataset[i]
                if (i + 1) % 2000 == 0:
                    print(f'      {i+1}/{n}')
            print(f'    Loaded {n} samples')
        else:
            raise FileNotFoundError(f'{pt_path} not found. Run: '
                                    'docker compose run --rm mmdar python3 train_1d.py --precompute')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def parse_args():
    parser = argparse.ArgumentParser(description='Train Azimuth1DNet (1D ablation)')
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_channels', type=int, default=41,
                        help='Input channels: 41 (stacked) or 1 (single-frame control)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden channel width in 1D model')
    parser.add_argument('--n_blocks', type=int, default=6,
                        help='Number of residual blocks')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--basepath', type=str, default='./dataset_5/')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment naming
    dt = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    M = args.n_channels - 1  # history frames (40 for 41ch, 0 for 1ch)
    name = f'1d_h{args.hidden}_b{args.n_blocks}_{args.n_channels}ch_b{args.batch}_lr{args.lr}_{dt}'
    LOG_DIR = os.path.join('./logs', name)
    os.makedirs(LOG_DIR, exist_ok=True)

    try:
        import subprocess
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = 'unknown'

    # Save hyperparameters
    params = {
        'model': 'Azimuth1DNet',
        'n_channels': args.n_channels,
        'hidden': args.hidden,
        'n_blocks': args.n_blocks,
        'batch_size': args.batch,
        'lr': args.lr,
        'num_epochs': args.epochs,
        'seed': args.seed,
        'msew': 0.9,
        'dicew': 0.1,
        'optim': 'adam',
        'weight_decay': 0.0005,
        'name': name,
        'git_sha': git_sha,
    }
    with open(os.path.join(LOG_DIR, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    writer = SummaryWriter(LOG_DIR)

    print(f'\n{"=" * 60}')
    print(f'EXPERIMENT: {name}')
    print(f'  n_channels={args.n_channels} (M={M}), hidden={args.hidden}, n_blocks={args.n_blocks}')
    print(f'  batch={args.batch}, lr={args.lr}, epochs={args.epochs}')
    print(f'  device={device}, git={git_sha}')
    print(f'{"=" * 60}\n')

    # Datasets — load from pre-computed .pt files (fast) or fall back to PNGs (slow)
    print('Loading datasets...')
    train_pt = os.path.join(args.basepath, f'train_{args.n_channels}ch.pt')
    test_pt = os.path.join(args.basepath, f'test_{args.n_channels}ch.pt')
    train_dataset = TensorDataset(train_pt,
                                   fallback_dataset=Dataset(args.basepath, 'train',
                                                            ABINS_LIDAR_ORIG=512, M=M)
                                   if not os.path.exists(train_pt) else None)
    test_dataset = TensorDataset(test_pt,
                                  fallback_dataset=Dataset(args.basepath, 'test',
                                                           ABINS_LIDAR_ORIG=512, M=M)
                                  if not os.path.exists(test_pt) else None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch,
                              shuffle=True, num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                             shuffle=False, num_workers=0,
                             pin_memory=(device.type == 'cuda'))

    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Test:  {len(test_dataset)} samples')
    print(f'  Steps/epoch: {len(train_loader)}\n')

    # Model
    model = Azimuth1DNet(n_channels=args.n_channels, hidden=args.hidden,
                         n_blocks=args.n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model params: {n_params:,} ({n_params/1e6:.1f}M)\n')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    bce_loss_fn = torch.nn.BCELoss()

    # Training loop
    t0 = time.time()
    best_val_loss = float('inf')
    best_val_epoch = -1
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        epoch_losses = []

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(X)

            loss = 0.9 * bce_loss_fn(preds, y) + 0.1 * dice_loss(preds, y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            writer.add_scalar('train/batch_loss', loss_val, global_step)
            global_step += 1

            if args.dry_run:
                print(f'  [dry_run] X={list(X.shape)}, preds={list(preds.shape)}, y={list(y.shape)}')
                print(f'  [dry_run] loss={loss_val:.6f}')
                print('[dry_run] PASS')
                writer.close()
                return LOG_DIR, name, time.time() - t0, {}

        # End-of-epoch
        epoch_loss = float(np.mean(epoch_losses))
        writer.add_scalar('train/loss', epoch_loss, epoch)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            elapsed = (time.time() - t0) / 60
            print(f'  Epoch {epoch + 1:3d}/{args.epochs}  '
                  f'loss={epoch_loss:.6f}  elapsed={elapsed:.1f}min')

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    preds = model(X)
                    vl = 0.9 * bce_loss_fn(preds, y) + 0.1 * dice_loss(preds, y)
                    val_losses.append(vl.item())
            val_loss = float(np.mean(val_losses))
            writer.add_scalar('val/loss', val_loss, epoch)
            print(f'    [val] epoch {epoch + 1}  val_loss={val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                torch.save(
                    {'state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch + 1,
                     'val_loss': val_loss},
                    os.path.join(LOG_DIR, 'best.pt_gen'),
                )
                print(f'    [val] -> new best (loss={val_loss:.6f}), saved best.pt_gen')
            model.train()

        # Periodic checkpoints
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                {'state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch + 1},
                os.path.join(LOG_DIR, f'{epoch + 1:03d}.pt_gen'),
            )

    train_time = time.time() - t0
    print(f'\nTraining complete: {train_time / 60:.1f} min '
          f'({train_time / args.epochs:.1f}s/epoch)')
    print(f'Best val_loss={best_val_loss:.6f} at epoch {best_val_epoch}')

    summary_data = {
        'name': name,
        'n_channels': args.n_channels,
        'hidden': args.hidden,
        'n_blocks': args.n_blocks,
        'batch': args.batch,
        'lr': args.lr,
        'epochs': args.epochs,
        'train_time_sec': train_time,
        'sec_per_epoch': train_time / args.epochs,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch,
        'n_params': n_params,
        'git_sha': git_sha,
    }
    with open(os.path.join(LOG_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    writer.close()
    return LOG_DIR, name, train_time, summary_data


def main():
    args = parse_args()
    LOG_DIR, name, train_time, summary = train(args)
    if not args.dry_run:
        print(f'\nLog dir: {LOG_DIR}')


if __name__ == '__main__':
    main()
