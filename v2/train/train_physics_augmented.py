"""Train physics-first Gaussian model WITH data augmentation + expanded val.

Key changes from train_physics_gaussian.py:
1. Uses split_v2 (8 val trajectories instead of 4)
2. Applies augmentation: horizontal flip, complex noise, temporal masking
3. Per-trajectory evaluation (not just frame-average)

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_physics_augmented.py --train
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.physics_frontend import PhysicsGaussianModel
from v2.data.split_v2 import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
from v2.data.augment import augment_sample
from v2.train.loss_gaussian import gaussian_composite_loss
from v2.eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch

K_PROTOTYPES = 64


class AugmentedGaussianDataset(Dataset):
    """Gaussian dataset with online augmentation."""

    def __init__(self, traj_id: int, processed_dir: str, window_size: int = 41,
                 augment: bool = False):
        self.window_size = window_size
        self.augment = augment
        self.radar = torch.load(
            os.path.join(processed_dir, f'radar_{traj_id}.pt'), weights_only=True)
        self.lidar = torch.load(
            os.path.join(processed_dir, f'lidar_{traj_id}.pt'), weights_only=True)
        self.protos = torch.load(
            os.path.join(processed_dir, f'proto_{traj_id}.pt'), weights_only=True)
        self.n_frames = self.radar.shape[0]
        self.traj_id = traj_id

    def __len__(self):
        return max(0, self.n_frames - self.window_size + 1)

    def __getitem__(self, idx):
        end = idx + self.window_size
        radar = self.radar[idx:end]
        lidar = self.lidar[end - 1]
        protos = self.protos[end - 1]

        if self.augment:
            radar, lidar, protos = augment_sample(radar, lidar, protos)

        return radar, lidar, protos


def build_augmented_dataloaders(processed_dir, window_size=41, batch_size=4, num_workers=4):
    split_configs = {
        'train': (TRAIN_TRAJS, True, True),
        'val': (VAL_TRAJS, False, False),
        'test': (TEST_TRAJS, False, False),
    }
    loaders = {}
    for split, (trajs, shuffle, augment) in split_configs.items():
        datasets = []
        for tid in trajs:
            if os.path.exists(os.path.join(processed_dir, f'proto_{tid}.pt')):
                datasets.append(AugmentedGaussianDataset(
                    tid, processed_dir, window_size, augment=augment))
        if datasets:
            loaders[split] = DataLoader(
                ConcatDataset(datasets), batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        else:
            loaders[split] = None
    return loaders


def train_epoch(model, loader, optimizer, device, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0
    loss_components = {}
    n_batches = 0
    for radar, lidar, protos in loader:
        radar = radar.to(device)
        lidar_xy = lidar[:, :, :2].to(device)
        protos = protos.to(device)
        n_gt = torch.full((radar.shape[0],), K_PROTOTYPES, device=device)
        out = model(radar)
        losses = gaussian_composite_loss(out, protos, lidar_xy, n_gt, epoch=epoch)
        optimizer.zero_grad()
        losses['total'].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += losses['total'].item()
        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0) + v.item()
        n_batches += 1
    avg = total_loss / max(n_batches, 1)
    comp_str = ' '.join(f'{k}={v/n_batches:.3f}' for k, v in loss_components.items())
    return avg, comp_str


def eval_per_trajectory(model, processed_dir, traj_ids, device, window_size=41,
                        threshold=0.3):
    """Evaluate per-trajectory, return trajectory-level median mod-H."""
    model.eval()
    traj_modh = []
    traj_chamfer = []

    with torch.no_grad():
        for tid in traj_ids:
            if not os.path.exists(os.path.join(processed_dir, f'proto_{tid}.pt')):
                continue
            ds = AugmentedGaussianDataset(tid, processed_dir, window_size, augment=False)
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            cd_list, mh_list = [], []
            for radar, lidar, protos in loader:
                points = model.predict_points(radar.to(device), threshold=threshold)
                gt_xy = lidar[0, :, :2].to(device)
                pred = points[0]
                if pred.shape[0] < 2:
                    continue
                cd_list.append(_chamfer_torch(pred, gt_xy))
                mh_list.append(_mod_hausdorff_torch(pred, gt_xy))

            if cd_list:
                traj_modh.append(float(np.mean(mh_list)))
                traj_chamfer.append(float(np.mean(cd_list)))

    if not traj_modh:
        return {'mod_h_traj_median': float('nan'), 'chamfer_traj_median': float('nan'),
                'n_trajs': 0}

    return {
        'mod_h_traj_median': float(np.median(traj_modh)),
        'chamfer_traj_median': float(np.median(traj_chamfer)),
        'mod_h_traj_mean': float(np.mean(traj_modh)),
        'mod_h_traj_max': float(np.max(traj_modh)),
        'n_trajs': len(traj_modh),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=41)
    parser.add_argument('--K', type=int, default=96)
    parser.add_argument('--N-az', type=int, default=64)
    parser.add_argument('--log-dir', default='logs/v2_physics_augmented')
    parser.add_argument('--processed-dir', default='v2/data/processed')
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = PhysicsGaussianModel(N_az=args.N_az, T=args.window_size, K=args.K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}', flush=True)
    print(f'Split: {len(TRAIN_TRAJS)} train, {len(VAL_TRAJS)} val, {len(TEST_TRAJS)} test', flush=True)
    print(f'Augmentation: flip(0.5) + noise(0.5, SNR 15-25dB) + mask(0.3, 2-6 frames)', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loaders = build_augmented_dataloaders(
        args.processed_dir, window_size=args.window_size,
        batch_size=args.batch_size, num_workers=4)
    print(f'Train samples: {len(loaders["train"].dataset)}', flush=True)
    print(f'Val samples: {len(loaders["val"].dataset) if loaders["val"] else 0}', flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    config['augmentation'] = 'flip(0.5)+noise(0.5,SNR15-25)+mask(0.3,2-6)'
    config['val_trajs'] = VAL_TRAJS
    config['train_trajs'] = TRAIN_TRAJS
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_mh = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, comp_str = train_epoch(model, loaders['train'], optimizer, device, epoch)
        scheduler.step()

        # Per-trajectory val evaluation every 5 epochs (expensive)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = eval_per_trajectory(
                model, args.processed_dir, VAL_TRAJS, device,
                args.window_size, threshold=0.3)
            val_mh = val_metrics['mod_h_traj_median']
        else:
            val_mh = float('nan')
            val_metrics = {}

        elapsed = time.time() - t0

        val_str = f'val_mh_traj {val_mh:.4f}' if not np.isnan(val_mh) else 'val skip'
        print(f'Ep {epoch:3d} | loss {train_loss:.4f} | {comp_str} | {val_str} | {elapsed:.0f}s',
              flush=True)

        if val_mh < best_val_mh:
            best_val_mh = val_mh
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, os.path.join(args.log_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.log_dir, f'epoch_{epoch:03d}.pt'))

    # Final test eval — per-trajectory
    print('\nTest evaluation (per-trajectory):', flush=True)
    best_ckpt = torch.load(os.path.join(args.log_dir, 'best.pt'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    print(f'Best epoch: {best_ckpt["epoch"]}', flush=True)

    for thresh in [0.0, 0.3, 0.5]:
        test_m = eval_per_trajectory(
            model, args.processed_dir, TEST_TRAJS, device,
            args.window_size, threshold=thresh)
        print(f'  thresh={thresh:.1f}: mod-H traj_median={test_m["mod_h_traj_median"]:.4f}, '
              f'traj_mean={test_m["mod_h_traj_mean"]:.4f}, traj_max={test_m["mod_h_traj_max"]:.4f}, '
              f'chamfer_median={test_m["chamfer_traj_median"]:.4f}', flush=True)

    print(f'\nBaseline: mod-H 0.189, Chamfer 0.295', flush=True)
    print(f'Physics-first (no aug): mod-H 0.261, Chamfer 0.330', flush=True)


if __name__ == '__main__':
    main()
