"""Decisive capacity test: train physics-first Gaussian on mixed-ID split.

Proves whether the architecture can learn high-ID environments when given the data.
If high-ID test mod-H drops to ~0.15-0.20 → data problem, architecture works.
If high-ID test mod-H stays ~0.30+ → architecture problem.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_mixed_split.py --train
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.physics_frontend import PhysicsGaussianModel
from v2.data.split_mixed import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
from v2.data.augment import augment_sample
from v2.train.loss_gaussian import gaussian_composite_loss
from v2.eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch
from v2.train.train_physics_augmented import (
    AugmentedGaussianDataset, train_epoch, eval_per_trajectory, K_PROTOTYPES,
)


def build_mixed_dataloaders(processed_dir, window_size=41, batch_size=4, num_workers=4):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=41)
    parser.add_argument('--K', type=int, default=96)
    parser.add_argument('--N-az', type=int, default=64)
    parser.add_argument('--log-dir', default='logs/v2_mixed_split')
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
    print(f'Split: {len(TRAIN_TRAJS)} train (inc high-ID), '
          f'{len(VAL_TRAJS)} val (inc high-ID), {len(TEST_TRAJS)} test', flush=True)
    print(f'High-ID in train: {[t for t in TRAIN_TRAJS if t > 200]}', flush=True)
    print(f'High-ID in test:  {[t for t in TEST_TRAJS if t > 200]}', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loaders = build_mixed_dataloaders(
        args.processed_dir, window_size=args.window_size,
        batch_size=args.batch_size, num_workers=4)
    print(f'Train samples: {len(loaders["train"].dataset)}', flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    config['split'] = 'mixed'
    config['high_id_train'] = [t for t in TRAIN_TRAJS if t > 200]
    config['high_id_val'] = [t for t in VAL_TRAJS if t > 200]
    config['high_id_test'] = [t for t in TEST_TRAJS if t > 200]
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_mh = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, comp_str = train_epoch(model, loaders['train'], optimizer, device, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = eval_per_trajectory(
                model, args.processed_dir, VAL_TRAJS, device,
                args.window_size, threshold=0.3)
            val_mh = val_metrics['mod_h_traj_median']
        else:
            val_mh = float('nan')

        elapsed = time.time() - t0
        val_str = f'val_mh_traj {val_mh:.4f}' if not np.isnan(val_mh) else 'val skip'
        print(f'Ep {epoch:3d} | loss {train_loss:.4f} | {comp_str} | {val_str} | {elapsed:.0f}s',
              flush=True)

        if val_mh < best_val_mh:
            best_val_mh = val_mh
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics if not np.isnan(val_mh) else {},
                'config': config,
            }, os.path.join(args.log_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.log_dir, f'epoch_{epoch:03d}.pt'))

    # Final test: per-trajectory breakdown, separate low-ID vs high-ID
    print('\n=== TEST EVALUATION ===', flush=True)
    best_ckpt = torch.load(os.path.join(args.log_dir, 'best.pt'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    print(f'Best epoch: {best_ckpt["epoch"]}', flush=True)

    low_id_test = [t for t in TEST_TRAJS if t < 200]
    high_id_test = [t for t in TEST_TRAJS if t > 200]

    for label, trajs in [('ALL test', TEST_TRAJS),
                          ('Low-ID test', low_id_test),
                          ('High-ID test', high_id_test)]:
        m = eval_per_trajectory(model, args.processed_dir, trajs, device,
                                args.window_size, threshold=0.3)
        print(f'{label:>14}: mod-H traj_median={m["mod_h_traj_median"]:.4f}, '
              f'traj_mean={m["mod_h_traj_mean"]:.4f}, '
              f'traj_max={m["mod_h_traj_max"]:.4f}, '
              f'chamfer={m["chamfer_traj_median"]:.4f}', flush=True)

    print(f'\nBaseline: mod-H 0.189', flush=True)
    print(f'Physics-first (low-ID only train): mod-H 0.261', flush=True)


if __name__ == '__main__':
    main()
