"""Train v2 temporal model with fewer output points (2048 instead of 8192).

Hypothesis: fewer points → easier precision problem → better mod-H.
Tests n_stages=1 (2048 pts), n_stages=2 (4096 pts) against baseline n_stages=3 (8192 pts).

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_fewer_points.py --n-stages 1
"""

import sys, os, time, json, argparse, copy
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.temporal import TemporalMagPhaseFusion
from v2.data.windowed_dataset import build_windowed_dataloaders
from v2.train.loss import composite_loss


def patch_decoder_stages(model: TemporalMagPhaseFusion, n_stages: int):
    """Modify decoder to use only n_stages densification stages.

    n_stages=1 → 2048 points, n_stages=2 → 4096, n_stages=3 → 8192 (default).
    """
    assert 1 <= n_stages <= 3
    original_stages = model.decoder.stages
    model.decoder.stages = nn.ModuleList(list(original_stages)[:n_stages])
    n_pts = 1024 * (2 ** n_stages)
    print(f'Patched decoder: {n_stages} stages → {n_pts} output points')
    return n_pts


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    for radar, lidar, norm in loader:
        radar = radar.to(device)
        lidar = lidar.to(device)
        pred_pts, conf = model(radar)
        loss = composite_loss(pred_pts, conf, lidar)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, device):
    """Evaluate with GPU-accelerated Chamfer + mod-H."""
    from v2.eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch
    model.eval()
    chamfer_list, modh_list = [], []
    with torch.no_grad():
        for radar, lidar, norm in loader:
            pred_pts, conf = model(radar.to(device))
            lidar = lidar.to(device)
            B = pred_pts.shape[0]
            for i in range(B):
                chamfer_list.append(_chamfer_torch(pred_pts[i], lidar[i]))
                modh_list.append(_mod_hausdorff_torch(pred_pts[i], lidar[i]))
    return {
        'chamfer_mean': float(np.mean(chamfer_list)),
        'mod_h_mean': float(np.mean(modh_list)),
        'n_samples': len(chamfer_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-stages', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--pretrained', default='logs/v2_mag_phase/best.pt',
                        help='Single-frame pretrained checkpoint for warm start')
    parser.add_argument('--window-size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_pts_str = {1: '2048', 2: '4096', 3: '8192'}[args.n_stages]
    log_dir = f'logs/v2_fewer_pts_{n_pts_str}'

    print(f'Training with {n_pts_str} output points ({args.n_stages} densification stages)')
    print(f'Device: {device}')

    # Build model
    model = TemporalMagPhaseFusion(N_az=256, bridge_out_ch=128, max_lag=args.window_size)
    n_pts = patch_decoder_stages(model, args.n_stages)

    # Try warm start from pretrained single-frame checkpoint
    if os.path.exists(args.pretrained):
        print(f'Loading pretrained: {args.pretrained}')
        ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # Load matching keys only (decoder stages may differ)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f'  Loaded {loaded}/{len(model_state)} parameters')

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loaders = build_windowed_dataloaders(
        'v2/data/processed', window_size=args.window_size,
        batch_size=args.batch_size, num_workers=4,
    )
    print(f'Train: {len(loaders["train"].dataset)}, Val: {len(loaders["val"].dataset)}, '
          f'Test: {len(loaders["test"].dataset)}')

    os.makedirs(log_dir, exist_ok=True)
    config = vars(args)
    config['n_pts'] = n_pts
    config['total_params'] = total_params
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, loaders['train'], optimizer, device)
        val_metrics = eval_epoch(model, loaders['val'], device)
        elapsed = time.time() - t0

        val_chamfer = val_metrics['chamfer_mean']
        print(f'Ep {epoch:3d} | train_loss {train_loss:.4f} | '
              f'val_chamfer {val_chamfer:.4f} | val_modH {val_metrics["mod_h_mean"]:.4f} | '
              f'{elapsed:.0f}s')

        if val_chamfer < best_val:
            best_val = val_chamfer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, os.path.join(log_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
            }, os.path.join(log_dir, f'epoch_{epoch:03d}.pt'))

    # Final test eval
    print('\nEvaluating best checkpoint on test set...')
    best_ckpt = torch.load(os.path.join(log_dir, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_metrics = eval_epoch(model, loaders['test'], device)
    print(f'TEST: Chamfer {test_metrics["chamfer_mean"]:.4f}, '
          f'mod-H {test_metrics["mod_h_mean"]:.4f}, '
          f'N={test_metrics["n_samples"]}')
    print(f'Baseline reference: Chamfer 0.295, mod-H 0.429 (v2 8192pts)')

    # Save test results
    results = {
        'test': test_metrics,
        'best_val': best_ckpt['val_metrics'],
        'config': config,
    }
    with open(os.path.join(log_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {log_dir}/test_results.json')


if __name__ == '__main__':
    main()
