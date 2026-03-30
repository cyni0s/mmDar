"""Train physics-first Gaussian radar model.

Classical FFT (fixed) → 2D encoder → 1D deep encoder → DETR → Gaussians.
Physics does the heavy lifting, network learns the delta.

Optionally initializes DETR decoder + 1D encoder from a previous checkpoint
(e.g., from train_gaussian_radar.py run).

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_physics_gaussian.py --train
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.physics_frontend import PhysicsGaussianModel
from v2.train.train_gaussian_radar import (
    GaussianDataset, build_gaussian_dataloaders, K_PROTOTYPES,
    train_epoch, eval_points,
)
from v2.train.loss_gaussian import gaussian_composite_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=41)
    parser.add_argument('--K', type=int, default=96)
    parser.add_argument('--N-az', type=int, default=64,
                        help='FFT azimuth bins (default 64, matched to sensor)')
    parser.add_argument('--pretrained', default='',
                        help='Previous Gaussian model checkpoint for decoder/encoder init')
    parser.add_argument('--log-dir', default='logs/v2_physics_gaussian')
    parser.add_argument('--processed-dir', default='v2/data/processed')
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = PhysicsGaussianModel(
        N_az=args.N_az, T=args.window_size, K=args.K, out_ch=128,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}', flush=True)

    # Optional: transfer decoder + 1D encoder weights from previous run
    if args.pretrained and os.path.exists(args.pretrained):
        print(f'Loading pretrained: {args.pretrained}', flush=True)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Map decoder.* and encoder.blocks_1d.* / encoder.output_proj.*
            # from previous model to new model
            for prefix_from, prefix_to in [
                ('decoder.', 'decoder.'),
            ]:
                mapped_key = k.replace(prefix_from, prefix_to, 1) if k.startswith(prefix_from) else None
                if mapped_key and mapped_key in model_state and model_state[mapped_key].shape == v.shape:
                    model_state[mapped_key] = v
                    loaded += 1
                    break
        model.load_state_dict(model_state)
        print(f'  Transferred {loaded} parameters (decoder)', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loaders = build_gaussian_dataloaders(
        args.processed_dir, window_size=args.window_size,
        batch_size=args.batch_size, num_workers=4,
    )
    print(f'Train: {len(loaders["train"].dataset)}, '
          f'Val: {len(loaders["val"].dataset) if loaders["val"] else 0}', flush=True)

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    config['K_prototypes'] = K_PROTOTYPES
    config['architecture'] = 'PhysicsGaussianModel'
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_mh = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, comp_str = train_epoch(model, loaders['train'], optimizer, device, epoch)
        scheduler.step()

        val_metrics = eval_points(model, loaders['val'], device) if loaders['val'] else {}
        elapsed = time.time() - t0

        val_mh = val_metrics.get('mod_h', float('nan'))
        val_cd = val_metrics.get('chamfer', float('nan'))
        print(f'Ep {epoch:3d} | loss {train_loss:.4f} | {comp_str} | '
              f'val_cd {val_cd:.4f} val_mh {val_mh:.4f} | {elapsed:.0f}s', flush=True)

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

    # Final test eval
    print('\nTest evaluation with threshold sweep:', flush=True)
    best_ckpt = torch.load(os.path.join(args.log_dir, 'best.pt'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    for thresh in [0.0, 0.3, 0.5, 0.7]:
        test_metrics = eval_points(model, loaders['test'], device, threshold=thresh)
        print(f'  thresh={thresh:.1f}: Chamfer {test_metrics["chamfer"]:.4f}, '
              f'mod-H {test_metrics["mod_h"]:.4f}, N={test_metrics["n"]}', flush=True)

    print(f'\nBaseline: Chamfer 0.295, mod-H 0.189', flush=True)
    print(f'Gaussian 1D (41fr): Chamfer 0.356, mod-H 0.278', flush=True)


if __name__ == '__main__':
    main()
