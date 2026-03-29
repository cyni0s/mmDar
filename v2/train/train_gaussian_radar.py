"""Train Gaussian radar model: raw IQ → Gaussian set → point cloud.

The leap experiment: learned beamspace + Gaussian set decoder + mod-H-aligned loss.
All new components, trained end-to-end.

Pre-requisite: fit GT prototypes offline (run with --fit-prototypes first).

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_gaussian_radar.py --fit-prototypes
  docker compose run --rm mmdar python3 v2/train/train_gaussian_radar.py --train
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.gaussian_head import GaussianRadarModel
from v2.data.windowed_dataset import build_windowed_dataloaders
from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, ALL_TRAJS
from v2.train.loss_gaussian import gaussian_composite_loss
from v2.eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch

K_PROTOTYPES = 64  # number of GT prototypes per frame


# ---------------------------------------------------------------------------
# Step 1: Fit GT prototypes offline
# ---------------------------------------------------------------------------

def fit_prototypes(processed_dir: str, K: int = K_PROTOTYPES):
    """Fit K-Means prototypes to each lidar frame. Save as .pt files.

    For each trajectory, saves: proto_{tid}.pt → (N, K, 2) float32
    """
    from sklearn.cluster import MiniBatchKMeans

    for tid in ALL_TRAJS:
        lidar_path = os.path.join(processed_dir, f'lidar_{tid}.pt')
        if not os.path.exists(lidar_path):
            continue

        lidar = torch.load(lidar_path, weights_only=True).numpy()  # (N, 8192, 3)
        N = lidar.shape[0]
        protos = np.zeros((N, K, 2), dtype=np.float32)

        for i in range(N):
            xy = lidar[i, :, :2].astype(np.float64)
            mask = (xy[:, 0] > 0) & (xy[:, 0] <= 10.8) & (np.abs(xy[:, 1]) <= 10.8)
            xy = xy[mask]
            if len(xy) < K:
                # Pad with copies if too few valid points
                if len(xy) < 2:
                    continue
                km = MiniBatchKMeans(n_clusters=min(K, len(xy)), n_init=1,
                                     random_state=0, batch_size=256, max_iter=30)
                km.fit(xy)
                centers = km.cluster_centers_
                # Pad to K
                n_reps = K // len(centers) + 1
                centers = np.tile(centers, (n_reps, 1))[:K]
            else:
                km = MiniBatchKMeans(n_clusters=K, n_init=1, random_state=0,
                                     batch_size=512, max_iter=30)
                km.fit(xy)
                centers = km.cluster_centers_
            protos[i] = centers.astype(np.float32)

        out_path = os.path.join(processed_dir, f'proto_{tid}.pt')
        torch.save(torch.from_numpy(protos), out_path)
        print(f'Traj {tid}: {N} frames → {out_path}', flush=True)

    print('Prototype fitting done.', flush=True)


# ---------------------------------------------------------------------------
# Step 2: Dataset that loads radar + prototypes + full lidar
# ---------------------------------------------------------------------------

class GaussianDataset(torch.utils.data.Dataset):
    """Loads windowed radar IQ + GT prototypes + full lidar for training."""

    def __init__(self, traj_id: int, processed_dir: str, window_size: int = 8):
        self.window_size = window_size
        self.radar = torch.load(
            os.path.join(processed_dir, f'radar_{traj_id}.pt'), weights_only=True
        )  # (N, 8, 512) complex
        self.lidar = torch.load(
            os.path.join(processed_dir, f'lidar_{traj_id}.pt'), weights_only=True
        )  # (N, 8192, 3) float
        self.protos = torch.load(
            os.path.join(processed_dir, f'proto_{traj_id}.pt'), weights_only=True
        )  # (N, K, 2) float
        self.n_frames = self.radar.shape[0]

    def __len__(self):
        return max(0, self.n_frames - self.window_size + 1)

    def __getitem__(self, idx):
        end = idx + self.window_size
        radar_window = self.radar[idx:end]       # (W, 8, 512) complex
        lidar = self.lidar[end - 1]              # (8192, 3) float — target frame
        protos = self.protos[end - 1]            # (K, 2) float — target frame prototypes
        return radar_window, lidar, protos


def build_gaussian_dataloaders(processed_dir, window_size=8, batch_size=8, num_workers=4):
    from torch.utils.data import ConcatDataset, DataLoader
    split_configs = {
        'train': (TRAIN_TRAJS, True),
        'val': (VAL_TRAJS, False),
        'test': (TEST_TRAJS, False),
    }
    loaders = {}
    for split, (trajs, shuffle) in split_configs.items():
        datasets = []
        for tid in trajs:
            proto_path = os.path.join(processed_dir, f'proto_{tid}.pt')
            if os.path.exists(proto_path):
                datasets.append(GaussianDataset(tid, processed_dir, window_size))
        if datasets:
            loaders[split] = DataLoader(
                ConcatDataset(datasets), batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            )
        else:
            loaders[split] = None
    return loaders


# ---------------------------------------------------------------------------
# Step 3: Training loop
# ---------------------------------------------------------------------------

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


def eval_points(model, loader, device, threshold=0.0):
    """Evaluate point cloud metrics (Chamfer + mod-H) on predicted centers."""
    model.eval()
    cd_list, mh_list = [], []
    with torch.no_grad():
        for radar, lidar, protos in loader:
            radar = radar.to(device)
            lidar_xy = lidar[:, :, :2].to(device)

            point_clouds = model.predict_points(radar, threshold=threshold)

            for b in range(len(point_clouds)):
                pred = point_clouds[b]  # (N_b, 2)
                gt = lidar_xy[b]        # (8192, 2)
                if pred.shape[0] < 2:
                    continue
                cd_list.append(_chamfer_torch(pred, gt))
                mh_list.append(_mod_hausdorff_torch(pred, gt))

    if not cd_list:
        return {'chamfer': float('nan'), 'mod_h': float('nan'), 'n': 0}
    return {
        'chamfer': float(np.mean(cd_list)),
        'mod_h': float(np.mean(mh_list)),
        'n': len(cd_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-prototypes', action='store_true',
                        help='Fit GT prototypes offline (run once)')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=8)
    parser.add_argument('--K', type=int, default=96, help='Number of Gaussian queries')
    parser.add_argument('--N-beam', type=int, default=32, help='Beamspace bins')
    parser.add_argument('--log-dir', default='logs/v2_gaussian_radar')
    parser.add_argument('--processed-dir', default='v2/data/processed')
    args = parser.parse_args()

    if args.fit_prototypes:
        print('Fitting GT prototypes...', flush=True)
        fit_prototypes(args.processed_dir, K=K_PROTOTYPES)
        return

    if not args.train:
        parser.print_help()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    model = GaussianRadarModel(
        N_beam=args.N_beam, T=args.window_size,
        K=args.K, hidden_ch=128,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}', flush=True)

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
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_mh = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, comp_str = train_epoch(model, loaders['train'], optimizer, device, epoch)
        scheduler.step()

        # Eval on val
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

    # Final test eval with threshold sweep
    print('\nTest evaluation with threshold sweep:', flush=True)
    best_ckpt = torch.load(os.path.join(args.log_dir, 'best.pt'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    for thresh in [0.0, 0.3, 0.5, 0.7]:
        test_metrics = eval_points(model, loaders['test'], device, threshold=thresh)
        print(f'  thresh={thresh:.1f}: Chamfer {test_metrics["chamfer"]:.4f}, '
              f'mod-H {test_metrics["mod_h"]:.4f}, N={test_metrics["n"]}', flush=True)

    print(f'\nBaseline reference: Chamfer 0.295, mod-H 0.189', flush=True)
    print(f'v2 point decoder:  Chamfer 0.295, mod-H 0.429', flush=True)


if __name__ == '__main__':
    main()
