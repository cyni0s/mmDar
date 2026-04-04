"""Consolidated training script for physics-first Gaussian radar model.

Trains PhysicsGaussianModel: ClassicalFFT → 2D encoder → 1D deep encoder →
DETR decoder → Gaussian set output. Hungarian NLL loss (mod-H aligned).

Supports three data splits via --split:
  original  — 21 train / 4 val / 19 test  (split.py)
  v2        — 17 train / 8 val / 19 test  (split_v2.py)
  mixed     — 25 train / 6 val / 13 test  (split_mixed.py, includes high-ID)

Pre-requisite: fit GT prototypes offline (run with --fit-prototypes first).

Usage:
  docker compose run --rm mmdar python3 train/train.py --fit-prototypes
  docker compose run --rm mmdar python3 train/train.py --train
  docker compose run --rm mmdar python3 train/train.py --train --split mixed --augment

References:
  - model/physics_frontend.py: PhysicsGaussianModel architecture
  - train/loss_gaussian.py: Hungarian NLL + coverage + cardinality + repulsion
"""

import sys
import os
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.physics_frontend import PhysicsGaussianModel
from train.loss_gaussian import gaussian_composite_loss
from eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch

SEED = 42
K_PROTOTYPES = 64


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Split resolution
# ---------------------------------------------------------------------------

def get_split(name: str):
    """Return (TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, ALL_TRAJS) for named split."""
    if name == "original":
        from data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, ALL_TRAJS
    elif name == "v2":
        from data.split_v2 import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
        ALL_TRAJS = sorted(set(TRAIN_TRAJS + VAL_TRAJS + TEST_TRAJS))
    elif name == "mixed":
        from data.split_mixed import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS
        ALL_TRAJS = sorted(set(TRAIN_TRAJS + VAL_TRAJS + TEST_TRAJS))
    else:
        raise ValueError(f"Unknown split: {name!r}. Use original/v2/mixed.")
    return TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, ALL_TRAJS


# ---------------------------------------------------------------------------
# Fit GT prototypes offline (K-Means on lidar XY)
# ---------------------------------------------------------------------------

def fit_prototypes(processed_dir: str, K: int = K_PROTOTYPES,
                   all_trajs: list[int] | None = None):
    """Fit K-Means prototypes to each lidar frame. Save as proto_{tid}.pt.

    Args:
        processed_dir: directory with lidar_{tid}.pt files
        K: number of prototypes per frame
        all_trajs: trajectory IDs to process (default: all from original split)
    """
    from sklearn.cluster import MiniBatchKMeans

    if all_trajs is None:
        from data.split import ALL_TRAJS
        all_trajs = ALL_TRAJS

    for tid in all_trajs:
        lidar_path = os.path.join(processed_dir, f"lidar_{tid}.pt")
        if not os.path.exists(lidar_path):
            continue

        lidar = torch.load(lidar_path, weights_only=True).numpy()  # (N, 8192, 3)
        N = lidar.shape[0]
        protos = np.zeros((N, K, 2), dtype=np.float32)

        for i in range(N):
            xy = lidar[i, :, :2].astype(np.float64)
            mask = (xy[:, 0] > 0) & (xy[:, 0] <= 10.8) & (np.abs(xy[:, 1]) <= 10.8)
            xy = xy[mask]
            if len(xy) < 2:
                continue
            n_clusters = min(K, len(xy))
            km = MiniBatchKMeans(
                n_clusters=n_clusters, n_init=1, random_state=0,
                batch_size=512, max_iter=30,
            )
            km.fit(xy)
            centers = km.cluster_centers_
            if len(centers) < K:
                n_reps = K // len(centers) + 1
                centers = np.tile(centers, (n_reps, 1))[:K]
            protos[i] = centers.astype(np.float32)

        out_path = os.path.join(processed_dir, f"proto_{tid}.pt")
        torch.save(torch.from_numpy(protos), out_path)
        print(f"Traj {tid}: {N} frames -> {out_path}", flush=True)

    print("Prototype fitting done.", flush=True)


def fit_prototypes_fps(processed_dir: str, K: int = 96,
                       all_trajs: list[int] | None = None):
    """Pick K GT points per frame via farthest-point sampling (real geometry, no clustering).

    Saves as proto_fps_{tid}.pt so both K-Means and FPS protos can coexist.

    Args:
        processed_dir: directory with lidar_{tid}.pt files
        K: number of prototype points per frame
        all_trajs: trajectory IDs to process
    """
    if all_trajs is None:
        from data.split import ALL_TRAJS
        all_trajs = ALL_TRAJS

    for tid in all_trajs:
        lidar_path = os.path.join(processed_dir, f"lidar_{tid}.pt")
        if not os.path.exists(lidar_path):
            continue

        lidar = torch.load(lidar_path, weights_only=True).numpy()  # (N, 8192, 3)
        N = lidar.shape[0]
        protos = np.zeros((N, K, 2), dtype=np.float32)

        for i in range(N):
            xy = lidar[i, :, :2].astype(np.float64)
            mask = (xy[:, 0] > 0) & (xy[:, 0] <= 10.8) & (np.abs(xy[:, 1]) <= 10.8)
            xy = xy[mask]
            if len(xy) < 2:
                continue
            if len(xy) <= K:
                n_reps = K // len(xy) + 1
                protos[i] = np.tile(xy, (n_reps, 1))[:K].astype(np.float32)
                continue

            # Farthest point sampling
            selected = np.zeros(K, dtype=np.int64)
            selected[0] = 0  # start from first point
            dists = np.full(len(xy), np.inf)
            for j in range(1, K):
                d = np.sum((xy - xy[selected[j - 1]]) ** 2, axis=1)
                dists = np.minimum(dists, d)
                selected[j] = np.argmax(dists)
            protos[i] = xy[selected].astype(np.float32)

        out_path = os.path.join(processed_dir, f"proto_fps_{tid}.pt")
        torch.save(torch.from_numpy(protos), out_path)
        print(f"Traj {tid}: {N} frames -> {out_path} (FPS K={K})", flush=True)

    print("FPS prototype fitting done.", flush=True)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class GaussianDataset(Dataset):
    """Windowed radar IQ + GT prototypes + full lidar (no augmentation)."""

    def __init__(self, traj_id: int, processed_dir: str, window_size: int = 8):
        self.window_size = window_size
        self.radar = torch.load(
            os.path.join(processed_dir, f"radar_{traj_id}.pt"), weights_only=True)
        self.lidar = torch.load(
            os.path.join(processed_dir, f"lidar_{traj_id}.pt"), weights_only=True)
        self.protos = torch.load(
            os.path.join(processed_dir, f"proto_{traj_id}.pt"), weights_only=True)
        self.n_frames = self.radar.shape[0]

    def __len__(self):
        return max(0, self.n_frames - self.window_size + 1)

    def __getitem__(self, idx):
        end = idx + self.window_size
        return (
            self.radar[idx:end],       # (W, 8, 512) complex
            self.lidar[end - 1],       # (8192, 3) float
            self.protos[end - 1],      # (K, 2) float
        )


class AugmentedGaussianDataset(Dataset):
    """Windowed radar IQ + GT prototypes + full lidar with online augmentation."""

    def __init__(self, traj_id: int, processed_dir: str, window_size: int = 41,
                 augment: bool = False, proto_method: str = "kmeans"):
        self.window_size = window_size
        self.augment = augment
        self.traj_id = traj_id
        self.radar = torch.load(
            os.path.join(processed_dir, f"radar_{traj_id}.pt"), weights_only=True)
        self.lidar = torch.load(
            os.path.join(processed_dir, f"lidar_{traj_id}.pt"), weights_only=True)
        proto_file = f"proto_fps_{traj_id}.pt" if proto_method == "fps" else f"proto_{traj_id}.pt"
        self.protos = torch.load(
            os.path.join(processed_dir, proto_file), weights_only=True)
        self.n_frames = self.radar.shape[0]

    def __len__(self):
        return max(0, self.n_frames - self.window_size + 1)

    def __getitem__(self, idx):
        end = idx + self.window_size
        radar = self.radar[idx:end]
        lidar = self.lidar[end - 1]
        protos = self.protos[end - 1]

        if self.augment:
            from data.augment import augment_sample
            radar, lidar, protos = augment_sample(radar, lidar, protos)

        return radar, lidar, protos


def build_dataloaders(processed_dir: str, train_trajs: list[int],
                      val_trajs: list[int], test_trajs: list[int],
                      window_size: int = 41, batch_size: int = 4,
                      num_workers: int = 4, augment: bool = False,
                      proto_method: str = "kmeans") -> dict:
    """Build train/val/test DataLoaders for Gaussian training.

    Args:
        processed_dir: directory with radar/lidar/proto .pt files
        train_trajs, val_trajs, test_trajs: trajectory ID lists
        window_size: number of consecutive radar frames per sample
        batch_size: DataLoader batch size
        num_workers: DataLoader worker processes
        augment: apply online augmentation to training data
        proto_method: "kmeans" or "fps" — which prototype files to load

    Returns:
        dict with "train", "val", "test" DataLoaders (or None if no data)
    """
    proto_prefix = "proto_fps_" if proto_method == "fps" else "proto_"
    split_configs = {
        "train": (train_trajs, True, augment),
        "val":   (val_trajs,   False, False),
        "test":  (test_trajs,  False, False),
    }
    loaders = {}
    for split, (trajs, shuffle, aug) in split_configs.items():
        datasets = []
        for tid in trajs:
            if os.path.exists(os.path.join(processed_dir, f"{proto_prefix}{tid}.pt")):
                datasets.append(AugmentedGaussianDataset(
                    tid, processed_dir, window_size, augment=aug,
                    proto_method=proto_method))
        if datasets:
            loaders[split] = DataLoader(
                ConcatDataset(datasets), batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        else:
            loaders[split] = None
    return loaders


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch, grad_clip=1.0,
                n_proto=K_PROTOTYPES, loss_kwargs=None):
    """Train one epoch. Returns (avg_loss, component_string)."""
    model.train()
    total_loss = 0
    loss_components = {}
    n_batches = 0
    lkw = loss_kwargs or {}

    for radar, lidar, protos in loader:
        radar = radar.to(device)
        lidar_xy = lidar[:, :, :2].to(device)
        protos = protos.to(device)
        n_gt = torch.full((radar.shape[0],), n_proto, device=device)

        out = model(radar)
        losses = gaussian_composite_loss(out, protos, lidar_xy, n_gt, epoch=epoch, **lkw)

        optimizer.zero_grad()
        losses["total"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses["total"].item()
        for k, v in losses.items():
            if k != "total":
                loss_components[k] = loss_components.get(k, 0) + v.item()
        n_batches += 1

    avg = total_loss / max(n_batches, 1)
    comp_str = " ".join(f"{k}={v / n_batches:.3f}" for k, v in loss_components.items())
    return avg, comp_str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_points(model, loader, device, threshold=0.0):
    """Frame-level Chamfer + mod-H on predicted Gaussian centers."""
    model.eval()
    cd_list, mh_list = [], []
    with torch.no_grad():
        for radar, lidar, protos in loader:
            radar = radar.to(device)
            lidar_xy = lidar[:, :, :2].to(device)
            point_clouds = model.predict_points(radar, threshold=threshold)
            for b in range(len(point_clouds)):
                pred = point_clouds[b]
                gt = lidar_xy[b]
                if pred.shape[0] < 2:
                    continue
                cd_list.append(_chamfer_torch(pred, gt))
                mh_list.append(_mod_hausdorff_torch(pred, gt))
    if not cd_list:
        return {"chamfer": float("nan"), "mod_h": float("nan"), "n": 0}
    return {
        "chamfer": float(np.mean(cd_list)),
        "mod_h": float(np.mean(mh_list)),
        "n": len(cd_list),
    }


def eval_per_trajectory(model, processed_dir, traj_ids, device,
                        window_size=41, threshold=0.3, proto_method="kmeans"):
    """Per-trajectory evaluation. Returns trajectory-level median mod-H."""
    model.eval()
    traj_modh = []
    traj_chamfer = []
    proto_prefix = "proto_fps_" if proto_method == "fps" else "proto_"

    with torch.no_grad():
        for tid in traj_ids:
            if not os.path.exists(os.path.join(processed_dir, f"{proto_prefix}{tid}.pt")):
                continue
            ds = AugmentedGaussianDataset(tid, processed_dir, window_size, augment=False,
                                           proto_method=proto_method)
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            cd_list, mh_list = [], []
            for radar, lidar, protos in loader:
                points = model.predict_points(radar.to(device), threshold=threshold)
                gt_xy = lidar[0, :, :2].to(device)
                pred = points[0]
                if pred.shape[0] < 2:
                    cd_list.append(10.8)
                    mh_list.append(10.8)
                    continue
                cd_list.append(_chamfer_torch(pred, gt_xy))
                mh_list.append(_mod_hausdorff_torch(pred, gt_xy))

            if cd_list:
                traj_modh.append(float(np.mean(mh_list)))
                traj_chamfer.append(float(np.mean(cd_list)))

    if not traj_modh:
        return {"mod_h_traj_median": float("nan"), "chamfer_traj_median": float("nan"),
                "n_trajs": 0}

    return {
        "mod_h_traj_median": float(np.median(traj_modh)),
        "chamfer_traj_median": float(np.median(traj_chamfer)),
        "mod_h_traj_mean": float(np.mean(traj_modh)),
        "mod_h_traj_max": float(np.max(traj_modh)),
        "n_trajs": len(traj_modh),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train physics-first Gaussian radar model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fit-prototypes", action="store_true",
                        help="Fit GT prototypes offline (run once)")
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--split", type=str, default="v2",
                        choices=["original", "v2", "mixed"],
                        help="Data split: original (21/4/19), v2 (17/8/19), mixed (25/6/13)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply online augmentation (flip, noise, temporal mask)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument("--K", type=int, default=96,
                        help="Number of Gaussian queries in DETR decoder")
    parser.add_argument("--N-az", type=int, default=64,
                        help="Azimuth bins in FFT frontend")
    parser.add_argument("--log-dir", default="logs/v2_gaussian")
    parser.add_argument("--processed-dir", default="data/processed")
    # Loss tuning
    parser.add_argument("--sigma-r-prior", type=float, default=0.3,
                        help="Prior target for range uncertainty (metres)")
    parser.add_argument("--sigma-perp-prior", type=float, default=0.3,
                        help="Prior target for perpendicular uncertainty (metres)")
    parser.add_argument("--huber-range-weight", type=float, default=0.1,
                        help="Weight for Huber range loss (0 = disabled)")
    # Prototype method
    parser.add_argument("--proto-method", type=str, default="kmeans",
                        choices=["kmeans", "fps"],
                        help="GT prototype method: kmeans (cluster centers) or fps (real points)")
    parser.add_argument("--K-proto", type=int, default=64,
                        help="Number of GT prototypes per frame (for --fit-prototypes)")
    args = parser.parse_args()

    set_seed(SEED)

    TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS, ALL_TRAJS = get_split(args.split)

    if args.fit_prototypes:
        print("Fitting GT prototypes...", flush=True)
        if args.proto_method == "fps":
            fit_prototypes_fps(args.processed_dir, K=args.K_proto, all_trajs=ALL_TRAJS)
        else:
            fit_prototypes(args.processed_dir, K=args.K_proto, all_trajs=ALL_TRAJS)
        return

    if not args.train:
        parser.print_help()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = PhysicsGaussianModel(
        N_az=args.N_az, T=args.window_size, K=args.K,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    print(f"Split ({args.split}): {len(TRAIN_TRAJS)} train, "
          f"{len(VAL_TRAJS)} val, {len(TEST_TRAJS)} test", flush=True)
    if args.augment:
        print("Augmentation: flip(0.5) + noise(0.5, SNR 15-25dB) + mask(0.3, 2-6 frames)",
              flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loaders = build_dataloaders(
        args.processed_dir, TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS,
        window_size=args.window_size, batch_size=args.batch_size,
        num_workers=4, augment=args.augment,
        proto_method=args.proto_method,
    )
    print(f"Train samples: {len(loaders['train'].dataset)}", flush=True)
    if loaders["val"]:
        print(f"Val samples: {len(loaders['val'].dataset)}", flush=True)
    print(f"Proto method: {args.proto_method}, sigma_r_prior: {args.sigma_r_prior}, "
          f"sigma_perp_prior: {args.sigma_perp_prior}, huber_range: {args.huber_range_weight}",
          flush=True)

    # Loss kwargs for configurable priors and Huber range
    loss_kwargs = {
        "sigma_r_prior": args.sigma_r_prior,
        "sigma_perp_prior": args.sigma_perp_prior,
        "w_huber_range": args.huber_range_weight,
    }
    # Determine n_proto from proto files
    n_proto = args.K_proto if args.proto_method == "fps" else K_PROTOTYPES

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config["n_params"] = n_params
    config["K_prototypes"] = n_proto
    config["train_trajs"] = TRAIN_TRAJS
    config["val_trajs"] = VAL_TRAJS
    config["test_trajs"] = TEST_TRAJS
    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_val_mh = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, comp_str = train_epoch(
            model, loaders["train"], optimizer, device, epoch,
            n_proto=n_proto, loss_kwargs=loss_kwargs)
        scheduler.step()

        # Per-trajectory val evaluation every 5 epochs (expensive)
        if loaders["val"] and (epoch % 5 == 0 or epoch == args.epochs - 1):
            val_metrics = eval_per_trajectory(
                model, args.processed_dir, VAL_TRAJS, device,
                args.window_size, threshold=0.3,
                proto_method=args.proto_method)
            val_mh = val_metrics["mod_h_traj_median"]
        else:
            val_mh = float("nan")
            val_metrics = {}

        elapsed = time.time() - t0
        val_str = f"val_mh_traj {val_mh:.4f}" if not np.isnan(val_mh) else "val skip"
        print(f"Ep {epoch:3d} | loss {train_loss:.4f} | {comp_str} | "
              f"{val_str} | {elapsed:.0f}s", flush=True)

        if val_mh < best_val_mh:
            best_val_mh = val_mh
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, os.path.join(args.log_dir, "best.pt"))

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics if not np.isnan(val_mh) else {},
            }, os.path.join(args.log_dir, f"epoch_{epoch:03d}.pt"))

    # Final test evaluation — per-trajectory with threshold sweep
    print("\n=== TEST EVALUATION ===", flush=True)
    best_ckpt = torch.load(
        os.path.join(args.log_dir, "best.pt"),
        map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Best epoch: {best_ckpt['epoch']}", flush=True)

    for thresh in [0.0, 0.3, 0.5]:
        test_m = eval_per_trajectory(
            model, args.processed_dir, TEST_TRAJS, device,
            args.window_size, threshold=thresh,
            proto_method=args.proto_method)
        print(f"  thresh={thresh:.1f}: mod-H traj_median={test_m['mod_h_traj_median']:.4f}, "
              f"traj_mean={test_m['mod_h_traj_mean']:.4f}, "
              f"traj_max={test_m['mod_h_traj_max']:.4f}, "
              f"chamfer_median={test_m['chamfer_traj_median']:.4f}", flush=True)

    # Low-ID vs high-ID breakdown (useful for mixed split)
    low_id_test = [t for t in TEST_TRAJS if t < 200]
    high_id_test = [t for t in TEST_TRAJS if t >= 200]
    if high_id_test:
        for label, trajs in [("Low-ID test", low_id_test),
                              ("High-ID test", high_id_test)]:
            m = eval_per_trajectory(
                model, args.processed_dir, trajs, device,
                args.window_size, threshold=0.3,
                proto_method=args.proto_method)
            print(f"  {label:>14}: mod-H traj_median={m['mod_h_traj_median']:.4f}, "
                  f"chamfer={m['chamfer_traj_median']:.4f}", flush=True)

    print(f"\nBaseline reference: Chamfer 0.295, mod-H 0.189 (test-selected)", flush=True)
    print(f"Honest baseline:   mod-H 0.296 (val-selected)", flush=True)


if __name__ == "__main__":
    main()
