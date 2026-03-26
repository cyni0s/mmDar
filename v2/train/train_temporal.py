"""Temporal training script for mmDar v2.

Trains TemporalMagPhaseFusion (MagnitudePhaseFusion + cross-attention) on
windowed radar sequences to produce lidar-quality point clouds. Builds on
the single-frame train.py with temporal-specific features:

Training strategy:
    - Loads pretrained single-frame weights via model.load_single_frame_weights()
    - Staged freeze: epochs 0..freeze_backbone_epochs-1 freeze beamformer+bridge+decoder,
      training only the temporal cross-attention block
    - After freeze, joint fine-tuning with backbone at lower LR (backbone_lr_factor)
    - Variable window: each epoch randomly picks N from window_sizes list
    - During freeze stage, N=1 is excluded (temporal block needs multi-frame input)
    - After unfreeze, N=1 is included to maintain single-frame identity
    - Validation runs at each eval_window_size separately for Pareto analysis
    - Early stopping on middle eval window size (e.g. N=5 from [1,3,5,8])

Loss:
    Same composite_loss as train.py (Chamfer + DCD + coverage + confidence)

Monitoring:
    - TensorBoard: train/total, train/chamfer, train/dcd, train/coverage, train/confidence
    - TensorBoard: val/chamfer_N{k}, val/mod_hausdorff_N{k} for each eval window size
    - TensorBoard: lr/temporal, lr/backbone
    - TensorBoard: point_cloud/std_x, point_cloud/std_y (collapse detection)
    - TensorBoard: train/window_size (current epoch's N)

Usage:
    python3 v2/train/train_temporal.py --pretrained-checkpoint logs/v2_mag_phase/best.pt

References:
    - v2/train/train.py: single-frame training (same loss, eval)
    - v2/model/temporal.py: TemporalMagPhaseFusion architecture
    - v2/data/windowed_dataset.py: WindowedTrajectoryDataset
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from v2.model.temporal import TemporalMagPhaseFusion
from v2.data.windowed_dataset import build_windowed_dataloaders
from v2.train.loss import composite_loss
from v2.eval.eval_adapter import _chamfer_torch, _mod_hausdorff_torch


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "batch_size": 12,
    "lr": 7e-5,
    "backbone_lr_factor": 0.1,
    "num_epochs": 50,
    "freeze_backbone_epochs": 5,
    "early_stop_patience": 10,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_epochs": 3,
    "checkpoint_every": 10,
    "log_dir": "logs/v2_temporal",
    "num_workers": 4,
    "processed_dir": "v2/data/processed",
    "pretrained_checkpoint": None,
    "window_sizes": [3, 5, 8],
    "eval_window_sizes": [1, 3, 5, 8],
    "max_window": 8,
}


# ---------------------------------------------------------------------------
# Evaluation at a specific window size
# ---------------------------------------------------------------------------

def evaluate_at_window(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_N: int,
) -> dict:
    """Evaluate model at a specific window size.

    For each batch, crops radar_window to the last eval_N frames and runs
    inference. Computes Chamfer and modified Hausdorff on XY-only 2D distances.

    Args:
        model:      TemporalMagPhaseFusion in eval mode
        dataloader: DataLoader yielding (radar_window, lidar, norm) tuples
                    where radar_window is (B, W, 8, 512) with W >= eval_N
        device:     Compute device
        eval_N:     Number of frames to use (crops to last eval_N)

    Returns:
        dict with 'chamfer', 'mod_hausdorff', 'n_samples'
    """
    model.eval()
    chamfer_accum = 0.0
    hausdorff_accum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            radar_window, lidar, _norm = batch
            radar_window = radar_window.to(device)  # (B, W, 8, 512)
            lidar = lidar.to(device)                # (B, 8192, 3)

            # Crop to last eval_N frames
            radar_crop = radar_window[:, -eval_N:, :, :]  # (B, eval_N, 8, 512)

            pred_pts, _conf = model(radar_crop)  # (B, 8192, 3)

            B = pred_pts.shape[0]
            for i in range(B):
                chamfer_accum += _chamfer_torch(pred_pts[i], lidar[i])
                hausdorff_accum += _mod_hausdorff_torch(pred_pts[i], lidar[i])
            n_samples += B

    if n_samples == 0:
        return {"chamfer": float("nan"), "mod_hausdorff": float("nan"), "n_samples": 0}

    return {
        "chamfer": chamfer_accum / n_samples,
        "mod_hausdorff": hausdorff_accum / n_samples,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict | None = None) -> dict:
    """Run one full temporal training job.

    Args:
        config: Dict of training hyperparameters. Any key not provided falls
                back to DEFAULT_CONFIG.

    Returns:
        dict with keys:
            'best_val_chamfer': float -- best validation Chamfer distance
            'best_epoch':       int   -- epoch at which best Chamfer was achieved
            'log_dir':          str   -- path to checkpoint/tensorboard directory
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_temporal] device: {device}")
    print(f"[train_temporal] config: {cfg}")

    # --- Directories ---
    log_dir = cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    best_ckpt_path = os.path.join(log_dir, "best.pt")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Data ---
    # Build dataset with max_window; we crop per-epoch inside the training loop
    processed_dir = cfg["processed_dir"]
    max_window = cfg["max_window"]
    loaders = build_windowed_dataloaders(
        processed_dir=processed_dir,
        window_size=max_window,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(
        f"[train_temporal] train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}, max_window: {max_window}"
    )

    # --- Model ---
    model = TemporalMagPhaseFusion(N_az=256, bridge_out_ch=128, max_lag=max_window)
    print("[train_temporal] Using TemporalMagPhaseFusion")

    # Load pretrained single-frame weights (beamformer + bridge + decoder)
    pretrained = cfg.get("pretrained_checkpoint")
    if pretrained:
        print(f"[train_temporal] Loading pretrained weights from: {pretrained}")
        model.load_single_frame_weights(pretrained)
    else:
        print("[train_temporal] WARNING: No pretrained checkpoint — training from scratch")

    model = model.to(device)

    # --- Initial freeze: only temporal params trainable ---
    freeze_backbone_epochs = cfg["freeze_backbone_epochs"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    backbone_lr_factor = cfg["backbone_lr_factor"]

    def _get_param_groups(mdl, lr_val, wd_val, freeze_backbone: bool):
        """Build param groups. When freeze_backbone=True, only temporal params."""
        temporal_params = []
        backbone_params = []
        for name, p in mdl.named_parameters():
            if name.startswith("temporal."):
                temporal_params.append(p)
            else:
                backbone_params.append(p)

        if freeze_backbone:
            # Freeze backbone
            for p in backbone_params:
                p.requires_grad = False
            for p in temporal_params:
                p.requires_grad = True
            return [{"params": temporal_params, "lr": lr_val, "weight_decay": wd_val}]
        else:
            # Unfreeze all, backbone at lower LR
            for p in backbone_params:
                p.requires_grad = True
            for p in temporal_params:
                p.requires_grad = True
            groups = []
            if temporal_params:
                groups.append({"params": temporal_params, "lr": lr_val, "weight_decay": wd_val})
            if backbone_params:
                groups.append({"params": backbone_params, "lr": lr_val * backbone_lr_factor, "weight_decay": wd_val})
            return groups

    backbone_frozen = freeze_backbone_epochs > 0
    param_groups = _get_param_groups(model, lr, weight_decay, freeze_backbone=backbone_frozen)
    optimizer = torch.optim.AdamW(param_groups)

    if backbone_frozen:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[train_temporal] Backbone frozen for {freeze_backbone_epochs} epochs. "
            f"Trainable: {n_trainable:,} / {n_total:,} params"
        )

    # --- LR schedule: warmup then cosine ---
    num_epochs = cfg["num_epochs"]
    warmup_epochs = cfg["warmup_epochs"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs),
        eta_min=1e-7,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # --- Window size config ---
    window_sizes = list(cfg["window_sizes"])
    eval_window_sizes = list(cfg["eval_window_sizes"])
    # Early stopping on middle eval window size
    early_stop_N = eval_window_sizes[len(eval_window_sizes) // 2]
    print(
        f"[train_temporal] Train window sizes: {window_sizes}, "
        f"Eval window sizes: {eval_window_sizes}, "
        f"Early stop on N={early_stop_N}"
    )

    # --- Training state ---
    best_val_chamfer = float("inf")
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- Unfreeze backbone at freeze_backbone_epochs ---
        if backbone_frozen and epoch >= freeze_backbone_epochs:
            backbone_frozen = False
            param_groups = _get_param_groups(model, lr, weight_decay, freeze_backbone=False)
            optimizer = torch.optim.AdamW(param_groups)
            # Reset scheduler for remaining epochs
            remaining = max(1, num_epochs - epoch)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-7
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"[train_temporal] Epoch {epoch}: Unfreezing backbone. "
                f"Trainable: {n_trainable:,} params. "
                f"Backbone LR: {lr * backbone_lr_factor:.2e}"
            )

        # --- Choose window size for this epoch ---
        if backbone_frozen:
            # During freeze, exclude N=1 (temporal block needs multi-frame)
            epoch_window_sizes = [n for n in window_sizes if n > 1]
        else:
            # After unfreeze, include N=1 to maintain identity
            epoch_window_sizes = window_sizes + [1] if 1 not in window_sizes else window_sizes
        current_N = random.choice(epoch_window_sizes)
        writer.add_scalar("train/window_size", current_N, epoch)
        print(f"[train_temporal] Epoch {epoch}: window_size N={current_N}")

        # --- Train epoch ---
        model.train()
        epoch_losses = {k: 0.0 for k in ("total", "chamfer", "dcd", "coverage", "confidence")}
        n_batches = 0

        for batch_idx, (radar_window, lidar, _norm) in enumerate(train_loader):
            radar_window = radar_window.to(device)  # (B, max_window, 8, 512)
            lidar = lidar.to(device)                # (B, 8192, 3)

            # Crop to current epoch's N (take last N frames)
            radar_crop = radar_window[:, -current_N:, :, :]  # (B, N, 8, 512)

            pts, conf = model(radar_crop)

            losses = composite_loss(
                pts,
                lidar,
                conf,
                epoch,
                use_dcd=True,
                use_coverage=True,
                use_confidence=True,
                coverage_threshold=0.25,
            )

            optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                val = losses.get(key)
                if val is not None:
                    epoch_losses[key] += val.item() if hasattr(val, "item") else float(val)
            n_batches += 1
            global_step += 1

        # LR scheduler step
        scheduler.step()

        # Log epoch-mean training losses
        for key in epoch_losses:
            writer.add_scalar(
                f"train/{key}", epoch_losses[key] / max(1, n_batches), epoch
            )

        # Log learning rates per param group
        for gi, pg in enumerate(optimizer.param_groups):
            label = "temporal" if gi == 0 else "backbone"
            writer.add_scalar(f"lr/{label}", pg["lr"], epoch)

        # --- Validation at each eval window size ---
        val_results = {}
        for eval_N in eval_window_sizes:
            if eval_N > max_window:
                continue
            metrics = evaluate_at_window(model, val_loader, device, eval_N)
            val_results[eval_N] = metrics
            writer.add_scalar(f"val/chamfer_N{eval_N}", metrics["chamfer"], epoch)
            writer.add_scalar(f"val/mod_hausdorff_N{eval_N}", metrics["mod_hausdorff"], epoch)

        # Point cloud collapse detection
        _log_point_cloud_stats(model, train_loader, device, writer, epoch, current_N)

        # Use early_stop_N for early stopping metric
        val_chamfer = val_results.get(early_stop_N, {}).get("chamfer", float("inf"))

        epoch_elapsed = time.time() - epoch_start
        val_summary = " | ".join(
            f"N{n}={val_results[n]['chamfer']:.4f}"
            for n in sorted(val_results.keys())
        )
        print(
            f"[train_temporal] Epoch {epoch:03d} | "
            f"loss={epoch_losses['total'] / max(1, n_batches):.4f} | "
            f"val_chamfer: {val_summary} | "
            f"N={current_N} | {epoch_elapsed:.1f}s"
        )

        # --- Early stopping ---
        if val_chamfer < best_val_chamfer:
            best_val_chamfer = val_chamfer
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_chamfer": val_chamfer,
                    "val_results": val_results,
                    "config": cfg,
                },
                best_ckpt_path,
            )
            print(
                f"[train_temporal]   => New best val_chamfer(N={early_stop_N})="
                f"{val_chamfer:.4f} at epoch {epoch}. Checkpoint saved."
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"[train_temporal] Early stopping at epoch {epoch} (patience exhausted)")
                break

        # --- Periodic checkpoint ---
        if (epoch + 1) % cfg["checkpoint_every"] == 0:
            periodic_path = os.path.join(log_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_chamfer": val_chamfer,
                    "val_results": val_results,
                    "config": cfg,
                },
                periodic_path,
            )
            print(f"[train_temporal]   => Periodic checkpoint saved: {periodic_path}")

    total_time = time.time() - start_time
    writer.close()

    print(
        f"[train_temporal] Done. Best val_chamfer(N={early_stop_N})="
        f"{best_val_chamfer:.4f} at epoch {best_epoch}. "
        f"Total time: {total_time / 60:.1f} min"
    )

    return {
        "best_val_chamfer": best_val_chamfer,
        "best_epoch": best_epoch,
        "log_dir": log_dir,
    }


def _log_point_cloud_stats(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    window_size: int,
    max_batches: int = 3,
) -> None:
    """Log point cloud statistics for template-collapse detection.

    Args:
        model:       TemporalMagPhaseFusion in eval mode
        loader:      Training DataLoader (windowed)
        device:      Compute device
        writer:      TensorBoard SummaryWriter
        epoch:       Current epoch
        window_size: Current window size (for cropping)
        max_batches: Number of batches to sample (default 3)
    """
    model.eval()
    std_x_list = []
    std_y_list = []

    with torch.no_grad():
        for i, (radar_window, _lidar, _norm) in enumerate(loader):
            if i >= max_batches:
                break
            radar_window = radar_window.to(device)
            radar_crop = radar_window[:, -window_size:, :, :]
            pts, _ = model(radar_crop)       # (B, 8192, 3)
            pts_flat = pts.view(-1, 3)       # (B*8192, 3)
            std_x_list.append(pts_flat[:, 0].std().item())
            std_y_list.append(pts_flat[:, 1].std().item())

    if std_x_list:
        writer.add_scalar("point_cloud/std_x", sum(std_x_list) / len(std_x_list), epoch)
        writer.add_scalar("point_cloud/std_y", sum(std_y_list) / len(std_y_list), epoch)

    model.train()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train mmDar v2 TemporalMagPhaseFusion (multi-frame)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pretrained-checkpoint", type=str, default=None,
        dest="pretrained_checkpoint",
        help="Path to single-frame MagnitudePhaseFusion best.pt (required for pretrained init)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
        dest="batch_size", help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_CONFIG["lr"],
        help="Initial learning rate (temporal params)"
    )
    parser.add_argument(
        "--backbone-lr-factor", type=float, default=DEFAULT_CONFIG["backbone_lr_factor"],
        dest="backbone_lr_factor",
        help="LR multiplier for backbone (beamformer+bridge+decoder) after unfreeze"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
        dest="num_epochs", help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--freeze-backbone-epochs", type=int,
        default=DEFAULT_CONFIG["freeze_backbone_epochs"],
        dest="freeze_backbone_epochs",
        help="Epochs to freeze backbone (beamformer+bridge+decoder), train only temporal"
    )
    parser.add_argument(
        "--log-dir", type=str, default=DEFAULT_CONFIG["log_dir"],
        dest="log_dir", help="Directory for checkpoints and TensorBoard logs"
    )
    parser.add_argument(
        "--processed-dir", type=str, default=DEFAULT_CONFIG["processed_dir"],
        dest="processed_dir", help="Directory with preprocessed .pt tensor files"
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_CONFIG["early_stop_patience"],
        dest="early_stop_patience", help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"],
        dest="weight_decay", help="AdamW weight decay"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_CONFIG["grad_clip"],
        dest="grad_clip", help="Gradient clipping norm (0 to disable)"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CONFIG["checkpoint_every"],
        dest="checkpoint_every", help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
        dest="num_workers", help="DataLoader worker processes"
    )
    parser.add_argument(
        "--window-sizes", type=str, default="3,5,8",
        dest="window_sizes_str",
        help="Comma-separated training window sizes (e.g. '3,5,8')"
    )
    parser.add_argument(
        "--eval-window-sizes", type=str, default="1,3,5,8",
        dest="eval_window_sizes_str",
        help="Comma-separated eval window sizes for Pareto curve (e.g. '1,3,5,8')"
    )
    parser.add_argument(
        "--max-window", type=int, default=DEFAULT_CONFIG["max_window"],
        dest="max_window",
        help="Maximum window size for dataset (builds windows of this size, crops per epoch)"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=DEFAULT_CONFIG["warmup_epochs"],
        dest="warmup_epochs", help="Number of LR warmup epochs"
    )

    args = parser.parse_args()

    # Parse comma-separated window sizes
    window_sizes = [int(x.strip()) for x in args.window_sizes_str.split(",")]
    eval_window_sizes = [int(x.strip()) for x in args.eval_window_sizes_str.split(",")]

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone_lr_factor": args.backbone_lr_factor,
        "num_epochs": args.num_epochs,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "log_dir": args.log_dir,
        "processed_dir": args.processed_dir,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "early_stop_patience": args.early_stop_patience,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "checkpoint_every": args.checkpoint_every,
        "num_workers": args.num_workers,
        "window_sizes": window_sizes,
        "eval_window_sizes": eval_window_sizes,
        "max_window": args.max_window,
        "warmup_epochs": args.warmup_epochs,
    }

    train(config)
