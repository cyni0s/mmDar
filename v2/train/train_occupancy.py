"""End-to-end training script for mmDar v2 polar occupancy model.

Trains OccupancyModel (FFT/LISTA beamformer -> Channelizer -> DilatedResHead)
on complex IQ radar measurements to produce polar occupancy grids (256 x 512).

Training strategy:
    - Focal BCE + Dice loss for class-imbalanced occupancy supervision
    - AdamW optimizer with warmup + cosine LR decay
    - Early stopping on validation Chamfer distance (occupancy -> point cloud)
    - Checkpoints saved every checkpoint_every epochs + best model

Monitoring:
    - TensorBoard: train/total, train/focal_bce, train/dice
    - TensorBoard: val/chamfer, val/mod_hausdorff
    - TensorBoard: lr (learning rate)

Usage:
    python3 v2/train/train_occupancy.py [--batch-size 12] [--lr 7e-5] [--epochs 50]

References:
    - Plan: Task 7 occupancy training script spec
"""

import argparse
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from v2.model.occupancy import OccupancyModel
from v2.train.loss_occupancy import occupancy_loss
from v2.eval.occupancy_eval import evaluate_occupancy_epoch
from v2.data.dataset import build_occupancy_dataloaders


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "batch_size": 12,
    "lr": 7e-5,
    "num_epochs": 50,
    "early_stop_patience": 10,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "warmup_epochs": 3,
    "checkpoint_every": 10,
    "log_dir": "logs/v2_occ",
    "num_workers": 4,
    "processed_dir": "v2/data/processed",
    "model_type": "fft",
    "mid_ch": 32,
    "n_blocks": 4,
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict | None = None) -> dict:
    """Run one full occupancy training job.

    Args:
        config: Dict of training hyperparameters. Any key not provided falls
                back to DEFAULT_CONFIG.

    Returns:
        dict with keys:
            'best_val_chamfer': float -- best validation Chamfer distance
            'best_epoch':       int   -- epoch at which best Chamfer was achieved
            'log_dir':          str   -- path to checkpoint/tensorboard directory
    """
    # Merge user config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_occ] device: {device}")
    print(f"[train_occ] config: {cfg}")

    # --- Directories ---
    log_dir = cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    best_ckpt_path = os.path.join(log_dir, "best.pt")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Data ---
    processed_dir = cfg["processed_dir"]
    loaders = build_occupancy_dataloaders(
        processed_dir=processed_dir,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(
        f"[train_occ] train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}"
    )

    # --- Model ---
    model_type = cfg.get("model_type", "fft")
    model = OccupancyModel(
        beamformer=model_type,
        N_az=256,
        mid_ch=cfg["mid_ch"],
        n_blocks=cfg["n_blocks"],
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_occ] OccupancyModel (beamformer={model_type}), "
          f"{n_params:,} trainable params")

    # --- Optimizer (simple AdamW, no special param groups) ---
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
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

    # --- Training state ---
    best_val_chamfer = float("inf")
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- Train epoch ---
        model.train()
        epoch_losses = {k: 0.0 for k in ("total", "focal_bce", "dice")}
        n_batches = 0

        for batch_idx, (radar, lidar, occ_label, norm) in enumerate(train_loader):
            radar = radar.to(device)           # (B, 8, 512) complex64
            occ_target = occ_label.unsqueeze(1).to(device)  # (B, 1, 256, 512)

            # Forward
            logits = model(radar)              # (B, 1, 256, 512)

            # Loss
            losses = occupancy_loss(logits, occ_target)

            optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                val = losses[key]
                epoch_losses[key] += val.item() if hasattr(val, "item") else float(val)
            n_batches += 1
            global_step += 1

        # LR scheduler step (once per epoch)
        scheduler.step()

        # Log epoch-mean training losses
        for key in epoch_losses:
            writer.add_scalar(
                f"train/{key}", epoch_losses[key] / max(1, n_batches), epoch
            )

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        # --- Validation ---
        model.eval()
        val_metrics = evaluate_occupancy_epoch(model, val_loader, device)
        val_chamfer = val_metrics["chamfer"]
        val_mhd = val_metrics["mod_hausdorff"]

        writer.add_scalar("val/chamfer", val_chamfer, epoch)
        writer.add_scalar("val/mod_hausdorff", val_mhd, epoch)

        epoch_elapsed = time.time() - epoch_start
        print(
            f"[train_occ] Epoch {epoch:03d} | "
            f"loss={epoch_losses['total'] / max(1, n_batches):.4f} | "
            f"val_chamfer={val_chamfer:.4f} | val_mH={val_mhd:.4f} | "
            f"lr={current_lr:.2e} | {epoch_elapsed:.1f}s"
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
                    "val_mod_hausdorff": val_mhd,
                    "config": cfg,
                },
                best_ckpt_path,
            )
            print(
                f"[train_occ]   => New best val_chamfer={val_chamfer:.4f} "
                f"at epoch {epoch}. Checkpoint saved."
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(
                    f"[train_occ] Early stopping at epoch {epoch} "
                    f"(patience exhausted)"
                )
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
                    "val_mod_hausdorff": val_mhd,
                    "config": cfg,
                },
                periodic_path,
            )
            print(f"[train_occ]   => Periodic checkpoint saved: {periodic_path}")

    total_time = time.time() - start_time
    writer.close()

    print(
        f"[train_occ] Done. Best val_chamfer={best_val_chamfer:.4f} at "
        f"epoch {best_epoch}. Total time: {total_time / 60:.1f} min"
    )

    return {
        "best_val_chamfer": best_val_chamfer,
        "best_epoch": best_epoch,
        "log_dir": log_dir,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train mmDar v2 OccupancyModel (polar occupancy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
        dest="batch_size", help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_CONFIG["lr"],
        help="Initial learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"],
        dest="num_epochs", help="Maximum number of training epochs"
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
        "--model-type", type=str, default=DEFAULT_CONFIG["model_type"],
        choices=["fft", "lista"],
        dest="model_type",
        help="Beamformer type: 'fft' (non-learnable) or 'lista' (learnable)"
    )
    parser.add_argument(
        "--mid-ch", type=int, default=DEFAULT_CONFIG["mid_ch"],
        dest="mid_ch", help="DilatedResHead internal channel width"
    )
    parser.add_argument(
        "--n-blocks", type=int, default=DEFAULT_CONFIG["n_blocks"],
        dest="n_blocks", help="Number of DilatedResBlocks"
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_CONFIG["early_stop_patience"],
        dest="early_stop_patience", help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CONFIG["checkpoint_every"],
        dest="checkpoint_every", help="Save checkpoint every N epochs"
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
        "--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
        dest="num_workers", help="DataLoader worker processes"
    )

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "log_dir": args.log_dir,
        "processed_dir": args.processed_dir,
        "model_type": args.model_type,
        "mid_ch": args.mid_ch,
        "n_blocks": args.n_blocks,
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_every": args.checkpoint_every,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
    }

    train(config)
