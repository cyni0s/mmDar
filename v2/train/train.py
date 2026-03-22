"""End-to-end training script for mmDar v2.

Trains RadarPointCloudModel (LISTA + Stage2Bridge + PointCloudDecoder) on
complex IQ radar measurements to produce dense lidar-quality 3D point clouds.

Training strategy:
    - Stage 1 (LISTA beamformer) frozen for first freeze_stage1_epochs epochs
    - Decoder + bridge learn basic shape from fixed FFT-like beamformer
    - After freeze_stage1_epochs, joint fine-tuning with cosine LR decay
    - DCD loss annealed in from epoch 5 onward
    - Early stopping on validation Chamfer (patience 10 epochs)
    - Checkpoints saved every checkpoint_every epochs + best model

Monitoring:
    - TensorBoard: train/total, train/chamfer, train/dcd, train/coverage, train/confidence
    - TensorBoard: val/chamfer, val/mod_hausdorff
    - TensorBoard: grad_norm/lista_layer_{k} (per LISTA layer, after Stage 1 unfreeze)
    - TensorBoard: lr (learning rate)
    - TensorBoard: point_cloud/std_x, point_cloud/std_y (collapse detection)

Usage:
    python3 v2/train/train.py [--batch-size 8] [--lr 5e-5] [--epochs 50] [--log-dir logs/v2]

References:
    - Plan 03-02: training script spec
    - CONTEXT.md: locked decisions (freeze_stage1, DCD annealing, early stopping)
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from v2.model import RadarPointCloudModel, MagnitudeBaseline, set_stage1_frozen
from v2.train.loss import composite_loss
from v2.eval.eval_adapter import evaluate_epoch
from v2.data.dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "batch_size": 8,
    "lr": 5e-5,
    "num_epochs": 50,
    "early_stop_patience": 10,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "lr_schedule": "cosine",
    "warmup_epochs": 3,
    "freeze_stage1_epochs": 0,
    "use_dcd_loss": True,
    "use_confidence_loss": True,
    "use_coverage_loss": True,
    "coverage_threshold": 0.25,
    "checkpoint_every": 10,
    "log_dir": "logs/v2",
    "num_workers": 4,
    "processed_dir": "v2/data/processed",
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict | None = None) -> dict:
    """Run one full training job.

    Args:
        config: Dict of training hyperparameters. Any key not provided falls
                back to DEFAULT_CONFIG. Supported keys:
                    batch_size, lr, num_epochs, early_stop_patience, optimizer,
                    weight_decay, grad_clip, lr_schedule, warmup_epochs,
                    freeze_stage1_epochs, use_dcd_loss, use_confidence_loss,
                    use_coverage_loss, coverage_threshold, checkpoint_every,
                    log_dir, num_workers, processed_dir

    Returns:
        dict with keys:
            'best_val_chamfer': float — best validation Chamfer distance achieved
            'best_epoch':       int   — epoch at which best Chamfer was achieved
            'log_dir':          str   — path to checkpoint/tensorboard directory
    """
    # Merge user config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Rename argparse snake_case args that overlap with config keys
    # (argparse dest uses underscores; DEFAULT_CONFIG uses underscores too)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    print(f"[train] config: {cfg}")

    # --- Directories ---
    log_dir = cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    best_ckpt_path = os.path.join(log_dir, "best.pt")
    writer = SummaryWriter(log_dir=log_dir)

    # --- Data ---
    processed_dir = cfg["processed_dir"]
    loaders = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(
        f"[train] train batches: {len(train_loader)}, val batches: {len(val_loader)}"
    )

    # --- Model ---
    model_type = cfg.get("model_type", "cvnn")
    if model_type == "magnitude":
        model = MagnitudeBaseline(N_az=256, bridge_out_ch=128)
        print("[train] Using MagnitudeBaseline (no phase, no learned beamforming)")
    else:
        model = RadarPointCloudModel(K=5, N_az=256, bridge_out_ch=128)
        print("[train] Using RadarPointCloudModel (CVNN + LISTA)")
    model = model.to(device)

    # Freeze Stage 1 initially (decoder-first staged training)
    freeze_epochs = cfg["freeze_stage1_epochs"]
    if freeze_epochs > 0:
        set_stage1_frozen(model, frozen=True)
        print(f"[train] Stage 1 (LISTA) frozen for first {freeze_epochs} epochs")
    else:
        print("[train] Joint training from epoch 0 (no Stage 1 freeze)")

    # --- Optimizer ---
    # Separate parameter groups: threshold params (rho) get smaller LR, no weight decay
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]

    def _build_optimizer(mdl, lr_val, wd_val):
        """Build AdamW with separate LR for threshold (rho) parameters."""
        threshold_params = []
        other_params = []
        for name, p in mdl.named_parameters():
            if not p.requires_grad:
                continue
            if "rho" in name:
                threshold_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr_val, "weight_decay": wd_val})
        if threshold_params:
            param_groups.append({"params": threshold_params, "lr": lr_val * 0.1, "weight_decay": 0.0})
        return torch.optim.AdamW(param_groups)

    optimizer = _build_optimizer(model, lr, weight_decay)

    # --- LR schedule: warmup then cosine ---
    # LinearLR ramps lr from start_factor*lr to lr over warmup_epochs steps.
    # CosineAnnealingLR then decays from lr to 0 over the remaining epochs.
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
    stage1_unfrozen = False

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- Freeze/unfreeze Stage 1 ---
        if not stage1_unfrozen and epoch >= cfg["freeze_stage1_epochs"]:
            set_stage1_frozen(model, frozen=False)
            stage1_unfrozen = True
            if cfg["freeze_stage1_epochs"] > 0:
                print(f"[train] Epoch {epoch}: Unfreezing Stage 1 (LISTA beamformer)")
                # Re-create optimizer with all parameters (including newly unfrozen ones)
                optimizer = _build_optimizer(model, lr, weight_decay)
                # Reset scheduler for the remaining epochs
                remaining = max(1, num_epochs - epoch)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-7
                )

        # --- Train epoch ---
        model.train()
        epoch_losses = {k: 0.0 for k in ("total", "chamfer", "dcd", "coverage", "confidence", "measurement_consistency")}
        n_batches = 0

        # Get steering matrix for measurement consistency loss (CVNN model only)
        has_lista = hasattr(model, "beamformer") and hasattr(model.beamformer, "A")

        for batch_idx, (radar, lidar, _norm_factor) in enumerate(train_loader):
            radar = radar.to(device)    # (B, 8, 512) complex64
            lidar = lidar.to(device)   # (B, 8192, 3) float32

            # Forward with intermediate LISTA output for measurement consistency
            if has_lista and hasattr(model, "forward_with_intermediates"):
                pts, conf, bf_out = model.forward_with_intermediates(radar)
                # Effective steering matrix: g * A (matching beamformer forward)
                A_eff = model.beamformer.g.unsqueeze(-1) * model.beamformer.A
            else:
                pts, conf = model(radar)
                bf_out = None
                A_eff = None

            losses = composite_loss(
                pts,
                lidar,
                conf,
                epoch,
                use_dcd=cfg["use_dcd_loss"],
                use_coverage=cfg["use_coverage_loss"],
                use_confidence=cfg["use_confidence_loss"],
                coverage_threshold=cfg["coverage_threshold"],
                lista_output=bf_out,
                radar_input=radar if bf_out is not None else None,
                steering_matrix=A_eff,
            )

            optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping (after backward, before step)
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            # TRAIN-03: Log gradient norm per LISTA layer (only when unfrozen)
            if stage1_unfrozen and hasattr(model, "beamformer") and hasattr(
                model.beamformer, "lista_layers"
            ):
                for k, layer in enumerate(model.beamformer.lista_layers):
                    total_norm_sq = 0.0
                    for p in layer.parameters():
                        if p.grad is not None:
                            total_norm_sq += p.grad.data.norm(2).item() ** 2
                    grad_norm = total_norm_sq ** 0.5
                    writer.add_scalar(
                        f"grad_norm/lista_layer_{k}", grad_norm, global_step
                    )

            optimizer.step()

            # Accumulate losses for epoch-level logging
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

        # Log learning rate (main param group = first group)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        # --- Stage diagnostics (cheap, always logged) ---
        # Per-stage parameter norms (detect if Stage 1 weights are changing after unfreeze)
        with torch.no_grad():
            s1_norm = sum(p.data.norm().item() ** 2 for p in model.beamformer.parameters()) ** 0.5
            br_norm = sum(p.data.norm().item() ** 2 for p in model.bridge.parameters()) ** 0.5
            dc_norm = sum(p.data.norm().item() ** 2 for p in model.decoder.parameters()) ** 0.5
            writer.add_scalar("param_norm/stage1_beamformer", s1_norm, epoch)
            writer.add_scalar("param_norm/stage2_bridge", br_norm, epoch)
            writer.add_scalar("param_norm/stage3_decoder", dc_norm, epoch)

            # Calibration vector g magnitude (should stay near 1.0 if hardware is good)
            if hasattr(model.beamformer, "cal_gain"):
                g = model.beamformer.cal_gain  # complex tensor (8,)
                writer.add_scalar("calibration/g_mean_mag", g.abs().mean().item(), epoch)
                writer.add_scalar("calibration/g_std_mag", g.abs().std().item(), epoch)

            # Beamformer output energy (detect if LISTA is producing useful output)
            try:
                sample_radar = next(iter(train_loader))[0][:1].to(device)
                bf_out = model.beamformer(sample_radar)  # (1, N_az, 512) complex
                writer.add_scalar("beamformer/output_energy", bf_out.abs().mean().item(), epoch)
                writer.add_scalar("beamformer/output_max", bf_out.abs().max().item(), epoch)
            except Exception:
                pass  # non-critical diagnostic

            # LISTA threshold (rho -> tau) values per layer
            if hasattr(model, "beamformer") and hasattr(model.beamformer, "lista_layers"):
                for k, layer in enumerate(model.beamformer.lista_layers):
                    if hasattr(layer, "rho"):
                        rho_val = layer.rho.item()
                        tau_val = layer.TAU_MIN + (layer.TAU_MAX - layer.TAU_MIN) * torch.sigmoid(layer.rho).item()
                        writer.add_scalar(f"lista/rho_layer_{k}", rho_val, epoch)
                        writer.add_scalar(f"lista/tau_layer_{k}", tau_val, epoch)

        # --- Validation ---
        model.eval()
        val_metrics = evaluate_epoch(model, val_loader, device)
        val_chamfer = val_metrics["chamfer"]
        val_mhd = val_metrics["mod_hausdorff"]

        writer.add_scalar("val/chamfer", val_chamfer, epoch)
        writer.add_scalar("val/mod_hausdorff", val_mhd, epoch)

        # Point cloud collapse detection — compute std on a small batch
        _log_point_cloud_stats(model, train_loader, device, writer, epoch)

        epoch_elapsed = time.time() - epoch_start
        print(
            f"[train] Epoch {epoch:03d} | "
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
                f"[train]   => New best val_chamfer={val_chamfer:.4f} at epoch {epoch}. "
                f"Checkpoint saved."
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"[train] Early stopping at epoch {epoch} (patience exhausted)")
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
            print(f"[train]   => Periodic checkpoint saved: {periodic_path}")

    total_time = time.time() - start_time
    writer.close()

    print(
        f"[train] Done. Best val_chamfer={best_val_chamfer:.4f} at epoch {best_epoch}. "
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
    max_batches: int = 3,
) -> None:
    """Log point cloud statistics for template-collapse detection.

    Computes mean std(pred_x) and std(pred_y) over a few training batches.
    A low std (<0.1m) indicates the model is collapsing to a fixed template.

    Args:
        model:      RadarPointCloudModel in eval mode
        loader:     Training DataLoader
        device:     Compute device
        writer:     TensorBoard SummaryWriter
        epoch:      Current epoch index (for writer x-axis)
        max_batches: Number of batches to sample (default 3, fast)
    """
    model.eval()
    std_x_list = []
    std_y_list = []

    with torch.no_grad():
        for i, (radar, _lidar, _norm) in enumerate(loader):
            if i >= max_batches:
                break
            radar = radar.to(device)
            pts, _ = model(radar)            # (B, 8192, 3)
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
        description="Train mmDar v2 RadarPointCloudModel",
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
        "--no-dcd", action="store_true", default=False,
        dest="no_dcd", help="Disable DCD loss"
    )
    parser.add_argument(
        "--no-confidence", action="store_true", default=False,
        dest="no_confidence", help="Disable confidence loss and weighted Chamfer"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", default=False,
        dest="no_coverage", help="Disable coverage loss"
    )
    parser.add_argument(
        "--freeze-stage1-epochs", type=int,
        default=DEFAULT_CONFIG["freeze_stage1_epochs"],
        dest="freeze_stage1_epochs",
        help="Number of epochs to freeze Stage 1 (LISTA) before joint fine-tuning"
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
        "--model-type", type=str, default="cvnn",
        choices=["cvnn", "magnitude"],
        dest="model_type",
        help="Model type: 'cvnn' (LISTA+complex) or 'magnitude' (FFT+magnitude baseline)"
    )

    args = parser.parse_args()

    # Build config dict from args
    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "log_dir": args.log_dir,
        "processed_dir": args.processed_dir,
        "use_dcd_loss": not args.no_dcd,
        "use_confidence_loss": not args.no_confidence,
        "use_coverage_loss": not args.no_coverage,
        "freeze_stage1_epochs": args.freeze_stage1_epochs,
        "early_stop_patience": args.early_stop_patience,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "checkpoint_every": args.checkpoint_every,
        "num_workers": args.num_workers,
        "model_type": args.model_type,
    }

    train(config)
