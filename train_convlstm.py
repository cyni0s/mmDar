"""Train UNet1ConvLSTM with trajectory-aware batching, dense supervision, and bf16 AMP.

Design notes:
- Zero-initialise ConvLSTM state every batch.  Stride-1 sliding windows overlap
  heavily (batch N = frames 0..40, batch N+1 = frames 1..41), so carrying hidden
  state across windows is mathematically corrupt.  Full BPTT within each T-frame
  window provides temporal learning without cross-window state carry.
- Dense supervision: loss computed at every timestep.  Final step weight = 1.0,
  intermediate steps weight = args.intermediate_weight (default 0.2).  Loss is
  divided by the total weight sum so gradient scale stays comparable to a
  single-frame baseline (Pitfall 6, RESEARCH.md).
- BCELoss is intentionally placed outside the bf16 autocast block; model output
  is cast to float32 before BCE to avoid numerical instability (established AMP
  pattern from Phase 1).
- Gradient norm clipping: scaler.unscale_() is called before clip_grad_norm_
  to operate on the true (unscaled) gradient magnitudes.

Usage:
  python3 train_convlstm.py --batch 4 --lr 1e-4 --epochs 200 --bf16
  python3 train_convlstm.py --batch 2 --lr 1e-4 --epochs 1 --bf16 --dry_run
"""

import os
import sys
import time
import json
import argparse
import datetime
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train_test_utils.model import UNet1ConvLSTM
from train_test_utils.dataloader import SequentialDataset, TrajectoryBatchSampler, seq_collate_fn
from train_test_utils.dice_score import dice_loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet1ConvLSTM')

    # Batch / LR / schedule
    parser.add_argument('--batch', type=int, default=4,
                        help='Number of trajectory slots per batch')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)

    # Sequence
    parser.add_argument('--max_T', type=int, default=41,
                        help='Maximum sequence length (frames per window)')
    parser.add_argument('--variable_t', action='store_true',
                        help='Sample T ~ Uniform(1, max_T) per batch')

    # Loss
    parser.add_argument('--intermediate_weight', type=float, default=0.2,
                        help='Loss weight for non-final timesteps')

    # Training stability
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient norm clipping threshold')
    parser.add_argument('--bf16', action='store_true',
                        help='Enable bfloat16 AMP for forward pass')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing in encoder (saves memory)')
    parser.add_argument('--temporal_chunk', type=int, default=0,
                        help='Chunk T frames into groups of this size for encoder/decoder '
                             '(0=all at once, 4=recommended for batch>=12 with T>=8)')

    # Validation / checkpointing
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save periodic checkpoint every N epochs')

    # Reproducibility / I/O
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--basepath', type=str, default='./dataset_5/')

    # Smoke test
    parser.add_argument('--dry_run', action='store_true',
                        help='Run exactly 1 batch and exit (smoke test)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dense supervision loss
# ---------------------------------------------------------------------------

def compute_dense_loss(preds, targets, intermediate_weight, bce_loss_fn, device):
    """Compute temporally-weighted dense loss over a sequence.

    Args:
        preds:   (B, T, 1, H, W) float32 tensor (model output, sigmoid applied)
        targets: (B, T, 1, H, W) float32 tensor (lidar ground truth)
        intermediate_weight: scalar weight for t < T-1 timesteps
        bce_loss_fn: BCELoss instance
        device: torch.device

    Returns:
        loss: scalar — weighted sum divided by total weight (normalised)
        weights: (T,) tensor of per-timestep weights
    """
    T = preds.shape[1]
    weights = torch.full((T,), intermediate_weight, device=device)
    weights[-1] = 1.0
    weight_sum = weights.sum()

    loss = sum(
        weights[t] * (
            0.9 * bce_loss_fn(preds[:, t].float(), targets[:, t].float())
            + 0.1 * dice_loss(preds[:, t].float(), targets[:, t].float())
        )
        for t in range(T)
    ) / weight_sum  # normalise by weight sum (Pitfall 6)

    return loss, weights


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(model, val_dataset, args, device, bce_loss_fn, epoch):
    """Run a single pass over validation trajectories and return mean loss.

    Validation uses fixed T=args.max_T, no gradient, no AMP.
    Returns val_loss (float).
    """
    model.eval()
    val_sampler = TrajectoryBatchSampler(
        val_dataset,
        batch_size=args.batch,
        max_T=args.max_T,
        variable_t=False,
        seed=args.seed + 9999,  # different seed from training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=seq_collate_fn,
        num_workers=0,   # simpler for validation; dataset is small (2 trajs)
        pin_memory=False,
    )

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            radar_seq, lidar_seq, traj_ids, _ = batch
            radar_seq = radar_seq.to(device)
            lidar_seq = lidar_seq.to(device)

            # Zero-init state (same as training)
            preds, _ = model(radar_seq, state=None)

            val_loss, _ = compute_dense_loss(
                preds, lidar_seq, args.intermediate_weight, bce_loss_fn, device
            )
            val_losses.append(val_loss.item())

    model.train()
    return float(np.mean(val_losses)) if val_losses else float('nan')


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Experiment naming / logging
    # ------------------------------------------------------------------
    dt = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    varT_tag = '_varT' if args.variable_t else ''
    name = (
        f'convlstm_b{args.batch}_lr{args.lr}'
        f'_T{args.max_T}{varT_tag}'
        f"_{'bf16' if args.bf16 else 'fp32'}_{dt}"
    )
    LOG_DIR = os.path.join('./logs', name)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Git commit ID (best-effort — no failure if not in a git repo)
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_sha = 'unknown'

    # Hyperparameter snapshot
    val_traj_ids = [138, 140]
    params = {
        'model': 'UNet1ConvLSTM',
        'hidden_channels': 256,
        'n_channels': 1,
        'n_classes': 1,
        'batch_size': args.batch,
        'lr': args.lr,
        'num_epochs': args.epochs,
        'max_T': args.max_T,
        'variable_t': args.variable_t,
        'intermediate_weight': args.intermediate_weight,
        'grad_clip': args.grad_clip,
        'mixed_precision': args.bf16,
        'gradient_checkpointing': args.gradient_checkpointing,
        'val_every': args.val_every,
        'save_every': args.save_every,
        'seed': args.seed,
        'num_workers': args.num_workers,
        'basepath': args.basepath,
        'val_traj_ids': val_traj_ids,
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
    print(f'  batch={args.batch}, lr={args.lr}, epochs={args.epochs}')
    print(f'  max_T={args.max_T}, variable_t={args.variable_t}')
    print(f'  intermediate_weight={args.intermediate_weight}')
    print(f'  bf16={args.bf16}, grad_clip={args.grad_clip}')
    print(f'  val_traj_ids={val_traj_ids}')
    print(f'  device={device}, git={git_sha}')
    print(f'{"=" * 60}\n')

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print('Loading datasets...')
    train_dataset = SequentialDataset(
        args.basepath, 'train', M=40,
        ABINS_LIDAR_ORIG=512,   # dataset_5 lidar PNGs are 256x512
        exclude_traj_ids=val_traj_ids,
    )
    val_dataset = SequentialDataset(
        args.basepath, 'train', M=40,
        ABINS_LIDAR_ORIG=512,
        include_traj_ids=val_traj_ids,
    )

    train_sampler = TrajectoryBatchSampler(
        train_dataset,
        batch_size=args.batch,
        max_T=args.max_T,
        variable_t=args.variable_t,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=seq_collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    n_train_targets = len(train_dataset)
    n_val_targets = len(val_dataset)
    n_train_trajs = len(train_dataset.traj_data)
    n_val_trajs = len(val_dataset.traj_data)
    print(f'  Train: {n_train_targets} eligible targets across {n_train_trajs} trajectories')
    print(f'  Val:   {n_val_targets} eligible targets across {n_val_trajs} trajectories')
    print(f'  Steps/epoch: {len(train_sampler)}\n')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = UNet1ConvLSTM(n_channels=1, n_classes=1,
                          use_checkpointing=args.gradient_checkpointing,
                          temporal_chunk_size=args.temporal_chunk).to(device)

    try:
        from torchinfo import summary as ti_summary
        ti_summary(model, input_size=(1, 1, 1, 256, 64), device=device)
    except Exception:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Model params: {n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    bce_loss_fn = torch.nn.BCELoss()

    use_amp = args.bf16 and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t0 = time.time()
    best_val_loss = float('inf')
    best_val_epoch = -1
    global_step = 0

    model.train()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_losses = []
        epoch_grad_norms = []

        for batch_idx, batch in enumerate(train_loader):
            radar_seq, lidar_seq, traj_ids, _ = batch

            # Move to device
            radar_seq = radar_seq.to(device, non_blocking=True)   # (B, T, 1, 256, 64)
            lidar_seq = lidar_seq.to(device, non_blocking=True)   # (B, T, 1, 256, 512)

            # Zero grad before forward (zero_grad at top of loop, not after step,
            # to keep the gradient zero-ing as close to backward as possible)
            optimizer.zero_grad(set_to_none=True)

            # IMPORTANT: Zero-init ConvLSTM state every batch.
            # Stride-1 sliding windows overlap so state carry across batches
            # is mathematically corrupt (see module docstring).
            state = None  # model._init_state() handles zero-init internally

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    preds, _ = model(radar_seq, state=state)
                # Dense loss in float32 (BCELoss must be outside autocast)
                loss, _ = compute_dense_loss(
                    preds, lidar_seq, args.intermediate_weight, bce_loss_fn, device
                )
                scaler.scale(loss).backward()
                # Unscale before clipping so clip operates on true grad magnitudes
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                preds, _ = model(radar_seq, state=state)
                loss, _ = compute_dense_loss(
                    preds, lidar_seq, args.intermediate_weight, bce_loss_fn, device
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()

            loss_val = loss.item()
            gn_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

            epoch_losses.append(loss_val)
            epoch_grad_norms.append(gn_val)

            # Per-batch TensorBoard
            writer.add_scalar('train/batch_loss', loss_val, global_step)
            writer.add_scalar('train/batch_grad_norm', gn_val, global_step)
            global_step += 1

            # dry_run: stop after first batch
            if args.dry_run:
                T = preds.shape[1]
                print(f'  [dry_run] batch shapes: radar={list(radar_seq.shape)}, '
                      f'lidar={list(lidar_seq.shape)}, preds={list(preds.shape)}')
                print(f'  [dry_run] T={T}, loss={loss_val:.6f}, grad_norm={gn_val:.4f}')
                print('[dry_run] PASS — exiting early.')
                writer.close()
                return LOG_DIR, name, time.time() - t0, {}

            del preds  # free VRAM before next batch
            gc.collect()

        # ------------------------------------------------------------------
        # End-of-epoch logging
        # ------------------------------------------------------------------
        epoch_loss = float(np.mean(epoch_losses))
        epoch_grad_norm = float(np.mean(epoch_grad_norms))
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/grad_norm', epoch_grad_norm, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)

        # Console print every 10 epochs + first + last
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            elapsed = (time.time() - t0) / 60
            print(f'  Epoch {epoch + 1:3d}/{args.epochs}  '
                  f'loss={epoch_loss:.6f}  grad_norm={epoch_grad_norm:.4f}  '
                  f'elapsed={elapsed:.1f}min')

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss = run_validation(
                model, val_dataset, args, device, bce_loss_fn, epoch
            )
            writer.add_scalar('val/loss', val_loss, epoch)
            print(f'    [val] epoch {epoch + 1}  val_loss={val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                best_ckpt_path = os.path.join(LOG_DIR, 'best.pt_gen')
                torch.save(
                    {'state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch + 1,
                     'val_loss': val_loss},
                    best_ckpt_path,
                )
                print(f'    [val] -> new best (loss={val_loss:.6f}), saved best.pt_gen')

        # ------------------------------------------------------------------
        # Periodic checkpoints
        # ------------------------------------------------------------------
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(LOG_DIR, f'{epoch + 1:03d}.pt_gen')
            torch.save(
                {'state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch + 1},
                ckpt_path,
            )

    # ------------------------------------------------------------------
    # Training summary
    # ------------------------------------------------------------------
    train_time = time.time() - t0
    print(f'\nTraining complete: {train_time / 60:.1f} min '
          f'({train_time / args.epochs:.1f}s/epoch)')
    print(f'Best val_loss={best_val_loss:.6f} at epoch {best_val_epoch}')

    summary_data = {
        'name': name,
        'batch': args.batch,
        'lr': args.lr,
        'epochs': args.epochs,
        'max_T': args.max_T,
        'variable_t': args.variable_t,
        'intermediate_weight': args.intermediate_weight,
        'grad_clip': args.grad_clip,
        'bf16': args.bf16,
        'seed': args.seed,
        'train_time_sec': train_time,
        'sec_per_epoch': train_time / args.epochs,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch,
        'git_sha': git_sha,
    }
    with open(os.path.join(LOG_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    writer.close()
    return LOG_DIR, name, train_time, summary_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    LOG_DIR, name, train_time, summary = train(args)
    if not args.dry_run:
        print(f'\nLog dir: {LOG_DIR}')


if __name__ == '__main__':
    main()
