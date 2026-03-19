"""Run a single training experiment with checkpoint sweep.

Usage:
  python3 run_experiment.py --batch 24 --lr 1.5e-4 --epochs 100 --bf16
  python3 run_experiment.py --batch 12 --lr 1e-4 --epochs 100

Trains, then sweeps all 10-epoch checkpoints by Chamfer distance.
Prints a summary at the end.
"""

import os
import sys
import time
import json
import argparse
import gc

import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from torchinfo import summary

import subprocess

from train_test_utils.dataloader import *
from train_test_utils.model import *
from train_test_utils.dice_score import dice_loss
from eval.eval_pointcloud import evaluate_experiment


def _train_convlstm(args):
    """Dispatch training to train_convlstm.py via subprocess."""
    import datetime
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"convlstm_b{args.batch}_lr{args.lr}_{'bf16' if args.bf16 else 'fp32'}_{dt}"
    LOG_DIR = f'./logs/{name}/'

    cmd = [
        'python3', 'train_convlstm.py',
        '--batch', str(args.batch),
        '--lr', str(args.lr),
        '--epochs', str(args.epochs),
        '--name', name,
    ]
    if args.bf16:
        cmd.append('--bf16')

    print(f'\nDispatching ConvLSTM training: {" ".join(cmd)}')
    t0 = time.time()
    subprocess.run(cmd, check=True)
    train_time = time.time() - t0

    return LOG_DIR, name, train_time


def train(args):
    # Dispatch to ConvLSTM training script when --model convlstm
    if getattr(args, 'model', 'baseline') == 'convlstm':
        return _train_convlstm(args)

    torch.manual_seed(0)
    device = torch.device('cuda')

    import datetime
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"sweep_b{args.batch}_lr{args.lr}_{'bf16' if args.bf16 else 'fp32'}_{dt}"
    LOG_DIR = f'./logs/{name}/'
    os.makedirs(LOG_DIR, exist_ok=True)

    # Save params
    params = {
        'batch_size': args.batch, 'lr': args.lr, 'num_epochs': args.epochs,
        'mixed_precision': args.bf16, 'history': 40, 'msew': 0.9, 'dicew': 0.1,
        'optim': 'adam', 'name': name,
    }
    with open(os.path.join(LOG_DIR, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    # Data
    basepath = './dataset_5/'
    orig_size = [256, 64, 512]
    reqd_size = [256, 64, 512]
    training_set = Dataset(basepath, 'train',
                           RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                           RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2],
                           M=40)
    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True)

    # Model
    gen = UNet1(41, 1).to(device)
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr, weight_decay=0.0005)
    mse_loss_fn = torch.nn.BCELoss()

    use_amp = args.bf16
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f'\n{"="*60}')
    print(f'EXPERIMENT: {name}')
    print(f'  batch={args.batch}, lr={args.lr}, epochs={args.epochs}, bf16={args.bf16}')
    print(f'  steps/epoch={len(train_loader)}')
    print(f'{"="*60}\n')

    t0 = time.time()

    for epoch in range(args.epochs):
        gen.train()
        losses = []
        gen_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (radar, lidar) in enumerate(train_loader):
            radar = radar.to(device)
            lidar = lidar.to(device)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    generated_images = gen(radar)
                generated_images_f32 = generated_images.float()
                loss1 = mse_loss_fn(generated_images_f32, lidar.float())
                loss2 = dice_loss(generated_images_f32, lidar.float())
                gen_loss = params['msew'] * loss1 + params['dicew'] * loss2
                scaler.scale(gen_loss).backward()
                scaler.step(gen_optimizer)
                scaler.update()
            else:
                generated_images = gen(radar)
                loss1 = mse_loss_fn(generated_images, lidar)
                loss2 = dice_loss(generated_images, lidar)
                gen_loss = params['msew'] * loss1 + params['dicew'] * loss2
                gen_loss.backward()
                gen_optimizer.step()

            gen_optimizer.zero_grad(set_to_none=True)
            losses.append(gen_loss.item())

        epoch_loss = np.mean(losses)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f'  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.6f}  elapsed={elapsed/60:.1f}min')

        # Save every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            checkpoint = {'state_dict': gen.state_dict(),
                          'optimizer_state_dict': gen_optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(LOG_DIR, f'{epoch:03d}.pt_gen'))

        gc.collect()

    train_time = time.time() - t0
    print(f'\nTraining complete: {train_time/60:.1f} min ({train_time/args.epochs:.1f}s/epoch)')

    return LOG_DIR, name, train_time


def sweep_checkpoints(LOG_DIR, name, sweep_only=None, model_type='baseline'):
    """Run inference + eval on saved checkpoints.

    For model_type='convlstm', dispatches each checkpoint to test_convlstm.py
    via subprocess and reads results from the saved metrics.json. This avoids
    duplicating ConvLSTM inference logic in this file.

    If sweep_only is provided, only evaluate those epoch numbers.
    Otherwise evaluate all saved checkpoints.
    """
    # Find checkpoints to evaluate
    ckpts = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.pt_gen') and f[0].isdigit()])
    if sweep_only:
        ckpts = [f for f in ckpts if int(f.split('.')[0]) in sweep_only]
    results = []

    print(f'\nSweeping {len(ckpts)} checkpoints ({model_type})...')

    if model_type == 'convlstm':
        # ConvLSTM path: dispatch to test_convlstm.py per checkpoint
        for ckpt_name in ckpts:
            epoch = int(ckpt_name.split('.')[0])
            ckpt_path = os.path.join(LOG_DIR, ckpt_name)
            save_path = os.path.join(LOG_DIR, f'test_imgs_ep{epoch:03d}')
            out_dir = f'results/{name}_ep{epoch:03d}'

            cmd = [
                'python3', 'test_convlstm.py',
                '--checkpoint', ckpt_path,
                '--output_dir', save_path,
                '--T', '41',
                '--eval',
                '--results_dir', out_dir,
                '--experiment_name', f'{name}_ep{epoch}',
            ]
            print(f'  Epoch {epoch:3d}: running inference+eval ...')
            subprocess.run(cmd, check=True)

            # Read metrics from the saved metrics.json
            metrics_path = os.path.join(out_dir, 'metrics.json')
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            agg = metrics_data['aggregate']
            cd = agg['chamfer_distance']['median']
            mh = agg['modified_hausdorff']['median']
            results.append({'epoch': epoch, 'chamfer': cd, 'mod_hausdorff': mh})
            print(f'  Epoch {epoch:3d}: Chamfer={cd:.4f}m  mod-H={mh:.4f}m')

        return results

    # Baseline path: inline inference
    device = torch.device('cuda')
    torch.manual_seed(0)

    basepath = './dataset_5/'
    orig_size = [256, 64, 512]
    reqd_size = [256, 64, 512]
    test_set = Dataset(basepath, 'test',
                       RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                       RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2],
                       M=40)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    ordered_filename = test_set.__filenames__()

    gen = UNet1(41, 1).to(device)

    for ckpt_name in ckpts:
        epoch = int(ckpt_name.split('.')[0])
        ckpt_path = os.path.join(LOG_DIR, ckpt_name)

        checkpoint = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(checkpoint['state_dict'])
        gen.eval()

        # Inference
        save_path = os.path.join(LOG_DIR, f'test_imgs_ep{epoch:03d}/')
        os.makedirs(save_path, exist_ok=True)

        for test_i, (test_data, test_label) in enumerate(test_loader):
            test_data, test_label = test_data.to(device), test_label.to(device)
            with torch.no_grad():
                pred = gen(test_data)
                pred_np = np.squeeze(pred.cpu().numpy())
                pred_np = (pred_np * 255).astype(np.uint8)
                Image.fromarray(pred_np).save(
                    os.path.join(save_path, f'{epoch:03d}_{ordered_filename[test_i]}_pred.png'))

                label_np = np.squeeze(test_label.cpu().numpy())
                label_np = (label_np * 255).astype(np.uint8)
                Image.fromarray(label_np).save(
                    os.path.join(save_path, f'{epoch:03d}_{ordered_filename[test_i]}_label.png'))

        # Eval
        out_dir = f'results/{name}_ep{epoch:03d}'
        agg = evaluate_experiment(
            pred_dir=save_path, label_dir=save_path,
            output_dir=out_dir, experiment_name=f'{name}_ep{epoch}',
            coordinate_mode='legacy_cartesian',
        )

        cd = agg['chamfer_distance']['median']
        mh = agg['modified_hausdorff']['median']
        results.append({'epoch': epoch, 'chamfer': cd, 'mod_hausdorff': mh})
        print(f'  Epoch {epoch:3d}: Chamfer={cd:.4f}m  mod-H={mh:.4f}m')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--sweep-epochs', type=str, default='10,20,30',
                        help='Comma-separated epochs to evaluate (default: 10,20,30)')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'convlstm'],
                        help='Model architecture to train and evaluate.')
    args = parser.parse_args()

    LOG_DIR, name, train_time = train(args)
    sweep_only = [int(e) for e in args.sweep_epochs.split(',')]
    results = sweep_checkpoints(LOG_DIR, name, sweep_only=sweep_only, model_type=args.model)

    # Summary
    print(f'\n{"="*60}')
    print(f'RESULTS: {name}')
    print(f'  Train time: {train_time/60:.1f} min ({train_time/args.epochs:.1f}s/epoch)')
    print(f'{"Epoch":>6}  {"Chamfer (m)":>12}  {"mod-H (m)":>12}')
    for r in results:
        print(f'{r["epoch"]:>6}  {r["chamfer"]:>12.4f}  {r["mod_hausdorff"]:>12.4f}')

    best_cd = min(results, key=lambda r: r['chamfer'])
    best_mh = min(results, key=lambda r: r['mod_hausdorff'])
    print(f'\nBest Chamfer:  epoch {best_cd["epoch"]} = {best_cd["chamfer"]:.4f}m')
    print(f'Best mod-H:    epoch {best_mh["epoch"]} = {best_mh["mod_hausdorff"]:.4f}m')
    print(f'Time to best Chamfer: ~{best_cd["epoch"] * train_time / args.epochs / 60:.1f} min')
    print(f'(Pretrained baseline: Chamfer 0.363m, mod-Hausdorff 0.247m)')
    print(f'{"="*60}')

    # Save summary JSON
    summary_data = {
        'name': name, 'batch': args.batch, 'lr': args.lr,
        'epochs': args.epochs, 'bf16': args.bf16,
        'train_time_sec': train_time,
        'sec_per_epoch': train_time / args.epochs,
        'results': results,
        'best_chamfer': best_cd,
        'best_mod_hausdorff': best_mh,
    }
    with open(f'results/{name}_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)


if __name__ == '__main__':
    main()
