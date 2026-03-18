"""Sweep checkpoints from a training run: inference + eval per checkpoint.
Outputs a CSV summary of Chamfer / mod-Hausdorff per epoch."""

import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from torchinfo import summary

from train_test_utils.dataloader import *
from train_test_utils.model import *
from eval.eval_pointcloud import evaluate_experiment

# --- Config ---
RUN_DIR = 'logs/baseline_3_20260317-181226'
EPOCHS_TO_EVAL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DATA = 5
HISTORY = 40
GPU = 1

# --- Setup ---
device = torch.device('cuda' if GPU else 'cpu')
torch.manual_seed(0)

# Load data once
basepath = f'./dataset_{DATA}/'
orig_size = [256, 64, 512]
reqd_size = [256, 64, 512]

train_params = {'history': HISTORY}
test_set = Dataset(basepath, 'test',
                   RBINS=reqd_size[0], ABINS_RADAR=reqd_size[1], ABINS_LIDAR=reqd_size[2],
                   RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1], ABINS_LIDAR_ORIG=orig_size[2],
                   M=HISTORY)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# Get ordered filenames (same method as test_radarhd.py)
ordered_filename = test_set.__filenames__()

print(f'Test set: {len(test_loader)} samples')

# Build model once
gen = UNet1(HISTORY + 1, 1).to(device)

results = []

for epoch in EPOCHS_TO_EVAL:
    ckpt_path = os.path.join(RUN_DIR, f'{epoch:03d}.pt_gen')
    if not os.path.exists(ckpt_path):
        print(f'Skipping epoch {epoch} — checkpoint not found')
        continue

    print(f'\n=== Epoch {epoch} ===')

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()

    # Inference
    save_path = os.path.join(RUN_DIR, f'test_imgs_ep{epoch:03d}/')
    os.makedirs(save_path, exist_ok=True)

    t0 = time.time()
    for test_i, (test_data, test_label) in enumerate(test_loader):
        test_data, test_label = test_data.to(device), test_label.to(device)
        with torch.no_grad():
            pred = gen(test_data)
            pred = np.squeeze(pred.cpu().numpy())
            pred = (pred * 255).astype(np.uint8)
            im1 = Image.fromarray(pred)
            im1.save(os.path.join(save_path, f'{epoch:03d}_{ordered_filename[test_i]}_pred.png'))

            label = np.squeeze(test_label.cpu().numpy())
            label = (label * 255).astype(np.uint8)
            im1 = Image.fromarray(label)
            im1.save(os.path.join(save_path, f'{epoch:03d}_{ordered_filename[test_i]}_label.png'))

    t1 = time.time()
    print(f'  Inference: {t1 - t0:.0f}s')

    # Eval
    out_dir = f'results/sweep_ep{epoch:03d}_legacy_cartesian'
    agg = evaluate_experiment(
        pred_dir=save_path,
        label_dir=save_path,
        output_dir=out_dir,
        experiment_name=f'sweep_epoch_{epoch}',
        coordinate_mode='legacy_cartesian',
    )

    cd = agg['chamfer_distance']['median']
    mh = agg['modified_hausdorff']['median']
    print(f'  Chamfer: {cd:.4f}m  mod-Hausdorff: {mh:.4f}m')

    results.append({'epoch': epoch, 'chamfer': cd, 'mod_hausdorff': mh})

# Summary
print('\n\n=== SWEEP SUMMARY ===')
print(f'{"Epoch":>6}  {"Chamfer (m)":>12}  {"mod-Hausdorff (m)":>18}')
for r in results:
    print(f'{r["epoch"]:>6}  {r["chamfer"]:>12.4f}  {r["mod_hausdorff"]:>18.4f}')

# Find best
best_cd = min(results, key=lambda r: r['chamfer'])
best_mh = min(results, key=lambda r: r['mod_hausdorff'])
print(f'\nBest Chamfer:      epoch {best_cd["epoch"]} = {best_cd["chamfer"]:.4f}m')
print(f'Best mod-Hausdorff: epoch {best_mh["epoch"]} = {best_mh["mod_hausdorff"]:.4f}m')
print(f'(Pretrained baseline: Chamfer 0.363m, mod-Hausdorff 0.247m)')
