"""Retrain baseline UNet1 with proper val split — honest evaluation.

Same hyperparameters as the best baseline (batch=12, lr=7e-5, fp32, 50 epochs).
But with a proper val split: checkpoint selected on VAL mod-H, not test.

This gives the baseline's TRUE honest performance for fair comparison.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_baseline_honest.py
"""

import sys, os, time, json
import torch
import torch.optim as optim
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from train_test_utils.model import UNet1
from train_test_utils.dataloader import Dataset
from train_test_utils.dice_score import dice_loss
from eval.eval_pointcloud import (
    polar_image_to_pointcloud, COORD_MODE_LEGACY,
    chamfer_distance, modified_hausdorff,
)

# Same hyperparams as best baseline run
BATCH = 12
LR = 7e-5
EPOCHS = 50
HISTORY = 40
LOG_DIR = 'logs/baseline_honest'

# Val split: use 8 trajectories from train pool
# dataset_5/train/ contains trajectories that are NOT in dataset_5/test/
# We identify val trajectories by their IDs after loading
VAL_TRAJ_IDS = {113, 119, 125, 131, 135, 136, 138, 140}


def split_dataset_by_trajectory(dataset, val_traj_ids):
    """Split a Dataset into train/val based on trajectory IDs in filenames."""
    train_indices = []
    val_indices = []
    filenames = dataset.__filenames__()
    for i, fname in enumerate(filenames):
        traj_id = int(fname.split('_')[0])
        if traj_id in val_traj_ids:
            val_indices.append(i)
        else:
            train_indices.append(i)
    return train_indices, val_indices


def eval_on_indices(model, dataset, indices, device, threshold=0.010):
    """Evaluate model on specific dataset indices. Returns Chamfer + mod-H."""
    model.eval()
    cd_list, mh_list = [], []
    with torch.no_grad():
        for idx in indices:
            radar, label = dataset[idx]
            radar = radar.unsqueeze(0).to(device)
            pred = model(radar)
            pred_np = np.clip(pred.squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            label_np = np.clip(label.squeeze().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            # Use float threshold (sigmoid space)
            pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
            pc_pred = polar_image_to_pointcloud(pred_bin, threshold=1,
                                                 coordinate_mode=COORD_MODE_LEGACY)
            pc_label = polar_image_to_pointcloud(label_np, threshold=1,
                                                  coordinate_mode=COORD_MODE_LEGACY)
            if pc_pred.shape[0] < 2 or pc_label.shape[0] < 2:
                continue
            cd_list.append(chamfer_distance(pc_pred, pc_label))
            mh_list.append(modified_hausdorff(pc_pred, pc_label))
    if not cd_list:
        return float('nan'), float('nan'), 0
    return float(np.median(cd_list)), float(np.median(mh_list)), len(cd_list)


def main():
    device = torch.device('cuda')
    torch.manual_seed(0)
    print(f'Baseline honest eval: batch={BATCH}, lr={LR}, fp32, {EPOCHS} epochs', flush=True)

    # Load ALL training data
    orig_size = [256, 64, 512]
    full_train_set = Dataset('dataset_5/', 'train',
                             RBINS=orig_size[0], ABINS_RADAR=orig_size[1],
                             ABINS_LIDAR=orig_size[2],
                             RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1],
                             ABINS_LIDAR_ORIG=orig_size[2], M=HISTORY)

    # Split into train/val by trajectory
    train_idx, val_idx = split_dataset_by_trajectory(full_train_set, VAL_TRAJ_IDS)
    print(f'Train: {len(train_idx)} samples, Val: {len(val_idx)} samples', flush=True)

    train_subset = torch.utils.data.Subset(full_train_set, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

    # Load test set
    test_set = Dataset('dataset_5/', 'test',
                       RBINS=orig_size[0], ABINS_RADAR=orig_size[1],
                       ABINS_LIDAR=orig_size[2],
                       RBINS_ORIG=orig_size[0], ABINS_RADAR_ORIG=orig_size[1],
                       ABINS_LIDAR_ORIG=orig_size[2], M=HISTORY)
    test_indices = list(range(len(test_set)))
    print(f'Test: {len(test_set)} samples', flush=True)

    # Model — same as baseline
    gen = UNet1(HISTORY + 1, 1).to(device)
    optimizer = optim.Adam(gen.parameters(), lr=LR, weight_decay=0.0005)
    bce_loss = torch.nn.BCELoss()

    os.makedirs(LOG_DIR, exist_ok=True)

    best_val_mh = float('inf')
    for epoch in range(EPOCHS):
        t0 = time.time()
        gen.train()
        total_loss = 0
        n_batches = 0
        for radar, lidar in train_loader:
            radar, lidar = radar.to(device), lidar.to(device)
            pred = gen(radar)
            loss = bce_loss(pred, lidar) + dice_loss(pred, lidar.float(), multiclass=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Val eval every 5 epochs (expensive with scipy)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            val_cd, val_mh, val_n = eval_on_indices(
                gen, full_train_set, val_idx, device, threshold=0.010)
            val_str = f'val_cd {val_cd:.4f} val_mh {val_mh:.4f} (n={val_n})'
            if val_mh < best_val_mh:
                best_val_mh = val_mh
                torch.save({'state_dict': gen.state_dict(), 'epoch': epoch,
                            'val_mh': val_mh},
                           os.path.join(LOG_DIR, 'best.pt_gen'))
        else:
            val_str = 'val skip'

        elapsed = time.time() - t0
        print(f'Ep {epoch:3d} | loss {train_loss:.4f} | {val_str} | {elapsed:.0f}s', flush=True)

        if (epoch + 1) % 10 == 0:
            torch.save({'state_dict': gen.state_dict(), 'epoch': epoch},
                       os.path.join(LOG_DIR, f'{epoch:03d}.pt_gen'))

    # Final test eval with threshold sweep
    print('\n=== TEST EVALUATION (honest, val-selected checkpoint) ===', flush=True)
    best_ckpt = torch.load(os.path.join(LOG_DIR, 'best.pt_gen'), map_location=device)
    gen.load_state_dict(best_ckpt['state_dict'])
    print(f'Best epoch: {best_ckpt["epoch"]}, val mod-H: {best_ckpt["val_mh"]:.4f}', flush=True)

    for thresh in [0.004, 0.008, 0.010, 0.012]:
        cd, mh, n = eval_on_indices(gen, test_set, test_indices, device, threshold=thresh)
        print(f'  thresh={thresh:.3f}: Chamfer {cd:.4f}, mod-H {mh:.4f}, N={n}', flush=True)

    print(f'\nOriginal baseline (test-selected): Chamfer 0.295, mod-H 0.189', flush=True)
    print(f'Original baseline (thresh=0.010, test-selected): mod-H 0.175', flush=True)


if __name__ == '__main__':
    main()
