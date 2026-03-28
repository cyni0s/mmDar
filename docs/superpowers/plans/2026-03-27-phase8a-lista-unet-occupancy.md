# Phase 8a: LISTA log_power + U-Net Polar Occupancy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a symmetric 2D U-Net on 41-frame channel-stacked LISTA log_power features to predict polar occupancy, replacing the v2 point decoder. Evaluate whether occupancy thresholding closes the mod-H gap (0.429 → target ≤ 0.200).

**Architecture:** FFTBeamformer(frozen) → log_power → reproject(sin_theta→angle-uniform) → downsample(512r→256r) → stack 41 frames → symmetric U-Net → (1, 256, 512) occupancy → threshold → variable-size point cloud.

**Tech Stack:** PyTorch, existing FFTBeamformer from v2/model/lista.py, existing unet_parts.py building blocks (DoubleConv, Down, Up), existing eval pipeline.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `v2/data/preprocess_lista.py` | Offline: raw IQ → LISTA log_power → reproject → save .pt |
| Create | `v2/data/lista_dataset.py` | Dataset: 41-frame stacking of LISTA features + rasterized lidar labels |
| Create | `v2/model/unet_occupancy.py` | Symmetric U-Net for (41, 256, 512) → (1, 256, 512) occupancy |
| Create | `v2/train/train_occupancy_unet.py` | Training script: BCE + Dice, Adam, checkpoints |
| Create | `v2/eval/occupancy_to_pc.py` | Threshold occupancy → point cloud → Chamfer/mod-H |
| Create | `v2/data/tests/test_lista_dataset.py` | Test dataset shapes, stacking, label consistency |
| Create | `v2/model/tests/test_unet_occupancy.py` | Test U-Net shapes, forward pass |

---

## Task 1: LISTA Preprocessing Script

Offline preprocessing: run FFTBeamformer on all raw IQ, compute log_power, reproject to baseline grid, save as .pt files.

**Files:**
- Create: `v2/data/preprocess_lista.py`

- [ ] **Step 1: Write the preprocessing script**

The script must:
1. Load `v2/data/processed/radar_{tid}.pt` → `(N, 8, 512)` complex64
2. Run FFTBeamformer(N_az=256) → `(N, 256, 512)` complex64
3. Compute log_power: `log(|Re|² + |Im|² + 1e-6)` → `(N, 256, 512)` float32
4. Reproject azimuth: 256 sin_theta-uniform → 512 angle-uniform via `F.grid_sample` 1D interpolation along azimuth axis. Grid mapping: for each target angle bin `a_k = -90° + 180°·k/511`, compute `sin(a_k)`, then find normalized coordinate in LISTA's sin_theta grid `u = sin(a_k)` (already in [-1, 1] which is LISTA's native range).
5. Downsample range: 512 → 256 bins via stride-2 slicing `[:, ::2]`
6. Result: `(N, 256_range, 512_az)` float32 — transposed to match baseline (rows=range, cols=azimuth)
7. Save as `v2/data/processed/lista_logpow_{tid}.pt`

Also rasterize lidar labels to the baseline eval grid:
1. Load `v2/data/processed/lidar_{tid}.pt` → `(N, 8192, 3)` float32
2. Per frame: convert (x,y) → (range, angle), bin into `(256_range, 512_az)` grid using eval_pointcloud.py constants (`_x_axis_grid`, `_y_axis_grid` via `searchsorted`)
3. Save as `v2/data/processed/lista_label_{tid}.pt` → `(N, 256, 512)` uint8

```python
"""Preprocess raw IQ through FFTBeamformer → log_power → baseline grid.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/data/preprocess_lista.py
"""
import os, torch, numpy as np
import torch.nn.functional as F
from v2.model.lista import FFTBeamformer
from v2.data.split import ALL_TRAJS
from eval.eval_pointcloud import _x_axis_grid, _y_axis_grid, RMAX, RBINS, ABINS

PROCESSED_DIR = 'v2/data/processed'
N_AZ_LISTA = 256

def reproject_azimuth(log_power: torch.Tensor) -> torch.Tensor:
    """Reproject from sin_theta-uniform (256 bins) to angle-uniform (512 bins).

    Args:
        log_power: (N, 256_az_sintheta, 512_range) float32

    Returns:
        (N, 512_az_angle, 512_range) float32
    """
    N, A_in, R = log_power.shape
    # Target angle grid: -90° to +90°, 512 bins (baseline convention)
    angles = torch.linspace(-90, 90, ABINS)  # degrees
    sin_vals = torch.sin(angles * torch.pi / 180)  # [-1, 1]
    # sin_vals are already in LISTA's native [-1, 1] range
    # For grid_sample: normalize to [-1, 1] (already there for sin_theta)
    # grid_sample expects (N, H_out, W_out, 2) grid
    # We interpolate along dim=1 (azimuth), keep dim=2 (range) unchanged
    grid_az = sin_vals.unsqueeze(1).expand(ABINS, R)  # (512, 512)
    grid_r = torch.linspace(-1, 1, R).unsqueeze(0).expand(ABINS, R)  # (512, 512)
    grid = torch.stack([grid_r, grid_az], dim=-1)  # (512, 512, 2) — (x=range, y=azimuth)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, 512, 512, 2)
    # Input for grid_sample: (N, 1, 256_az, 512_r) — treat as (N, C, H, W)
    inp = log_power.unsqueeze(1)  # (N, 1, 256, 512)
    out = F.grid_sample(inp, grid.to(inp.device), mode='bilinear',
                        padding_mode='zeros', align_corners=True)
    return out.squeeze(1)  # (N, 512, 512)


def rasterize_lidar_to_baseline_grid(pts: np.ndarray) -> np.ndarray:
    """Rasterize (8192, 3) point cloud to baseline eval grid (256r × 512az).

    Uses searchsorted on eval_pointcloud.py grid constants for consistency.
    """
    grid = np.zeros((RBINS, ABINS), dtype=np.uint8)
    if len(pts) == 0:
        return grid
    x, y = pts[:, 0], pts[:, 1]
    mask = (x >= 0) & (x <= RMAX) & (y >= -RMAX) & (y <= RMAX)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return grid
    row = np.clip(np.searchsorted(_x_axis_grid, x, side='left'), 0, RBINS - 1)
    col = np.clip(np.searchsorted(_y_axis_grid, y, side='left'), 0, ABINS - 1)
    grid[row, col] = 1
    return grid


def process_trajectory(tid: int, bf: FFTBeamformer):
    radar_path = os.path.join(PROCESSED_DIR, f'radar_{tid}.pt')
    lidar_path = os.path.join(PROCESSED_DIR, f'lidar_{tid}.pt')
    if not os.path.exists(radar_path):
        print(f'  Skip {tid}: no radar file')
        return

    radar = torch.load(radar_path, weights_only=True)  # (N, 8, 512) complex
    lidar = torch.load(lidar_path, weights_only=True).numpy()  # (N, 8192, 3)
    N = radar.shape[0]
    print(f'  Traj {tid}: {N} frames')

    # Beamform + log_power (batched, GPU)
    device = next(bf.parameters(), torch.tensor(0)).device
    CHUNK = 256
    logpow_list = []
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        with torch.no_grad():
            spec = bf(radar[s:e].to(device))  # (chunk, 256, 512) complex
        lp = torch.log(spec.real**2 + spec.imag**2 + 1e-6).cpu()  # (chunk, 256, 512)
        logpow_list.append(lp)
    log_power = torch.cat(logpow_list, dim=0)  # (N, 256, 512)

    # Reproject azimuth + downsample range
    reprojected = reproject_azimuth(log_power)  # (N, 512_az, 512_r)
    downsampled = reprojected[:, :, ::2]  # (N, 512_az, 256_r)
    # Transpose to (N, 256_r, 512_az) — baseline convention (rows=range, cols=azimuth)
    features = downsampled.permute(0, 2, 1).contiguous()  # (N, 256, 512)

    # Save features
    out_path = os.path.join(PROCESSED_DIR, f'lista_logpow_{tid}.pt')
    torch.save(features.to(torch.float16), out_path)  # float16 to save space

    # Rasterize lidar labels
    labels = np.stack([rasterize_lidar_to_baseline_grid(lidar[i]) for i in range(N)])
    label_path = os.path.join(PROCESSED_DIR, f'lista_label_{tid}.pt')
    torch.save(torch.from_numpy(labels), label_path)

    print(f'    Features: {features.shape}, Labels: {labels.shape}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bf = FFTBeamformer(N_az=N_AZ_LISTA).to(device)
    bf.eval()
    print(f'Processing {len(ALL_TRAJS)} trajectories on {device}')
    for tid in ALL_TRAJS:
        process_trajectory(tid, bf)
    print('Done.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run preprocessing inside Docker**

```bash
docker compose run --rm mmdar python3 v2/data/preprocess_lista.py
```

Expected: ~10-15 min for all 44 trajectories. Creates `lista_logpow_{tid}.pt` (float16) and `lista_label_{tid}.pt` (uint8) for each trajectory. Total disk: ~12GB float16 features + ~1GB labels.

- [ ] **Step 3: Verify shapes**

```bash
docker compose run --rm mmdar python3 -c "
import torch
f = torch.load('v2/data/processed/lista_logpow_117.pt', weights_only=True)
l = torch.load('v2/data/processed/lista_label_117.pt', weights_only=True)
print(f'Features: {f.shape} {f.dtype}')  # (N, 256, 512) float16
print(f'Labels: {l.shape} {l.dtype}')    # (N, 256, 512) uint8
print(f'Label occupancy rate: {l.float().mean():.4f}')  # should be ~0.5-2%
"
```

- [ ] **Step 4: Commit**

```bash
git add v2/data/preprocess_lista.py
git commit -m "feat(v2/data): LISTA log_power preprocessing to baseline grid"
```

---

## Task 2: Dataset for 41-Frame Stacking

**Files:**
- Create: `v2/data/lista_dataset.py`
- Create: `v2/data/tests/test_lista_dataset.py`

- [ ] **Step 1: Write test**

```python
# v2/data/tests/test_lista_dataset.py
import torch
import pytest


def test_lista_dataset_shapes():
    """Verify dataset returns correct shapes for 41-frame stacking."""
    from v2.data.lista_dataset import LISTAOccDataset
    # Use trajectory 117 (test set, should exist after preprocessing)
    ds = LISTAOccDataset(traj_id=117, processed_dir='v2/data/processed', history=40)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (41, 256, 512), f"Expected (41, 256, 512), got {x.shape}"
    assert y.shape == (1, 256, 512), f"Expected (1, 256, 512), got {y.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert y.min() >= 0 and y.max() <= 1


def test_lista_dataset_stacking_order():
    """History frames should be oldest-first, current frame last."""
    from v2.data.lista_dataset import LISTAOccDataset
    ds = LISTAOccDataset(traj_id=117, processed_dir='v2/data/processed', history=40)
    # Frame at index 40 (first valid sample) should use frames 0..40
    x, y = ds[0]
    assert x.shape[0] == 41  # 40 history + 1 current
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose run --rm mmdar python3 -m pytest v2/data/tests/test_lista_dataset.py -v
```

- [ ] **Step 3: Implement dataset**

```python
# v2/data/lista_dataset.py
"""Dataset for 41-frame stacked LISTA log_power features + occupancy labels.

Mirrors the baseline Dataset class behavior: stacks M history frames + 1 current
frame as input channels. Labels are binary polar occupancy on the baseline grid.

Usage:
    from v2.data.lista_dataset import build_lista_dataloaders
    loaders = build_lista_dataloaders('v2/data/processed', history=40, batch_size=12)
    for x, y in loaders['train']:
        # x: (B, 41, 256, 512) float32 — stacked LISTA log_power
        # y: (B, 1, 256, 512) float32 — binary occupancy label
"""
import os
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS


class LISTAOccDataset(Dataset):
    """Single-trajectory dataset for LISTA log_power + occupancy labels.

    Args:
        traj_id: trajectory ID
        processed_dir: directory containing lista_logpow_{tid}.pt and lista_label_{tid}.pt
        history: number of history frames (M). Total input channels = history + 1.
    """

    def __init__(self, traj_id: int, processed_dir: str, history: int = 40):
        self.history = history
        self.features = torch.load(
            os.path.join(processed_dir, f'lista_logpow_{traj_id}.pt'),
            weights_only=True,
        ).float()  # (N, 256, 512) float32

        self.labels = torch.load(
            os.path.join(processed_dir, f'lista_label_{traj_id}.pt'),
            weights_only=True,
        ).float()  # (N, 256, 512) float32

        assert self.features.shape[0] == self.labels.shape[0]
        self.n_frames = self.features.shape[0]

    def __len__(self):
        return max(0, self.n_frames - self.history)

    def __getitem__(self, idx):
        # Stack history + current frame
        start = idx
        end = idx + self.history + 1  # inclusive of current
        x = self.features[start:end]  # (history+1, 256, 512)
        y = self.labels[end - 1].unsqueeze(0)  # (1, 256, 512) — current frame label
        return x, y


def build_lista_dataloaders(
    processed_dir: str,
    history: int = 40,
    batch_size: int = 12,
    num_workers: int = 4,
) -> dict:
    """Build train/val/test DataLoaders for LISTA occupancy training.

    Returns dict with 'train', 'val', 'test' DataLoader keys.
    """
    split_configs = {
        'train': (TRAIN_TRAJS, True),
        'val': (VAL_TRAJS, False),
        'test': (TEST_TRAJS, False),
    }
    loaders = {}
    for split, (trajs, shuffle) in split_configs.items():
        datasets = []
        for tid in trajs:
            feat_path = os.path.join(processed_dir, f'lista_logpow_{tid}.pt')
            if os.path.exists(feat_path):
                datasets.append(LISTAOccDataset(tid, processed_dir, history))
        if datasets:
            combined = ConcatDataset(datasets)
            loaders[split] = DataLoader(
                combined, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True,
            )
        else:
            loaders[split] = None
    return loaders
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm mmdar python3 -m pytest v2/data/tests/test_lista_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add v2/data/lista_dataset.py v2/data/tests/test_lista_dataset.py
git commit -m "feat(v2/data): LISTA occupancy dataset with 41-frame stacking"
```

---

## Task 3: Symmetric U-Net for Occupancy

**Files:**
- Create: `v2/model/unet_occupancy.py`
- Create: `v2/model/tests/test_unet_occupancy.py`

- [ ] **Step 1: Write test**

```python
# v2/model/tests/test_unet_occupancy.py
import torch
import pytest


def test_unet_occ_forward_shape():
    """U-Net output matches input spatial dims with 1 output channel."""
    from v2.model.unet_occupancy import UNetOcc
    model = UNetOcc(n_channels=41, n_classes=1)
    x = torch.randn(2, 41, 256, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 512), f"Expected (2, 1, 256, 512), got {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output should be sigmoid-bounded"


def test_unet_occ_param_count():
    """U-Net should have 2-6M params (not 75K like Phase 3, not 17M like baseline)."""
    from v2.model.unet_occupancy import UNetOcc
    model = UNetOcc(n_channels=41, n_classes=1)
    n_params = sum(p.numel() for p in model.parameters())
    assert 2_000_000 < n_params < 6_000_000, f"Expected 2-6M params, got {n_params:,}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose run --rm mmdar python3 -m pytest v2/model/tests/test_unet_occupancy.py -v
```

- [ ] **Step 3: Implement symmetric U-Net**

```python
# v2/model/unet_occupancy.py
"""Symmetric U-Net for polar occupancy prediction.

Takes (B, C_in, 256, 512) feature maps and outputs (B, 1, 256, 512) occupancy.
No asymmetric azimuth upsampling (unlike baseline UNet1 which does 64→512).

Uses the same DoubleConv/Down/Up building blocks from train_test_utils/unet_parts.py.
"""
import torch.nn as nn
from train_test_utils.unet_parts import DoubleConv, Down, Up, OutConv


class UNetOcc(nn.Module):
    """Symmetric 4-level U-Net for polar occupancy.

    Architecture:
        Encoder: inc(C→64) → down1(64→128) → down2(128→256) → down3(256→512) → down4(512→512)
        Decoder: up1(1024→256) → up2(512→128) → up3(256→64) → up4(128→64) → outc(64→1) → sigmoid

    Args:
        n_channels: input channels (default 41 for 41-frame stacking)
        n_classes: output channels (default 1 for binary occupancy)
        bilinear: use bilinear upsampling (default True, matches baseline)
    """

    def __init__(self, n_channels=41, n_classes=1, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.sigmoid(x)
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm mmdar python3 -m pytest v2/model/tests/test_unet_occupancy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add v2/model/unet_occupancy.py v2/model/tests/test_unet_occupancy.py
git commit -m "feat(v2/model): symmetric U-Net for polar occupancy prediction"
```

---

## Task 4: Occupancy → Point Cloud Eval

**Files:**
- Create: `v2/eval/occupancy_to_pc.py`

- [ ] **Step 1: Write the eval conversion**

This module converts predicted occupancy grids to point clouds and computes metrics. Uses the baseline's eval grid constants for Cartesian conversion.

```python
# v2/eval/occupancy_to_pc.py
"""Convert polar occupancy predictions to point clouds and evaluate.

Uses the same grid constants as eval/eval_pointcloud.py for Cartesian conversion.
Threshold → findNonZero → grid-to-meters → Chamfer/mod-H.
"""
import numpy as np
import torch
import cv2
from eval.eval_pointcloud import (
    _x_axis_grid, _y_axis_grid, RBINS, ABINS,
    polar_image_to_pointcloud, COORD_MODE_LEGACY,
    chamfer_distance, modified_hausdorff,
)


def occupancy_to_pointcloud(occ: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert (256, 512) occupancy probability to (N, 2) point cloud.

    Uses the baseline eval grid for coordinate conversion.
    The input is assumed to be on the baseline's (256r × 512az) angle-uniform grid.

    Args:
        occ: (256, 512) float, values in [0, 1]
        threshold: binarization threshold

    Returns:
        (N, 2) float64 point cloud in meters (x, y)
    """
    binary = (occ >= threshold).astype(np.uint8) * 255
    return polar_image_to_pointcloud(binary, threshold=1,
                                     coordinate_mode=COORD_MODE_LEGACY)


def evaluate_occupancy_model(model, dataloader, device,
                             threshold: float = 0.5) -> dict:
    """Run occupancy model inference and compute point cloud metrics.

    Args:
        model: U-Net occupancy model, output (B, 1, 256, 512) sigmoid
        dataloader: yields (features, labels) batches
        device: torch device
        threshold: occupancy binarization threshold

    Returns:
        dict with chamfer_mean, mod_h_mean, n_samples
    """
    model.eval()
    chamfer_accum = 0.0
    hausdorff_accum = 0.0
    n_samples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            pred = model(features.to(device))  # (B, 1, 256, 512)
            pred_np = pred.squeeze(1).cpu().numpy()  # (B, 256, 512)
            label_np = labels.squeeze(1).cpu().numpy()  # (B, 256, 512)

            for i in range(pred_np.shape[0]):
                pc_pred = occupancy_to_pointcloud(pred_np[i], threshold)
                pc_label = occupancy_to_pointcloud(label_np[i], 0.5)

                if pc_pred.shape[0] == 0 or pc_label.shape[0] == 0:
                    continue

                chamfer_accum += chamfer_distance(pc_pred, pc_label)
                hausdorff_accum += modified_hausdorff(pc_pred, pc_label)
                n_samples += 1

    if n_samples == 0:
        return {'chamfer_mean': float('nan'), 'mod_h_mean': float('nan'),
                'n_samples': 0}

    return {
        'chamfer_mean': chamfer_accum / n_samples,
        'mod_h_mean': hausdorff_accum / n_samples,
        'n_samples': n_samples,
    }
```

Note: This uses scipy.cdist (CPU) via the existing eval functions. For full test set eval (~18K samples), switch to GPU eval or accept ~30 min runtime. The point clouds from occupancy are small (~500-2000 points, not 8192) so scipy is faster than the chunked torch.cdist approach.

- [ ] **Step 2: Commit**

```bash
git add v2/eval/occupancy_to_pc.py
git commit -m "feat(v2/eval): occupancy to point cloud conversion + metrics"
```

---

## Task 5: Training Script

**Files:**
- Create: `v2/train/train_occupancy_unet.py`

- [ ] **Step 1: Write the training script**

```python
# v2/train/train_occupancy_unet.py
"""Train symmetric U-Net on LISTA log_power features for polar occupancy.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/train/train_occupancy_unet.py

Matches baseline training setup: Adam, lr=7e-5, BCE + Dice loss.
Saves checkpoints every 10 epochs. Evaluates on val set after each epoch.
"""
import os, sys, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.model.unet_occupancy import UNetOcc
from v2.data.lista_dataset import build_lista_dataloaders


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice loss (same as baseline RadarHD)."""
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        pred = model(features)
        loss = composite_loss(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            pred = model(features)
            loss = composite_loss(pred, labels)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log-dir', default='logs/v2_lista_unet_occ')
    parser.add_argument('--processed-dir', default='v2/data/processed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = UNetOcc(n_channels=41, n_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    loaders = build_lista_dataloaders(
        args.processed_dir, history=40,
        batch_size=args.batch_size, num_workers=4,
    )
    print(f'Train: {len(loaders["train"].dataset)} samples')
    print(f'Val: {len(loaders["val"].dataset)} samples')

    os.makedirs(args.log_dir, exist_ok=True)
    config = vars(args)
    config['n_params'] = n_params
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, loaders['train'], optimizer, device)
        val_loss = val_epoch(model, loaders['val'], device)
        elapsed = time.time() - t0

        print(f'Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(args.log_dir, 'best.pt'))

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(args.log_dir, f'epoch_{epoch:03d}.pt'))

    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints in: {args.log_dir}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add v2/train/train_occupancy_unet.py
git commit -m "feat(v2/train): training script for LISTA U-Net occupancy model"
```

---

## Task 6: Run Preprocessing + Training

- [ ] **Step 1: Run LISTA preprocessing (offline, ~15 min)**

```bash
docker compose run --rm mmdar python3 v2/data/preprocess_lista.py
```

- [ ] **Step 2: Verify preprocessed data**

```bash
docker compose run --rm mmdar python3 -c "
import torch, os
for tid in [117, 135, 227]:
    f = torch.load(f'v2/data/processed/lista_logpow_{tid}.pt', weights_only=True)
    l = torch.load(f'v2/data/processed/lista_label_{tid}.pt', weights_only=True)
    print(f'Traj {tid}: features {f.shape} {f.dtype}, labels {l.shape} {l.dtype}, occ_rate {l.float().mean():.4f}')
"
```

- [ ] **Step 3: Start training (background, ~2 min/epoch × 50 = ~100 min)**

```bash
docker compose run --rm mmdar python3 v2/train/train_occupancy_unet.py \
    --epochs 50 --batch-size 12 --lr 7e-5 \
    --log-dir logs/v2_lista_unet_occ
```

- [ ] **Step 4: Evaluate best checkpoint (threshold sweep)**

```bash
docker compose run --rm mmdar python3 -c "
import torch, sys
sys.path.insert(0, '.')
from v2.model.unet_occupancy import UNetOcc
from v2.data.lista_dataset import build_lista_dataloaders
from v2.eval.occupancy_to_pc import evaluate_occupancy_model

device = torch.device('cuda')
model = UNetOcc(41, 1).to(device)
ckpt = torch.load('logs/v2_lista_unet_occ/best.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

loaders = build_lista_dataloaders('v2/data/processed', history=40, batch_size=1, num_workers=0)

for thresh in [0.3, 0.5, 0.7]:
    results = evaluate_occupancy_model(model, loaders['test'], device, threshold=thresh)
    print(f'Threshold {thresh:.1f}: Chamfer {results[\"chamfer_mean\"]:.4f}, '
          f'mod-H {results[\"mod_h_mean\"]:.4f}, N={results[\"n_samples\"]}')
"
```

- [ ] **Step 5: Document results in results/README.md and commit**

```bash
git add results/README.md logs/v2_lista_unet_occ/config.json
git commit -m "results: Phase 8a — LISTA U-Net occupancy (Chamfer X.XXX, mod-H X.XXX)"
```
