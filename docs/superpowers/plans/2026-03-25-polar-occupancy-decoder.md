# Polar Occupancy Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the point cloud template decoder with a polar occupancy decoder that preserves angular topology from the LISTA beamformer, fixing the 2x mod-Hausdorff gap vs baseline.

**Architecture:** LISTA beamformer (256 azimuth x 512 range, complex) -> channelize to [Re, Im, log_power] (3ch real) -> dilated residual conv head -> (1, 256, 512) polar occupancy logits. Eval converts occupied cells to Cartesian point cloud for Chamfer/mod-H comparison.

**Tech Stack:** PyTorch, scipy (already used in eval_adapter.py), existing LISTA/FFT beamformers, focal BCE + Dice loss.

**Blockers addressed (from Codex review):**
1. Eval compares predicted occupancy points against **original lidar point clouds**, NOT re-rasterized GT occupancy points. This ensures numbers are comparable to baseline.
2. Empty predictions are penalized (max distance), not skipped.
3. Training uses soft Gaussian labels; eval uses original lidar 3D points. These are separate.
4. Channelizer includes explicit per-channel normalization (InstanceNorm2d with affine=True, normalizes each of Re/Im/log_power independently over spatial dims).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `v2/data/rasterize.py` | Lidar point cloud (8192,3) -> polar occupancy grid (256,512) |
| Create | `v2/data/tests/test_rasterize.py` | Unit tests for rasterization |
| Create | `v2/model/occupancy.py` | Channelizer + DilatedResHead + OccupancyModel assembly |
| Create | `v2/model/tests/test_occupancy.py` | Unit tests for occupancy model |
| Create | `v2/train/loss_occupancy.py` | Focal BCE + Dice loss |
| Create | `v2/train/tests/test_loss_occupancy.py` | Unit tests for loss |
| Create | `v2/eval/occupancy_eval.py` | Occupancy map -> point cloud -> Chamfer/mod-H |
| Create | `v2/eval/tests/test_occupancy_eval.py` | Unit tests for eval conversion |
| Create | `v2/train/train_occupancy.py` | Training script for occupancy model |
| Modify | `v2/data/dataset.py` | Add OccupancyDataset that loads radar + occupancy labels |
| Modify | `v2/model/__init__.py` | Export new model classes |

---

## Coordinate Convention (CRITICAL)

All modules MUST use LISTA's native grid consistently:

- **Azimuth axis (dim 0, H=256):** sin(theta) grid from -1 to +1, i.e. sin_theta[k] = -1 + 2*k/255 for k=0..255. Maps to theta from -90deg to +90deg.
- **Range axis (dim 1, W=512):** range bins 0..511 mapping to [0, R_MAX] meters. R_MAX = 10.8m (from eval constants).
- **Tensor layout:** (B, C, A=256, R=512) where A=azimuth, R=range. Same axis order as LISTA output (B, N_az, R).
- **Polar-to-Cartesian:** x = range * cos(theta), y = range * sin(theta), z = 0.

This differs from the baseline PNG convention (rows=range, cols=azimuth with 512 azimuth bins). Our grid has 256 azimuth bins (LISTA native) not 512. This is fine because eval computes point cloud metrics in Cartesian space, not pixel space.

---

## Task 1: Polar Rasterization

Converts lidar point clouds to polar occupancy grids matching LISTA's angular grid.

**Files:**
- Create: `v2/data/rasterize.py`
- Create: `v2/data/tests/test_rasterize.py`

- [ ] **Step 1: Write failing test for rasterize_to_polar**

```python
# v2/data/tests/test_rasterize.py
import numpy as np
import torch
from v2.data.rasterize import rasterize_to_polar

def test_single_point_broadside():
    """A point at (x=5, y=0, z=0) should land at azimuth=0, range=5m."""
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.shape == (256, 512)
    assert occ.dtype == np.float32
    # Broadside: sin(theta)=0 -> bin 127 or 128 (center of 256 grid)
    az_bin = 128  # round((0+1)*255/2) = 127.5 -> 128
    r_bin = round(5.0 / 10.8 * 511)  # ~236
    assert occ[az_bin, r_bin] > 0, f"Expected occupied at ({az_bin}, {r_bin})"
    assert occ.sum() > 0 and occ.sum() < 10, "Should have ~1 occupied cell"

def test_point_at_45deg():
    """A point at 45deg azimuth, range=7m."""
    theta = np.radians(45)
    r = 7.0
    pts = np.array([[r * np.cos(theta), r * np.sin(theta), 0.0]], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    sin_val = np.sin(theta)  # ~0.707
    expected_az = round((sin_val + 1.0) * 255 / 2.0)  # ~218
    expected_r = round(r / 10.8 * 511)  # ~331
    assert occ[expected_az, expected_r] > 0

def test_empty_cloud():
    """Empty point cloud -> all-zero occupancy."""
    pts = np.zeros((0, 3), dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.sum() == 0

def test_out_of_range_filtered():
    """Points beyond r_max or behind sensor (x<0) should be filtered."""
    pts = np.array([
        [15.0, 0.0, 0.0],   # beyond r_max=10.8
        [-1.0, 0.0, 0.0],   # behind sensor
        [5.0, 0.0, 0.0],    # valid
    ], dtype=np.float32)
    occ = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8)
    assert occ.sum() > 0 and occ.sum() < 5  # only the valid point

def test_gaussian_softening():
    """With sigma>0, occupied cells should have soft Gaussian spread."""
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    occ_hard = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8, sigma=0)
    occ_soft = rasterize_to_polar(pts, N_az=256, N_r=512, r_max=10.8, sigma=1.0)
    assert occ_soft.sum() > occ_hard.sum(), "Soft labels should spread"
    assert occ_soft.max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/data/tests/test_rasterize.py -v`
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement rasterize_to_polar**

```python
# v2/data/rasterize.py
"""Rasterize 3D lidar point clouds into polar occupancy grids.

Converts (N, 3) XYZ point clouds to (N_az, N_r) binary/soft occupancy
grids matching LISTA's angular grid convention:
    sin_theta[k] = -1 + 2*k/(N_az - 1), k = 0..N_az-1
    range[j] = j * r_max / (N_r - 1), j = 0..N_r-1

Coordinate convention:
    x = range * cos(theta)  (forward, always >= 0 for valid targets)
    y = range * sin(theta)  (lateral)
    z = ignored (flat ground prior)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def rasterize_to_polar(
    pts: np.ndarray,
    N_az: int = 256,
    N_r: int = 512,
    r_max: float = 10.8,
    sigma: float = 0.0,
) -> np.ndarray:
    """Convert XYZ point cloud to polar occupancy grid.

    Args:
        pts:   (N, 3) float32 point cloud [x, y, z]. z is ignored.
        N_az:  Number of azimuth bins (default 256, matches LISTA).
        N_r:   Number of range bins (default 512).
        r_max: Maximum range in meters (default 10.8).
        sigma: Gaussian softening sigma in bins (0 = hard binary).

    Returns:
        (N_az, N_r) float32 occupancy grid, values in [0, 1].
    """
    occ = np.zeros((N_az, N_r), dtype=np.float32)

    if len(pts) == 0:
        return occ

    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x**2 + y**2)
    sin_theta = np.zeros_like(r)
    np.divide(y, r, out=sin_theta, where=(r > 1e-8))

    # Filter: valid range and forward-facing
    mask = (r > 0.01) & (r <= r_max) & (x > 0) & (np.abs(sin_theta) <= 1.0)
    r = r[mask]
    sin_theta = sin_theta[mask]

    if len(r) == 0:
        return occ

    # Bin indices (LISTA convention)
    az_bins = np.round((sin_theta + 1.0) * (N_az - 1) / 2.0).astype(int)
    r_bins = np.round(r / r_max * (N_r - 1)).astype(int)

    # Clip to valid range
    az_bins = np.clip(az_bins, 0, N_az - 1)
    r_bins = np.clip(r_bins, 0, N_r - 1)

    occ[az_bins, r_bins] = 1.0

    if sigma > 0:
        occ = gaussian_filter(occ, sigma=sigma)
        occ = np.clip(occ / max(occ.max(), 1e-8), 0.0, 1.0)

    return occ
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/data/tests/test_rasterize.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add v2/data/rasterize.py v2/data/tests/test_rasterize.py
git commit -m "feat(v2): add polar rasterization for occupancy labels"
```

---

## Task 2: Occupancy Dataset

Extends the data pipeline to return polar occupancy labels alongside radar frames.

**Files:**
- Modify: `v2/data/dataset.py`
- Create: `v2/data/tests/test_occupancy_dataset.py`

- [ ] **Step 1: Write failing test for OccupancyTrajectoryDataset**

```python
# v2/data/tests/test_occupancy_dataset.py
import os
import torch
import numpy as np
import pytest
from v2.data.rasterize import rasterize_to_polar


@pytest.fixture
def tmp_processed(tmp_path):
    """Create minimal fake processed data with occupancy labels."""
    N = 5
    traj_id = 999
    radar = torch.randn(N, 8, 512, dtype=torch.complex64)
    lidar = torch.randn(N, 8192, 3)
    norm = torch.ones(N)

    # Rasterize lidar to occupancy
    occ_list = []
    for i in range(N):
        pts = lidar[i].numpy()
        # Place some valid points
        pts[:10, 0] = np.random.uniform(1, 10, 10)  # x > 0
        pts[:10, 1] = np.random.uniform(-5, 5, 10)
        pts[:10, 2] = 0
        occ_list.append(rasterize_to_polar(pts[:10]))
    occ = torch.from_numpy(np.stack(occ_list))  # (N, 256, 512)

    torch.save(radar, str(tmp_path / f"radar_{traj_id}.pt"))
    torch.save(lidar, str(tmp_path / f"lidar_{traj_id}.pt"))
    torch.save(norm, str(tmp_path / f"norm_{traj_id}.pt"))
    torch.save(occ, str(tmp_path / f"occ_{traj_id}.pt"))

    return str(tmp_path), traj_id


def test_occupancy_dataset_loads(tmp_processed):
    from v2.data.dataset import OccupancyTrajectoryDataset
    proc_dir, tid = tmp_processed
    ds = OccupancyTrajectoryDataset(tid, proc_dir)
    assert len(ds) == 5
    radar, lidar, occ, norm = ds[0]
    assert radar.shape == (8, 512)
    assert radar.dtype == torch.complex64
    assert lidar.shape == (8192, 3)
    assert lidar.dtype == torch.float32
    assert occ.shape == (256, 512)
    assert occ.dtype == torch.float32
    assert norm.shape == ()


def test_occupancy_dataset_augment(tmp_processed):
    from v2.data.dataset import OccupancyTrajectoryDataset
    proc_dir, tid = tmp_processed
    ds = OccupancyTrajectoryDataset(tid, proc_dir, augment=True)
    r1, _, o1, _ = ds[0]
    r2, _, o2, _ = ds[0]
    # Radar should differ (random augmentation), occupancy unchanged
    assert not torch.allclose(r1, r2), "Augmentation should change radar"
    assert torch.allclose(o1, o2), "Occupancy labels should not be augmented"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/data/tests/test_occupancy_dataset.py -v`
Expected: ImportError (OccupancyTrajectoryDataset doesn't exist)

- [ ] **Step 3: Add OccupancyTrajectoryDataset to dataset.py**

Add to `v2/data/dataset.py` after the existing TrajectoryDataset class:

```python
class OccupancyTrajectoryDataset(Dataset):
    """Dataset returning radar frames + polar occupancy labels.

    Parameters
    ----------
    traj_id : int
    processed_dir : str
        Must contain occ_{traj_id}.pt (N, 256, 512) float32.
    augment : bool
        If True, apply IQ-domain augmentation to radar (NOT to occupancy).

    Returns (per item)
    ------------------
    radar_frame : (8, 512) complex64
    lidar_frame : (8192, 3) float32 original lidar (for eval against real GT)
    occ_label : (256, 512) float32 polar occupancy (for training loss)
    norm_factor : scalar float32
    """

    def __init__(self, traj_id, processed_dir, augment=False, noise_sigma=0.01):
        super().__init__()
        self.traj_id = traj_id
        self.augment = augment
        self.noise_sigma = noise_sigma

        self.radar = torch.load(
            os.path.join(processed_dir, f"radar_{traj_id}.pt"), weights_only=True
        )
        self.lidar = torch.load(
            os.path.join(processed_dir, f"lidar_{traj_id}.pt"), weights_only=True
        )
        self.occ = torch.load(
            os.path.join(processed_dir, f"occ_{traj_id}.pt"), weights_only=True
        )
        self.norm_factors = torch.load(
            os.path.join(processed_dir, f"norm_{traj_id}.pt"), weights_only=True
        )

        assert self.radar.shape[0] == self.lidar.shape[0] == self.occ.shape[0] == self.norm_factors.shape[0]
        assert self.occ.shape[1:] == (256, 512), f"Expected (256, 512), got {self.occ.shape[1:]}"

    def __len__(self):
        return self.radar.shape[0]

    def __getitem__(self, idx):
        radar_frame = self.radar[idx].clone()
        lidar_frame = self.lidar[idx]  # (8192, 3) original lidar for eval
        occ_label = self.occ[idx]  # (256, 512) rasterized occupancy for loss
        norm_factor = self.norm_factors[idx]

        if self.augment:
            radar_frame = self._augment(radar_frame)

        return radar_frame, lidar_frame, occ_label, norm_factor

    def _augment(self, r):
        """Same IQ augmentation as TrajectoryDataset."""
        theta = random.uniform(0.0, 2.0 * torch.pi)
        r = r * torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))
        noise = self.noise_sigma * (
            torch.randn_like(r.real) + 1j * torch.randn_like(r.imag)
        ).to(torch.complex64)
        r = r + noise
        shift = random.randint(-2, 2)
        r = torch.roll(r, shift, dims=-1)
        return r
```

Also add a `build_occupancy_dataloaders` function following the same pattern as `build_dataloaders` but using `OccupancyTrajectoryDataset` and checking for `occ_{tid}.pt` files.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/data/tests/test_occupancy_dataset.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add v2/data/dataset.py v2/data/tests/test_occupancy_dataset.py
git commit -m "feat(v2): add OccupancyTrajectoryDataset for polar occupancy training"
```

---

## Task 3: Occupancy Model (Channelizer + Dilated Residual Head)

The core model: LISTA -> channelize -> dilated conv head -> occupancy logits.

**Files:**
- Create: `v2/model/occupancy.py`
- Create: `v2/model/tests/test_occupancy.py`
- Modify: `v2/model/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
# v2/model/tests/test_occupancy.py
import torch
import pytest


def test_channelizer_output_shape():
    from v2.model.occupancy import Channelizer
    ch = Channelizer()
    x = torch.randn(2, 256, 512, dtype=torch.complex64)
    out = ch(x)
    assert out.shape == (2, 3, 256, 512), f"Expected (2,3,256,512), got {out.shape}"
    assert out.dtype == torch.float32


def test_channelizer_preserves_info():
    """Channelizer output should have 3 channels with finite values."""
    from v2.model.occupancy import Channelizer
    ch = Channelizer()
    x = torch.randn(2, 256, 512, dtype=torch.complex64)
    out = ch(x)
    assert out.shape == (2, 3, 256, 512)
    assert torch.isfinite(out).all(), "Output should be all finite"
    # After LayerNorm, channels should be roughly zero-mean unit-var
    assert out.mean().abs() < 0.5, f"Mean should be near zero: {out.mean()}"


def test_dilated_res_head_output_shape():
    from v2.model.occupancy import DilatedResHead
    head = DilatedResHead(in_ch=3, mid_ch=32, n_blocks=3)
    x = torch.randn(2, 3, 256, 512)
    out = head(x)
    assert out.shape == (2, 1, 256, 512), f"Expected (2,1,256,512), got {out.shape}"


def test_occupancy_model_end_to_end():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="fft")
    x = torch.randn(2, 8, 512, dtype=torch.complex64)
    logits = model(x)
    assert logits.shape == (2, 1, 256, 512)
    assert logits.dtype == torch.float32


def test_occupancy_model_param_count():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="fft", mid_ch=32, n_blocks=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 1_000_000, f"Model too large: {n_params} params (budget: <1M)"
    print(f"Occupancy model params: {n_params:,}")


def test_occupancy_model_lista():
    from v2.model.occupancy import OccupancyModel
    model = OccupancyModel(beamformer="lista", K=3)
    x = torch.randn(2, 8, 512, dtype=torch.complex64)
    logits = model(x)
    assert logits.shape == (2, 1, 256, 512)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/model/tests/test_occupancy.py -v`
Expected: ImportError

- [ ] **Step 3: Implement Channelizer**

```python
# v2/model/occupancy.py
"""Polar occupancy decoder for mmDar v2.

Replaces the point cloud template decoder with a topology-preserving
polar occupancy head. The LISTA beamformer's 2D (azimuth x range)
structure is maintained throughout — no angular collapse.

Architecture:
    LISTA/FFT -> (B, 256, 512) complex
    Channelizer -> (B, 3, 256, 512) real [Re, Im, log_power]
    DilatedResHead -> (B, 1, 256, 512) occupancy logits

The output is logits (pre-sigmoid). Apply sigmoid only at inference.
"""

import torch
import torch.nn as nn


class Channelizer(nn.Module):
    """Convert complex angular spectrum to real-valued feature channels.

    Channels: [Re(x), Im(x), log(|x|^2 + eps)]
    Each channel is independently normalized via LayerNorm over spatial dims.

    The log-power channel provides magnitude information on a
    perceptually-linear scale, while Re/Im preserve full phase.
    Per-channel LayerNorm ensures Re/Im and log_power are on comparable
    scales regardless of beamformer amplitude (FFT vs LISTA).
    """

    def __init__(self, N_az: int = 256, N_r: int = 512, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # InstanceNorm2d: normalizes each channel independently over (H, W)
        # This ensures Re, Im, log_power are on comparable scales without
        # mixing information across channels.
        self.norm = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, A, R) complex64 angular spectrum from beamformer

        Returns:
            (B, 3, A, R) float32 channels [Re, Im, log_power], normalized
        """
        re = x.real  # (B, A, R)
        im = x.imag  # (B, A, R)
        log_pow = torch.log(re**2 + im**2 + self.eps)  # (B, A, R)
        out = torch.stack([re, im, log_pow], dim=1)  # (B, 3, A, R)
        return self.norm(out)
```

- [ ] **Step 4: Implement DilatedResBlock and DilatedResHead**

Add to `v2/model/occupancy.py`:

```python
class DilatedResBlock(nn.Module):
    """Residual block with dilated 2D convolution.

    Conv2d(dilation=d) -> GroupNorm -> ReLU -> Conv2d(dilation=1) -> GroupNorm
    + residual connection.
    """

    def __init__(self, ch: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DilatedResHead(nn.Module):
    """Lightweight dilated residual conv head for polar occupancy.

    Input projection (in_ch -> mid_ch) followed by N residual blocks
    with increasing dilation (1, 2, 4, 1, 2, 4, ...) for multi-scale
    receptive field. Final 1x1 conv to 1-channel logits.

    Args:
        in_ch:    Input channels (default 3 for [Re, Im, log_power])
        mid_ch:   Hidden channel width (default 32)
        n_blocks: Number of residual blocks (default 4)
        groups:   GroupNorm groups (default 8, must divide mid_ch)
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 32, n_blocks: int = 4,
                 groups: int = 8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, mid_ch),
            nn.ReLU(inplace=True),
        )

        dilations = [1, 2, 4]
        blocks = []
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(DilatedResBlock(mid_ch, dilation=d, groups=groups))
        self.blocks = nn.Sequential(*blocks)

        self.output_proj = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, A, R) real feature map

        Returns:
            (B, 1, A, R) occupancy logits (pre-sigmoid)
        """
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)
```

- [ ] **Step 5: Implement OccupancyModel assembly**

Add to `v2/model/occupancy.py`:

```python
from v2.model.lista import FFTBeamformer, LISTABeamformer


class OccupancyModel(nn.Module):
    """Full occupancy prediction model: beamformer + channelizer + conv head.

    Args:
        beamformer: "fft" or "lista" (default "fft")
        K:          LISTA unrolling depth (only used when beamformer="lista")
        N_az:       Angular bins (default 256)
        mid_ch:     Conv head hidden channels (default 32)
        n_blocks:   Number of residual blocks (default 4)
    """

    def __init__(self, beamformer: str = "fft", K: int = 5, N_az: int = 256,
                 mid_ch: int = 32, n_blocks: int = 4):
        super().__init__()
        if beamformer == "lista":
            self.beamformer = LISTABeamformer(K=K, N_az=N_az)
        else:
            self.beamformer = FFTBeamformer(N_az=N_az)

        self.channelizer = Channelizer()
        self.head = DilatedResHead(in_ch=3, mid_ch=mid_ch, n_blocks=n_blocks)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, 8, 512) complex64 raw radar input

        Returns:
            (B, 1, 256, 512) float32 occupancy logits
        """
        spec = self.beamformer(y)       # (B, N_az, 512) complex
        features = self.channelizer(spec)  # (B, 3, N_az, 512) real
        logits = self.head(features)    # (B, 1, N_az, 512)
        return logits
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/model/tests/test_occupancy.py -v`
Expected: All 6 tests PASS

- [ ] **Step 7: Export from __init__.py**

Add to `v2/model/__init__.py`:

```python
from v2.model.occupancy import OccupancyModel, Channelizer, DilatedResHead
```

- [ ] **Step 8: Commit**

```bash
git add v2/model/occupancy.py v2/model/tests/test_occupancy.py v2/model/__init__.py
git commit -m "feat(v2): add polar occupancy model with dilated residual head"
```

---

## Task 4: Occupancy Loss (Focal BCE + Dice)

**Files:**
- Create: `v2/train/loss_occupancy.py`
- Create: `v2/train/tests/test_loss_occupancy.py`

- [ ] **Step 1: Write failing tests**

```python
# v2/train/tests/test_loss_occupancy.py
import torch
import pytest


def test_focal_bce_zero_on_perfect():
    from v2.train.loss_occupancy import focal_bce_loss
    logits = torch.tensor([[[[10.0, -10.0]]]])  # confident correct
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = focal_bce_loss(logits, target)
    assert loss.item() < 0.01, f"Perfect prediction should have near-zero loss: {loss}"


def test_focal_bce_high_on_wrong():
    from v2.train.loss_occupancy import focal_bce_loss
    logits = torch.tensor([[[[-10.0, 10.0]]]])  # confident wrong
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = focal_bce_loss(logits, target)
    assert loss.item() > 1.0, f"Wrong prediction should have high loss: {loss}"


def test_dice_loss_range():
    from v2.train.loss_occupancy import dice_loss
    logits = torch.randn(2, 1, 256, 512)
    target = (torch.rand(2, 1, 256, 512) > 0.99).float()  # sparse
    loss = dice_loss(logits, target)
    assert 0.0 <= loss.item() <= 1.0, f"Dice loss should be in [0,1]: {loss}"


def test_occupancy_loss_composite():
    from v2.train.loss_occupancy import occupancy_loss
    logits = torch.randn(2, 1, 256, 512)
    target = (torch.rand(2, 1, 256, 512) > 0.99).float()
    losses = occupancy_loss(logits, target)
    assert "total" in losses
    assert "focal_bce" in losses
    assert "dice" in losses
    assert losses["total"].requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/test_loss_occupancy.py -v`

- [ ] **Step 3: Implement loss functions**

```python
# v2/train/loss_occupancy.py
"""Occupancy prediction losses for mmDar v2.

Combines focal BCE (handles class imbalance) with Dice loss (overlap metric).
Labels are polar occupancy grids: ~0.8% positive pixels (very sparse).

total = focal_bce + dice_weight * dice
"""

import torch
import torch.nn.functional as F


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal binary cross-entropy for sparse occupancy.

    Focal loss down-weights easy negatives so the model focuses on
    hard positives (occupied cells at the noise floor).

    Args:
        logits: (B, 1, A, R) pre-sigmoid predictions
        target: (B, 1, A, R) ground truth in [0, 1]
        alpha:  Weight for positive class (default 0.25)
        gamma:  Focusing exponent (default 2.0)

    Returns:
        Scalar focal BCE loss
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_weight = alpha_t * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss for occupancy overlap.

    Args:
        logits: (B, 1, A, R) pre-sigmoid
        target: (B, 1, A, R) ground truth
        smooth: Smoothing constant to avoid division by zero

    Returns:
        Scalar 1 - Dice coefficient
    """
    pred = torch.sigmoid(logits)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def occupancy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Composite occupancy loss: focal BCE + Dice.

    Args:
        logits:       (B, 1, A, R) pre-sigmoid predictions
        target:       (B, 1, A, R) ground truth
        dice_weight:  Weight for Dice loss (default 1.0)
        focal_alpha:  Focal loss alpha
        focal_gamma:  Focal loss gamma

    Returns:
        Dict with 'total', 'focal_bce', 'dice' (all scalar tensors)
    """
    f_bce = focal_bce_loss(logits, target, focal_alpha, focal_gamma)
    d_loss = dice_loss(logits, target)
    total = f_bce + dice_weight * d_loss
    return {"total": total, "focal_bce": f_bce, "dice": d_loss}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/test_loss_occupancy.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add v2/train/loss_occupancy.py v2/train/tests/test_loss_occupancy.py
git commit -m "feat(v2): add focal BCE + Dice occupancy loss"
```

---

## Task 5: Occupancy Eval Adapter

Converts polar occupancy maps to Cartesian point clouds for Chamfer/mod-H evaluation.

**Files:**
- Create: `v2/eval/occupancy_eval.py`
- Create: `v2/eval/tests/test_occupancy_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# v2/eval/tests/test_occupancy_eval.py
import numpy as np
import torch
import pytest


def test_occupancy_to_points_broadside():
    """Single occupied cell at broadside -> point near (r, 0)."""
    from v2.eval.occupancy_eval import occupancy_to_points
    occ = np.zeros((256, 512), dtype=np.float32)
    r_bin = 236  # ~5.0m
    az_bin = 128  # sin(theta)~0 -> broadside
    occ[az_bin, r_bin] = 1.0
    pts = occupancy_to_points(occ, threshold=0.5, r_max=10.8)
    assert pts.shape[1] == 3
    assert len(pts) == 1
    # x ~ 5.0, y ~ 0.0, z = 0
    assert abs(pts[0, 0] - 5.0) < 0.1, f"x={pts[0,0]}, expected ~5.0"
    assert abs(pts[0, 1]) < 0.1, f"y={pts[0,1]}, expected ~0.0"


def test_occupancy_to_points_empty():
    from v2.eval.occupancy_eval import occupancy_to_points
    occ = np.zeros((256, 512), dtype=np.float32)
    pts = occupancy_to_points(occ, threshold=0.5)
    assert len(pts) == 0 or pts.shape == (0, 3)


def test_evaluate_occupancy_epoch_smoke():
    """Smoke test: model outputs logits, eval produces metrics dict."""
    from v2.eval.occupancy_eval import evaluate_occupancy_epoch
    # Minimal mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.zeros(B, 1, 256, 512)

    model = MockModel()
    # Minimal dataloader: 1 batch of (radar, lidar_pts, occ_label, norm)
    radar = torch.randn(2, 8, 512, dtype=torch.complex64)
    lidar = torch.zeros(2, 8192, 3)
    lidar[0, 0] = torch.tensor([5.0, 0.0, 0.0])  # one valid point
    lidar[1, 0] = torch.tensor([5.0, 0.0, 0.0])
    occ = torch.zeros(2, 256, 512)
    norm = torch.ones(2)
    loader = [(radar, lidar, occ, norm)]

    metrics = evaluate_occupancy_epoch(model, loader, torch.device("cpu"))
    assert "chamfer" in metrics
    assert "mod_hausdorff" in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/eval/tests/test_occupancy_eval.py -v`

- [ ] **Step 3: Implement occupancy_to_points and evaluate_occupancy_epoch**

```python
# v2/eval/occupancy_eval.py
"""Polar occupancy evaluation: convert occupancy maps to point clouds and
compute Chamfer distance and modified Hausdorff distance.

Uses LISTA's angular grid convention:
    sin_theta[k] = -1 + 2*k/(N_az-1)
    range[j] = j * r_max / (N_r-1)
    x = range * cos(theta), y = range * sin(theta), z = 0
"""

import numpy as np
import torch
from v2.eval.eval_adapter import chamfer_distance_np, mod_hausdorff_np
from v2.data.rasterize import rasterize_to_polar


def occupancy_to_points(
    occ: np.ndarray,
    threshold: float = 0.5,
    r_max: float = 10.8,
) -> np.ndarray:
    """Convert polar occupancy grid to XYZ point cloud.

    Args:
        occ: (N_az, N_r) float32 occupancy probabilities in [0, 1]
        threshold: Detection threshold
        r_max: Maximum range in meters

    Returns:
        (N_pts, 3) float32 Cartesian point cloud [x, y, z]
    """
    N_az, N_r = occ.shape
    az_bins, r_bins = np.where(occ > threshold)

    if len(az_bins) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # LISTA grid: sin_theta = -1 + 2*k/(N_az-1)
    sin_theta = -1.0 + 2.0 * az_bins / (N_az - 1)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    cos_theta = np.sqrt(1.0 - sin_theta**2)

    r = r_bins * r_max / (N_r - 1)

    x = r * cos_theta
    y = r * sin_theta
    z = np.zeros_like(x)

    return np.stack([x, y, z], axis=1).astype(np.float32)


MAX_PENALTY_DIST = 20.0  # meters — penalty for empty predictions


def evaluate_occupancy_epoch(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    threshold: float = 0.5,
    r_max: float = 10.8,
) -> dict:
    """Run occupancy model and compute point cloud metrics.

    CRITICAL: Predictions are compared against ORIGINAL lidar point clouds
    (from the dataset), NOT against re-rasterized occupancy labels. This
    ensures metrics are comparable to the baseline and not polluted by
    rasterization artifacts or label softening.

    For each sample:
    1. Model predicts logits -> sigmoid -> threshold -> pred point cloud
    2. GT is the original lidar (8192, 3) XY-only point cloud
    3. Compute Chamfer and mod-Hausdorff in Cartesian XY space

    Empty predictions (no cells above threshold) receive a penalty
    distance of MAX_PENALTY_DIST meters, NOT skipped.

    Args:
        model:      OccupancyModel with forward(radar) -> (B,1,A,R) logits
        dataloader: yields (radar, lidar_pts, occ_label, norm) batches
        device:     compute device
        threshold:  occupancy detection threshold

    Returns:
        dict with 'chamfer', 'mod_hausdorff', 'n_samples'
    """
    model.eval()
    chamfer_sum = 0.0
    hausdorff_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            radar, lidar_gt, _occ_label, _norm = batch
            radar = radar.to(device)
            logits = model(radar)  # (B, 1, A, R)
            pred_occ = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, A, R)
            gt_pts_batch = lidar_gt.numpy()  # (B, 8192, 3) original lidar

            for i in range(pred_occ.shape[0]):
                pred_pts = occupancy_to_points(pred_occ[i], threshold, r_max)
                gt_pts = gt_pts_batch[i]  # (8192, 3) original lidar

                # Filter GT to scene volume (x>0, r<=r_max) for fair comparison
                r_gt = np.sqrt(gt_pts[:, 0]**2 + gt_pts[:, 1]**2)
                valid = (gt_pts[:, 0] > 0) & (r_gt <= r_max) & (r_gt > 0.01)
                gt_pts = gt_pts[valid]

                if len(gt_pts) == 0:
                    continue  # no valid GT -> skip (not model's fault)

                if len(pred_pts) == 0:
                    # Penalty for predicting nothing
                    chamfer_sum += MAX_PENALTY_DIST
                    hausdorff_sum += MAX_PENALTY_DIST
                else:
                    chamfer_sum += chamfer_distance_np(pred_pts, gt_pts)
                    hausdorff_sum += mod_hausdorff_np(pred_pts, gt_pts)
                n_samples += 1

    if n_samples == 0:
        return {"chamfer": float("nan"), "mod_hausdorff": float("nan"), "n_samples": 0}

    return {
        "chamfer": chamfer_sum / n_samples,
        "mod_hausdorff": hausdorff_sum / n_samples,
        "n_samples": n_samples,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/eval/tests/test_occupancy_eval.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add v2/eval/occupancy_eval.py v2/eval/tests/test_occupancy_eval.py
git commit -m "feat(v2): add occupancy eval adapter with polar-to-Cartesian conversion"
```

---

## Task 6: Preprocessing — Generate Occupancy Labels

Run rasterization on all processed lidar .pt files to generate occ_{traj}.pt files.

**Files:**
- Modify: `v2/data/rasterize.py` (add CLI entry point)

- [ ] **Step 1: Add batch rasterization CLI to rasterize.py**

Add to `v2/data/rasterize.py`:

```python
def rasterize_trajectory(
    lidar_pt_path: str,
    output_path: str,
    N_az: int = 256,
    N_r: int = 512,
    r_max: float = 10.8,
    sigma: float = 0.5,
) -> int:
    """Rasterize all frames in a lidar .pt file to polar occupancy.

    Args:
        lidar_pt_path: Path to lidar_{traj}.pt, shape (N, 8192, 3)
        output_path:   Path to write occ_{traj}.pt, shape (N, 256, 512)
        sigma:         Gaussian softening (0.5 bins default)

    Returns:
        Number of frames processed
    """
    import torch
    lidar = torch.load(lidar_pt_path, weights_only=True).numpy()
    N = lidar.shape[0]
    occ_list = []
    for i in range(N):
        occ_list.append(rasterize_to_polar(lidar[i], N_az, N_r, r_max, sigma))
    occ = np.stack(occ_list)
    torch.save(torch.from_numpy(occ), output_path)
    return N


if __name__ == "__main__":
    import argparse
    import glob
    import os
    import torch

    parser = argparse.ArgumentParser(description="Rasterize lidar .pt to occupancy")
    parser.add_argument("--processed-dir", default="v2/data/processed")
    parser.add_argument("--sigma", type=float, default=0.5)
    args = parser.parse_args()

    lidar_files = sorted(glob.glob(os.path.join(args.processed_dir, "lidar_*.pt")))
    print(f"Found {len(lidar_files)} lidar .pt files")

    for lf in lidar_files:
        traj_id = os.path.basename(lf).replace("lidar_", "").replace(".pt", "")
        out_path = os.path.join(args.processed_dir, f"occ_{traj_id}.pt")
        if os.path.exists(out_path):
            print(f"  occ_{traj_id}.pt exists, skipping")
            continue
        n = rasterize_trajectory(lf, out_path, sigma=args.sigma)
        print(f"  occ_{traj_id}.pt: {n} frames rasterized")
```

- [ ] **Step 2: Run rasterization in Docker**

```bash
docker compose run --rm mmdar python3 -m v2.data.rasterize --processed-dir v2/data/processed --sigma 0.5
```

Expected: occ_{traj_id}.pt files created for all 44 trajectories.

- [ ] **Step 3: Verify a sample occ file**

```bash
docker compose run --rm mmdar python3 -c "
import torch
occ = torch.load('v2/data/processed/occ_112.pt', weights_only=True)
print(f'Shape: {occ.shape}, dtype: {occ.dtype}')
print(f'Nonzero fraction: {(occ > 0).float().mean():.4f}')
print(f'Max: {occ.max():.3f}, Mean of nonzero: {occ[occ > 0].mean():.3f}')
"
```

Expected: Shape (N_frames, 256, 512), dtype float32, ~0.5-2% nonzero.

- [ ] **Step 4: Commit rasterize.py update (NOT the .pt files)**

```bash
git add v2/data/rasterize.py
git commit -m "feat(v2): add batch rasterization CLI for occupancy labels"
```

---

## Task 7: Training Script

**Files:**
- Create: `v2/train/train_occupancy.py`

- [ ] **Step 1: Write training script**

Follows the same structure as `v2/train/train.py` but uses OccupancyModel and occupancy_loss. Key differences:
- Model: OccupancyModel(beamformer="fft" or "lista")
- Loss: occupancy_loss (focal BCE + Dice)
- Eval: evaluate_occupancy_epoch (occupancy -> point cloud -> Chamfer/mod-H)
- No confidence loss, no DCD, no coverage hinge
- Same training recipe: AdamW, cosine LR, warmup, early stopping on val Chamfer

```python
# v2/train/train_occupancy.py
"""Training script for polar occupancy model.

Usage:
    python3 v2/train/train_occupancy.py --log-dir logs/v2_occ_fft --model-type fft
    python3 v2/train/train_occupancy.py --log-dir logs/v2_occ_lista --model-type lista
"""
# [Full training script following the exact pattern of v2/train/train.py]
# Key changes:
# - Uses OccupancyModel instead of RadarPointCloudModel
# - Uses build_occupancy_dataloaders
# - Uses occupancy_loss instead of composite_loss
# - Uses evaluate_occupancy_epoch instead of evaluate_epoch
# - Occupancy labels: (B, 256, 512) -> unsqueeze to (B, 1, 256, 512) for loss
# - Logs: train/total, train/focal_bce, train/dice, val/chamfer, val/mod_hausdorff
# - Checkpoint selection by val Chamfer (same as before)
```

The training script should be ~200 lines following the exact pattern of `v2/train/train.py` (lines 73-401) with the model/loss/eval substitutions above.

- [ ] **Step 2: Smoke test (1 epoch, CPU)**

```bash
cd /git/mmDar && python -m pytest -x --timeout=60 -k "test_training" v2/train/tests/
```

Or run directly:
```bash
docker compose run --rm mmdar python3 v2/train/train_occupancy.py \
  --epochs 2 --batch-size 4 --log-dir logs/v2_occ_smoke --model-type fft
```

- [ ] **Step 3: Commit**

```bash
git add v2/train/train_occupancy.py
git commit -m "feat(v2): add occupancy training script"
```

---

## Task 8: First Experiment — FFT Baseline Occupancy

Run the occupancy model with FFT beamformer to validate the approach before LISTA.

- [ ] **Step 1: Train FFT occupancy model**

```bash
docker compose run --rm mmdar python3 v2/train/train_occupancy.py \
  --model-type fft --batch-size 12 --lr 7e-5 --epochs 50 \
  --log-dir logs/v2_occ_fft --checkpoint-every 10
```

- [ ] **Step 2: Evaluate and compare**

Expected metrics to compare against:
| Model | Chamfer | Mod-H | Notes |
|-------|---------|-------|-------|
| Baseline (41-frame PNG) | 0.295 | 0.189 | Our reproduced best |
| v2 MagnitudeBaseline | 0.317 | 0.399 | Point decoder, 1 frame |
| v2 Mag+Phase | 0.309 | 0.423 | Point decoder, 1 frame |
| **v2 FFT Occupancy** | **???** | **???** | **Occupancy decoder, 1 frame** |

The key metric: does mod-Hausdorff drop from ~0.4 toward ~0.2? If yes, the topology fix is confirmed.

- [ ] **Step 3: Update results/README.md**

---

## Task 9 (Future): Ablations

Only after Task 8 validates the approach. NOT part of the initial implementation.

1. **LISTA beamformer**: Replace FFT with LISTA, freeze initially, fine-tune
2. **Complex stem**: Add 1-2 ComplexConv2d layers before channelization
3. **Coordinate channels**: Add normalized range + azimuth as extra input channels
4. **Label softening**: Sweep sigma in {0, 0.25, 0.5, 1.0}
5. **Threshold tuning**: Sweep detection threshold on val set against Chamfer/mod-H
6. **Temporal fusion**: Stack 3-5 frames as extra channels (after occupancy validated)

---

## Success Criteria

1. mod-Hausdorff drops from ~0.4 to <0.25 (proves topology fix works)
2. Chamfer stays at or below 0.32 (no regression from existing single-frame)
3. Model stays under 1M trainable params
4. Inference latency under 3ms per frame
5. All tests pass, no changes to existing v2 modules (except dataset.py addition)
