# Physics-Informed Radar Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resolution-aware polar recall loss (Tversky) and radar positive-support loss to the single-frame MagnitudePhaseFusion model, targeting the mod-Hausdorff coverage gap.

**Architecture:** Differentiable soft-splatting (bilinear binning + separable Gaussian blur + exponential bounding) converts predicted/GT points to polar occupancy grids. Tversky loss penalizes missing GT coverage. Radar support loss penalizes uncovered strong-return cells.

**Tech Stack:** PyTorch (conv2d for Gaussian blur, scatter for soft-binning). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-26-physics-losses-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `v2/train/loss_physics.py` | soft_splat, ra_recall_loss, radar_support_loss |
| Create | `v2/train/tests/test_loss_physics.py` | Unit tests for all physics loss components |
| Modify | `v2/model/__init__.py` | Add forward_with_intermediates to MagnitudePhaseFusion |
| Modify | `v2/train/loss.py` | Add physics losses to composite_loss |
| Modify | `v2/train/train.py` | Pass beamformer_power through training loop |

---

## Task 1: Differentiable Soft-Splatting

The core primitive: convert point cloud (B, N, 3) to bounded polar occupancy (B, 1, N_az, N_r).

**Files:**
- Create: `v2/train/loss_physics.py`
- Create: `v2/train/tests/test_loss_physics.py`

- [ ] **Step 1: Write failing tests for soft_splat**

```python
# v2/train/tests/test_loss_physics.py
import torch
import pytest


def test_soft_splat_output_shape():
    from v2.train.loss_physics import soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0], [3.0, 2.0, 0.0]]])  # (1, 2, 3)
    occ = soft_splat(pts)
    assert occ.shape == (1, 1, 256, 512), f"Expected (1,1,256,512), got {occ.shape}"


def test_soft_splat_bounded():
    """Output must be in [0, 1) due to 1 - exp(-I) bounding."""
    from v2.train.loss_physics import soft_splat
    # Dense point cloud — many overlapping points
    pts = torch.randn(2, 8192, 3) * 3 + torch.tensor([5.0, 0.0, 0.0])
    pts[..., 0] = pts[..., 0].abs()  # x > 0
    occ = soft_splat(pts)
    assert occ.min() >= 0.0, f"Min below 0: {occ.min()}"
    assert occ.max() < 1.0, f"Max >= 1: {occ.max()}"


def test_soft_splat_empty():
    from v2.train.loss_physics import soft_splat
    pts = torch.zeros(1, 0, 3)
    occ = soft_splat(pts)
    assert occ.sum() == 0


def test_soft_splat_gradient_flows():
    """Gradients must flow back to point positions."""
    from v2.train.loss_physics import soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0]]], requires_grad=True)
    occ = soft_splat(pts)
    loss = occ.sum()
    loss.backward()
    assert pts.grad is not None
    assert pts.grad.abs().sum() > 0, "No gradient to point positions"


def test_soft_splat_broadside_peak():
    """Point at (5, 0, 0) should peak near broadside azimuth."""
    from v2.train.loss_physics import soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0]]])
    occ = soft_splat(pts)
    occ_2d = occ[0, 0]  # (256, 512)
    # Broadside: az_bin ~128, r_bin ~236
    peak_az, peak_r = divmod(occ_2d.argmax().item(), 512)
    assert 120 < peak_az < 136, f"Peak azimuth {peak_az}, expected ~128"
    assert 230 < peak_r < 242, f"Peak range {peak_r}, expected ~236"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/test_loss_physics.py -v`

- [ ] **Step 3: Implement soft_splat**

```python
# v2/train/loss_physics.py
"""Physics-informed radar losses for mmDar v2.

Provides differentiable soft-splatting of point clouds into polar occupancy
grids, plus Tversky recall loss and radar positive-support loss.

Physics parameters (IWR1443, 8-element λ/2 ULA, 4GHz bandwidth):
    σ_r = 1.0 bins (~2.1cm, from range resolution c/(2B) = 3.75cm)
    σ_u = 14 bins (fixed in sin-theta space, from angular resolution 2/N)
    Grid: 256 azimuth (sin_theta ∈ [-1,1]) × 512 range ([0, 10.8m])
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Create a 1D Gaussian kernel (normalized to sum=1)."""
    radius = int(math.ceil(sigma * truncate))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


class SoftSplat(nn.Module):
    """Differentiable soft-splatting: pts (B, N, 3) → O (B, 1, H, W) ∈ [0, 1).

    Pipeline:
        1. Bilinear soft-binning onto (H, W) intensity grid
        2. Separable Gaussian blur (σ_u along azimuth, σ_r along range)
        3. O = 1 - exp(-I)  — bounds to [0, 1)

    Args:
        N_az: azimuth bins (default 256)
        N_r: range bins (default 512)
        r_max: max range meters (default 10.8)
        sigma_r: range PSF sigma in bins (default 1.0)
        sigma_u: azimuth PSF sigma in bins (default 14.0)
    """

    def __init__(self, N_az=256, N_r=512, r_max=10.8, sigma_r=1.0, sigma_u=14.0):
        super().__init__()
        self.N_az = N_az
        self.N_r = N_r
        self.r_max = r_max

        # Pre-compute separable Gaussian blur kernels as registered buffers
        kr = _make_gaussian_kernel_1d(sigma_r)
        ku = _make_gaussian_kernel_1d(sigma_u)
        # Conv2d format: (out_ch, in_ch/groups, kH, kW)
        self.register_buffer("kernel_r", kr.view(1, 1, 1, -1))  # blur along range (W)
        self.register_buffer("kernel_u", ku.view(1, 1, -1, 1))  # blur along azimuth (H)
        self.pad_r = kr.shape[0] // 2
        self.pad_u = ku.shape[0] // 2

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: (B, N, 3) float32 point cloud [x, y, z]
        Returns:
            (B, 1, N_az, N_r) float32 occupancy in [0, 1)
        """
        B = pts.shape[0]
        device = pts.device
        intensity = torch.zeros(B, 1, self.N_az, self.N_r, device=device)

        if pts.shape[1] == 0:
            return intensity

        x, y = pts[..., 0], pts[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        u = y / (r + 1e-8)  # sin(theta)

        # Filter to valid points (forward-facing, within range)
        valid = (x > 0.01) & (r <= self.r_max) & (u.abs() <= 1.0)

        # Continuous bin coordinates
        r_coord = r / self.r_max * (self.N_r - 1)  # [0, N_r-1]
        u_coord = (u + 1.0) * (self.N_az - 1) / 2.0  # [0, N_az-1]

        # Bilinear soft-binning: distribute each point to 4 nearest cells
        r_floor = r_coord.long().clamp(0, self.N_r - 2)
        u_floor = u_coord.long().clamp(0, self.N_az - 2)
        r_frac = (r_coord - r_floor.float()).clamp(0, 1)
        u_frac = (u_coord - u_floor.float()).clamp(0, 1)

        # Zero out invalid points
        mask = valid.float()
        w00 = (1 - r_frac) * (1 - u_frac) * mask
        w01 = r_frac * (1 - u_frac) * mask
        w10 = (1 - r_frac) * u_frac * mask
        w11 = r_frac * u_frac * mask

        # Vectorized scatter-add using flat linear indices (no per-batch loop)
        # Linear index: batch_offset + u * N_r + r
        batch_idx = torch.arange(B, device=device).view(B, 1).expand_as(r_floor)
        flat_size = self.N_az * self.N_r
        base = batch_idx * flat_size  # (B, N)

        intensity_flat = intensity.view(B, -1)  # (B, N_az * N_r)
        idx00 = base + u_floor * self.N_r + r_floor
        idx01 = base + u_floor * self.N_r + (r_floor + 1).clamp(max=self.N_r - 1)
        idx10 = base + (u_floor + 1).clamp(max=self.N_az - 1) * self.N_r + r_floor
        idx11 = base + (u_floor + 1).clamp(max=self.N_az - 1) * self.N_r + (r_floor + 1).clamp(max=self.N_r - 1)

        intensity_flat.scatter_add_(1, idx00, w00)
        intensity_flat.scatter_add_(1, idx01, w01)
        intensity_flat.scatter_add_(1, idx10, w10)
        intensity_flat.scatter_add_(1, idx11, w11)
        intensity = intensity_flat.view(B, 1, self.N_az, self.N_r)

        # Separable Gaussian blur
        intensity = F.conv2d(intensity, self.kernel_u, padding=(self.pad_u, 0))
        intensity = F.conv2d(intensity, self.kernel_r, padding=(0, self.pad_r))

        # Bound to [0, 1)
        return 1.0 - torch.exp(-intensity)


# Module-level convenience function
_default_splat = None

def soft_splat(pts, N_az=256, N_r=512, r_max=10.8, sigma_r=1.0, sigma_u=14.0):
    """Functional API for soft-splatting. Creates module on first call."""
    global _default_splat
    if _default_splat is None or _default_splat.N_az != N_az:
        _default_splat = SoftSplat(N_az, N_r, r_max, sigma_r, sigma_u)
    # Move buffers to match input device
    if _default_splat.kernel_r.device != pts.device:
        _default_splat = _default_splat.to(pts.device)
    return _default_splat(pts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/test_loss_physics.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add v2/train/loss_physics.py v2/train/tests/test_loss_physics.py
git commit -m "feat(v2): add differentiable soft-splatting for polar occupancy"
```

---

## Task 2: Tversky Recall Loss + Radar Support Loss

Add the two physics-informed loss functions on top of soft_splat.

**Files:**
- Modify: `v2/train/loss_physics.py` (add functions)
- Modify: `v2/train/tests/test_loss_physics.py` (add tests)

- [ ] **Step 1: Write failing tests**

```python
# Add to v2/train/tests/test_loss_physics.py

def test_ra_recall_perfect_overlap():
    """Perfect overlap should give near-zero loss."""
    from v2.train.loss_physics import ra_recall_loss, soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0], [3.0, 2.0, 0.0]]])
    O_pred = soft_splat(pts)
    O_gt = soft_splat(pts.detach())
    loss = ra_recall_loss(O_pred, O_gt)
    assert loss.item() < 0.1, f"Perfect overlap should have low loss: {loss}"


def test_ra_recall_no_overlap():
    """No overlap should give high loss."""
    from v2.train.loss_physics import ra_recall_loss, soft_splat
    pts_pred = torch.tensor([[[2.0, 0.0, 0.0]]])
    pts_gt = torch.tensor([[[8.0, 5.0, 0.0]]])
    O_pred = soft_splat(pts_pred)
    O_gt = soft_splat(pts_gt.detach())
    loss = ra_recall_loss(O_pred, O_gt)
    assert loss.item() > 0.8, f"No overlap should have high loss: {loss}"


def test_ra_recall_gradient():
    from v2.train.loss_physics import ra_recall_loss, soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0]]], requires_grad=True)
    gt = torch.tensor([[[5.0, 1.0, 0.0]]])
    O_pred = soft_splat(pts)
    O_gt = soft_splat(gt)
    loss = ra_recall_loss(O_pred, O_gt)
    loss.backward()
    assert pts.grad is not None and pts.grad.abs().sum() > 0


def test_radar_support_loss_shape():
    from v2.train.loss_physics import radar_support_loss, soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0]]])
    O_pred = soft_splat(pts)
    power = torch.rand(1, 256, 512)  # fake beamformer power
    loss = radar_support_loss(O_pred, power)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_radar_support_zero_power():
    """Zero power = no radar-positive cells = zero loss."""
    from v2.train.loss_physics import radar_support_loss, soft_splat
    pts = torch.tensor([[[5.0, 0.0, 0.0]]])
    O_pred = soft_splat(pts)
    power = torch.zeros(1, 256, 512)
    loss = radar_support_loss(O_pred, power)
    assert loss.item() == 0.0
```

- [ ] **Step 2: Run tests to verify new ones fail**

- [ ] **Step 3: Implement ra_recall_loss and radar_support_loss**

Add to `v2/train/loss_physics.py`:

```python
def ra_recall_loss(
    O_pred: torch.Tensor,
    O_gt: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Resolution-aware Tversky recall loss in polar space.

    α=0.3 (FP lightly penalized), β=0.7 (FN heavily penalized).
    Operates on soft occupancy grids from soft_splat.

    Args:
        O_pred: (B, 1, H, W) predicted occupancy in [0, 1)
        O_gt:   (B, 1, H, W) GT occupancy in [0, 1) (detached)
        alpha:  false positive weight
        beta:   false negative weight (higher = more recall pressure)
        smooth: smoothing constant

    Returns:
        Scalar loss: 1 - Tversky index
    """
    O_pred = O_pred.squeeze(1)  # (B, H, W)
    O_gt = O_gt.squeeze(1)

    # Use relu-based FP/FN for correct soft-occupancy Tversky:
    # FP = where pred exceeds GT, FN = where GT exceeds pred
    # This gives zero loss for O_pred == O_gt (perfect overlap).
    FP = F.relu(O_pred - O_gt).sum(dim=(-2, -1))
    FN = F.relu(O_gt - O_pred).sum(dim=(-2, -1))
    intersection = (torch.min(O_pred, O_gt)).sum(dim=(-2, -1))

    tversky = (intersection + smooth) / (intersection + alpha * FP + beta * FN + smooth)
    return (1 - tversky).mean()


def radar_support_loss(
    O_pred: torch.Tensor,
    beamformer_power: torch.Tensor,
    k: float = 6.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Radar positive-support loss: coverage at strong-return cells.

    Where beamformer detects strong returns (heuristic threshold),
    at least one predicted point should cover that cell.

    beamformer_power MUST be linear power (|X|^2), not dB or magnitude.

    Args:
        O_pred:           (B, 1, H, W) predicted occupancy
        beamformer_power: (B, H, W) linear power from beamformer
        k:                threshold factor (default 6.0, ~0.25% false alarm)
        eps:              log stability

    Returns:
        Scalar cross-entropy loss at radar-positive cells
    """
    O_pred_sq = O_pred.squeeze(1)  # (B, H, W)

    # Heuristic threshold per range bin: compare each cell to azimuth mean
    mean_power = beamformer_power.mean(dim=1, keepdim=True)  # (B, 1, W)
    mask = (beamformer_power > mean_power * k).float()  # (B, H, W)

    n_positive = mask.sum()
    if n_positive == 0:
        return torch.tensor(0.0, device=O_pred.device)

    loss = -(mask * torch.log(O_pred_sq + eps)).sum() / n_positive.clamp_min(1.0)
    return loss
```

- [ ] **Step 4: Run all tests**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/test_loss_physics.py -v`
Expected: All 10 PASS

- [ ] **Step 5: Commit**

```bash
git add v2/train/loss_physics.py v2/train/tests/test_loss_physics.py
git commit -m "feat(v2): add Tversky recall loss and radar support loss"
```

---

## Task 3: Model forward_with_intermediates + Loss Integration

Wire physics losses into the training pipeline.

**Files:**
- Modify: `v2/model/__init__.py` — add forward_with_intermediates to MagnitudePhaseFusion
- Modify: `v2/train/loss.py` — add physics losses to composite_loss
- Modify: `v2/train/train.py` — pass beamformer_power through training loop

- [ ] **Step 1: Add forward_with_intermediates to MagnitudePhaseFusion**

Read `v2/model/__init__.py` and add after MagnitudePhaseFusion.forward():

```python
def forward_with_intermediates(self, y):
    """Forward pass returning beamformer power for physics losses.

    Returns:
        pts:  (B, 8192, 3) float32
        conf: (B, 8192, 1) float32
        beamformer_power: (B, N_az, 512) float32 — LINEAR power |X|²
    """
    spec = self.beamformer(y)           # (B, N_az, 512) complex
    mag = safe_modulus(spec)             # (B, N_az, 512) float — magnitude
    beamformer_power = mag ** 2          # (B, N_az, 512) float — LINEAR power
    # Shape is explicitly (B, N_az=256, R=512) = (B, H, W) — no squeeze needed

    phase = torch.angle(spec)
    sin_ph = torch.sin(phase)
    cos_ph = torch.cos(phase)
    gate = (mag > mag.mean(dim=1, keepdim=True) * 0.1).float()

    fused = torch.cat([mag, sin_ph * gate, cos_ph * gate], dim=1)
    features = self.bridge(fused)
    pts, conf = self.decoder(features)
    return pts, conf, beamformer_power
```

- [ ] **Step 2: Add physics losses to composite_loss**

Add two new optional parameters to `composite_loss()` in `v2/train/loss.py`:

```python
def composite_loss(
    pred_pts, gt_pts, conf_logits, epoch,
    # ... existing params ...
    beamformer_power: torch.Tensor | None = None,
    use_physics_loss: bool = False,
    physics_recall_weight: float = 0.15,
    physics_support_weight: float = 0.03,
) -> dict:
```

Inside the function, after existing losses:

```python
    # --- Physics losses ---
    if use_physics_loss:
        from v2.train.loss_physics import SoftSplat, ra_recall_loss, radar_support_loss
        # Cache SoftSplat instance as function attribute (created once per device)
        if not hasattr(composite_loss, '_splat') or composite_loss._splat.kernel_r.device != pred_pts.device:
            composite_loss._splat = SoftSplat().to(pred_pts.device)
        splat = composite_loss._splat
        O_pred = splat(pred_pts)
        O_gt = splat(gt_pts.detach())
        ra_loss = ra_recall_loss(O_pred, O_gt)
        if beamformer_power is not None:
            rs_loss = radar_support_loss(O_pred, beamformer_power)
        else:
            rs_loss = torch.tensor(0.0, device=pred_pts.device)
    else:
        ra_loss = torch.tensor(0.0, device=pred_pts.device)
        rs_loss = torch.tensor(0.0, device=pred_pts.device)

    total = (
        ch_loss + dcd_w * dc_loss + 0.1 * cov_loss + 0.01 * conf_l + 0.1 * mc_loss
        + physics_recall_weight * ra_loss
        + physics_support_weight * rs_loss
    )

    return {
        "total": total,
        "chamfer": ch_loss, "dcd": dc_loss, "coverage": cov_loss,
        "confidence": conf_l, "measurement_consistency": mc_loss,
        "ra_recall": ra_loss, "radar_support": rs_loss,
    }
```

- [ ] **Step 3: Update train.py to use physics losses**

In `v2/train/train.py`, inside the training loop (around line 222-248):

When model_type is "mag_phase", use `forward_with_intermediates`:

```python
if model_type == "mag_phase" and cfg.get("use_physics_loss", False):
    pts, conf, bf_power = model.forward_with_intermediates(radar)
else:
    pts, conf = model(radar)
    bf_power = None
```

Pass to composite_loss:

```python
losses = composite_loss(
    pts, lidar, conf, epoch,
    # ... existing args ...
    beamformer_power=bf_power,
    use_physics_loss=cfg.get("use_physics_loss", False),
    physics_recall_weight=cfg.get("physics_recall_weight", 0.15),
    physics_support_weight=cfg.get("physics_support_weight", 0.03),
)
```

Add TensorBoard logging for the new losses:

```python
writer.add_scalar("train/ra_recall", epoch_losses["ra_recall"] / n_batches, epoch)
writer.add_scalar("train/radar_support", epoch_losses["radar_support"] / n_batches, epoch)
```

Add CLI args:

```python
parser.add_argument("--use-physics-loss", action="store_true", default=False)
parser.add_argument("--physics-recall-weight", type=float, default=0.15)
parser.add_argument("--physics-support-weight", type=float, default=0.03)
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `cd /git/mmDar && python -m pytest v2/train/tests/ v2/model/tests/ -v --ignore=v2/train/tests/test_loss_physics.py`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add v2/model/__init__.py v2/train/loss.py v2/train/train.py
git commit -m "feat(v2): integrate physics losses into composite_loss and training loop"
```

---

## Task 4: Run Experiment

- [ ] **Step 1: Train with physics losses**

```bash
docker compose run --rm mmdar python3 -m v2.train.train \
  --model-type mag_phase --batch-size 12 --lr 7e-5 --epochs 50 \
  --log-dir logs/v2_mag_phase_physics \
  --use-physics-loss \
  --physics-recall-weight 0.15 \
  --physics-support-weight 0.03 \
  --checkpoint-every 10
```

- [ ] **Step 2: Compare against baseline single-frame (no physics)**

| Model | Chamfer (m) | Mod-H (m) | Notes |
|-------|-------------|-----------|-------|
| v2 Mag+Phase (no physics) | 0.309 | 0.423 | Previous best |
| v2 Mag+Phase + physics losses | ??? | ??? | This experiment |

Key metric: does mod-H improve? Even 0.40 → 0.38 would validate the approach.

- [ ] **Step 3: Report directed Chamfer separately**

After training, evaluate with directed Chamfer:
- pred→gt (precision): how close are predicted points to GT?
- gt→pred (recall): how close are GT points to predictions?

The physics losses should specifically improve gt→pred (recall direction).

- [ ] **Step 4: Update results/README.md with findings**

---

## Success Criteria

1. mod-H improves from 0.423 toward <0.40 (any improvement validates approach)
2. Chamfer stays ≤ 0.32 (no regression)
3. gt→pred directed Chamfer specifically improves (recall direction)
4. Training is stable (no divergence from physics losses)
5. All existing tests still pass
