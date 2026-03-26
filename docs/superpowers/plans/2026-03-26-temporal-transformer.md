# Temporal Cross-Attention Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-range-bin temporal cross-attention to the proven MagnitudePhaseFusion model, enabling multi-frame fusion that improves mod-Hausdorff from 0.423m toward the 41-frame baseline's 0.189m.

**Architecture:** Existing per-frame encoder (frozen from checkpoint) → shared bridge → (B, N, 128, 512) → temporal cross-attention residual adapter → (B, 128, 512) → unchanged PointCloudDecoder. Temporal block: 1 pre-LN cross-attention layer (d=128, 4 heads, ff=256), ~131K params. Current frame as query, past frames as key/value, learnable relative lag encoding. N=1 is identity by construction.

**Tech Stack:** PyTorch nn.MultiheadAttention, existing v2 pipeline. No new dependencies.

**Reviewed by:** Claude + Codex (agreed 100% on architecture, training recipe, and success criteria).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `v2/model/temporal.py` | CrossAttnBlock + TemporalCrossAttention + TemporalMagPhaseFusion model |
| Create | `v2/model/tests/test_temporal.py` | Unit tests for temporal module |
| Create | `v2/data/windowed_dataset.py` | WindowedTrajectoryDataset returning N-frame windows |
| Create | `v2/data/tests/test_windowed_dataset.py` | Tests for windowed dataset |
| Create | `v2/train/train_temporal.py` | Training script with variable-N, staged freeze/unfreeze |
| Modify | `v2/model/__init__.py` | Export TemporalMagPhaseFusion |

---

## Task 1: Windowed Dataset

Returns N consecutive radar frames + lidar label for the last frame.

**Files:**
- Create: `v2/data/windowed_dataset.py`
- Create: `v2/data/tests/test_windowed_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# v2/data/tests/test_windowed_dataset.py
import torch
import pytest

@pytest.fixture
def tmp_traj(tmp_path):
    """Create fake trajectory with 20 frames."""
    traj_id = 999
    N = 20
    torch.save(torch.randn(N, 8, 512, dtype=torch.complex64), str(tmp_path / f"radar_{traj_id}.pt"))
    torch.save(torch.randn(N, 8192, 3), str(tmp_path / f"lidar_{traj_id}.pt"))
    torch.save(torch.ones(N), str(tmp_path / f"norm_{traj_id}.pt"))
    return str(tmp_path), traj_id

def test_windowed_dataset_shape(tmp_traj):
    from v2.data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=5)
    # With 20 frames and window=5, eligible targets: frames 4..19 = 16 samples
    assert len(ds) == 16
    radar_window, lidar, norm = ds[0]
    assert radar_window.shape == (5, 8, 512)
    assert radar_window.dtype == torch.complex64
    assert lidar.shape == (8192, 3)

def test_windowed_dataset_single_frame(tmp_traj):
    """Window size 1 should return single frames (all 20 eligible)."""
    from v2.data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=1)
    assert len(ds) == 20
    radar_window, lidar, norm = ds[0]
    assert radar_window.shape == (1, 8, 512)

def test_windowed_dataset_temporal_order(tmp_traj):
    """Window should be chronologically ordered, last frame = target."""
    from v2.data.windowed_dataset import WindowedTrajectoryDataset
    proc_dir, tid = tmp_traj
    ds = WindowedTrajectoryDataset(tid, proc_dir, window_size=3)
    radar_window, lidar, norm = ds[5]  # target = frame 7 (index 5 + window-1=2 offset)
    # Verify window is frames [5, 6, 7]
    raw_radar = torch.load(str(tmp_traj[0]) + f"/radar_{tid}.pt", weights_only=True)
    assert torch.allclose(radar_window[-1], raw_radar[7])  # last = target frame
    assert torch.allclose(radar_window[0], raw_radar[5])   # first = oldest
```

- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Implement WindowedTrajectoryDataset**

```python
# v2/data/windowed_dataset.py
"""Windowed trajectory dataset for temporal fusion training.

Returns N consecutive radar frames + lidar label for the last (target) frame.
The window slides over each trajectory with stride 1.
"""
import os
import random
import torch
from torch.utils.data import Dataset


class WindowedTrajectoryDataset(Dataset):
    """Returns (window_size, 8, 512) radar windows + last-frame lidar.

    Eligible samples: frames with >= (window_size - 1) preceding frames.

    Args:
        traj_id: trajectory ID
        processed_dir: directory with .pt files
        window_size: number of consecutive frames (default 5)
        augment: apply IQ augmentation to radar (phase rotation + noise only, NO range shift)
    """

    def __init__(self, traj_id, processed_dir, window_size=5, augment=False, noise_sigma=0.01):
        super().__init__()
        self.window_size = window_size
        self.augment = augment
        self.noise_sigma = noise_sigma

        self.radar = torch.load(os.path.join(processed_dir, f"radar_{traj_id}.pt"), weights_only=True)
        self.lidar = torch.load(os.path.join(processed_dir, f"lidar_{traj_id}.pt"), weights_only=True)
        self.norm_factors = torch.load(os.path.join(processed_dir, f"norm_{traj_id}.pt"), weights_only=True)

        self.n_frames = self.radar.shape[0]
        # Eligible targets: frames with enough history
        self.n_eligible = self.n_frames - (window_size - 1)

    def __len__(self):
        return max(0, self.n_eligible)

    def __getitem__(self, idx):
        # Window: [idx, idx+1, ..., idx+window_size-1]
        # Target (last frame): idx + window_size - 1
        target_idx = idx + self.window_size - 1
        radar_window = self.radar[idx:idx + self.window_size].clone()  # (W, 8, 512)
        lidar = self.lidar[target_idx]  # (8192, 3) — label for last frame only
        norm = self.norm_factors[target_idx]

        if self.augment:
            # Same augmentation applied to ALL frames in window (consistent phase/noise)
            theta = random.uniform(0.0, 2.0 * 3.141592653589793)
            phase_rot = torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))
            noise = self.noise_sigma * (
                torch.randn_like(radar_window.real) + 1j * torch.randn_like(radar_window.imag)
            ).to(torch.complex64)
            radar_window = radar_window * phase_rot + noise

        return radar_window, lidar, norm
```

- [ ] **Step 4: Add build_windowed_dataloaders function**

Same pattern as build_dataloaders but uses WindowedTrajectoryDataset. Takes window_size param.

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git add v2/data/windowed_dataset.py v2/data/tests/test_windowed_dataset.py
git commit -m "feat(v2): add windowed trajectory dataset for temporal fusion"
```

---

## Task 2: Temporal Cross-Attention Module

The core transformer: per-range-bin cross-attention with learnable lag encoding.

**Files:**
- Create: `v2/model/temporal.py`
- Create: `v2/model/tests/test_temporal.py`

- [ ] **Step 1: Write failing tests**

```python
# v2/model/tests/test_temporal.py
import torch
import pytest

def test_temporal_identity_single_frame():
    """N=1 should return input unchanged (identity by construction)."""
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 1, 128, 512)  # B=2, N=1
    out = module(x)
    assert out.shape == (2, 128, 512)
    assert torch.allclose(out, x[:, 0], atol=1e-6), "N=1 should be identity"

def test_temporal_multi_frame_shape():
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 5, 128, 512)  # B=2, N=5
    out = module(x)
    assert out.shape == (2, 128, 512)

def test_temporal_different_frames_different_output():
    """Multi-frame output should differ from single-frame (temporal info used)."""
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    x = torch.randn(2, 5, 128, 512)
    out_multi = module(x)
    out_single = module(x[:, -1:, :, :])  # just last frame
    assert not torch.allclose(out_multi, out_single, atol=1e-3), \
        "Multi-frame should produce different output than single-frame"

def test_temporal_param_count():
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4, ff_dim=256, max_lag=16)
    n_params = sum(p.numel() for p in module.parameters())
    assert n_params < 200_000, f"Too many params: {n_params}"
    print(f"Temporal module params: {n_params:,}")

def test_temporal_variable_n():
    """Should handle different N values without error."""
    from v2.model.temporal import TemporalCrossAttention
    module = TemporalCrossAttention(d_model=128, n_heads=4)
    for N in [1, 3, 5, 8]:
        x = torch.randn(2, N, 128, 512)
        out = module(x)
        assert out.shape == (2, 128, 512), f"Failed for N={N}"

def test_full_model_end_to_end():
    from v2.model.temporal import TemporalMagPhaseFusion
    model = TemporalMagPhaseFusion()
    x = torch.randn(2, 5, 8, 512, dtype=torch.complex64)  # B=2, N=5
    pts, conf = model(x)
    assert pts.shape == (2, 8192, 3)
    assert conf.shape == (2, 8192, 1)

def test_full_model_single_frame():
    from v2.model.temporal import TemporalMagPhaseFusion
    model = TemporalMagPhaseFusion()
    x = torch.randn(2, 1, 8, 512, dtype=torch.complex64)  # N=1
    pts, conf = model(x)
    assert pts.shape == (2, 8192, 3)
```

- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Implement TemporalCrossAttention**

```python
# v2/model/temporal.py
"""Temporal cross-attention module for multi-frame radar fusion.

Per-range-bin cross-attention: current frame queries past frames.
Residual design: fused = current + delta, so N=1 is identity by construction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.model.cvnn import safe_modulus
from v2.model.lista import FFTBeamformer
from v2.model.decoder import PointCloudDecoder


class CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention block: cross-attn + residual + FFN + residual.

    Query length is always 1 (current frame at one range bin).
    No self-attention (wasted with query_len=1).
    """

    def __init__(self, d_model=128, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        """
        Args:
            q:  (batch, 1, d_model) — current frame query
            kv: (batch, N-1, d_model) — history key-values
        Returns:
            (batch, 1, d_model) — temporal DELTA only (no residual here;
            the outer TemporalCrossAttention adds current + delta)
        """
        # Cross-attention (no residual — delta-only block)
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)

        # FFN on the attention output
        delta = self.ffn(self.ln_ff(attn_out))
        return delta


class TemporalCrossAttention(nn.Module):
    """Per-range-bin temporal cross-attention with residual design.

    At each of R=512 range positions independently:
    - Query: current (last) frame feature (dim d_model)
    - Keys/Values: past frames + learnable lag encoding
    - Output: current + delta (residual)

    N=1 returns current unchanged (identity by construction).

    Args:
        d_model:   Feature dimension (default 128, matches bridge output)
        n_heads:   Attention heads (default 4)
        ff_dim:    FFN hidden dim (default 256)
        max_lag:   Maximum supported history length (default 16)
        dropout:   Dropout rate (default 0.1)
    """

    def __init__(self, d_model=128, n_heads=4, ff_dim=256, max_lag=16, dropout=0.1):
        super().__init__()
        self.block = CrossAttnBlock(d_model, n_heads, ff_dim, dropout)
        # Learnable relative lag encoding: lag_embed[k] for k=0..max_lag-1
        # k=0 = most recent history frame, k=1 = second most recent, etc.
        self.lag_embed = nn.Embedding(max_lag, d_model)

    def forward(self, x):
        """
        Args:
            x: (B, N, C, R) — N frames of bridge features, last = current
        Returns:
            (B, C, R) — fused features for current frame
        """
        B, N, C, R = x.shape
        current = x[:, -1]  # (B, C, R) — current frame

        if N == 1:
            return current  # identity — no history to attend to

        history = x[:, :-1]  # (B, N-1, C, R) — past frames

        # Per-range-bin: reshape so each range bin is an independent batch item
        # Query: (B*R, 1, C)
        q = current.permute(0, 2, 1).reshape(B * R, 1, C)

        # KV: (B*R, N-1, C)
        kv = history.permute(0, 3, 1, 2).reshape(B * R, N - 1, C)

        # Add learnable lag encoding to KV
        # Lag indices: most recent history = lag 0, oldest = lag N-2
        lag_indices = torch.arange(N - 1, device=x.device).flip(0)  # [N-2, N-3, ..., 0]
        lag_enc = self.lag_embed(lag_indices)  # (N-1, C)
        kv = kv + lag_enc.unsqueeze(0)  # broadcast over B*R

        # Cross-attention → delta only (block returns delta, not q+delta)
        delta = self.block(q, kv)  # (B*R, 1, C) — pure temporal update

        # Reshape back: (B*R, 1, C) → (B, R, C) → (B, C, R)
        delta = delta.squeeze(1).reshape(B, R, C).permute(0, 2, 1)

        # Guard: lag must not exceed embedding table
        assert N - 1 <= self.lag_embed.num_embeddings, \
            f"History length {N-1} exceeds max_lag {self.lag_embed.num_embeddings}"

        # Residual: current + temporal update (single addition, no double-add)
        return current + delta
```

- [ ] **Step 4: Implement TemporalMagPhaseFusion model**

```python
class TemporalMagPhaseFusion(nn.Module):
    """MagnitudePhaseFusion + temporal cross-attention.

    Per-frame: FFT → mag/sin/cos → Conv1d bridge → (128, 512)
    Temporal: cross-attention fusion across N frames → (128, 512)
    Decoder: unchanged PointCloudDecoder → (8192, 3)
    """

    def __init__(self, N_az=256, bridge_out_ch=128, max_lag=16):
        super().__init__()
        self.beamformer = FFTBeamformer(N_az=N_az)
        self.bridge = nn.Sequential(
            nn.Conv1d(N_az * 3, N_az, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(N_az, bridge_out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(16, bridge_out_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = TemporalCrossAttention(
            d_model=bridge_out_ch, n_heads=4, ff_dim=256, max_lag=max_lag
        )
        self.decoder = PointCloudDecoder(feature_ch=bridge_out_ch)

    def forward(self, y_seq):
        """
        Args:
            y_seq: (B, N, 8, 512) complex64 — N consecutive frames
        Returns:
            pts: (B, 8192, 3), conf: (B, 8192, 1)
        """
        B, N, A, R = y_seq.shape

        # Per-frame encoding (batched, shared weights)
        y_flat = y_seq.reshape(B * N, A, R)
        spec = self.beamformer(y_flat)
        mag = safe_modulus(spec)
        phase = torch.angle(spec)
        sin_ph = torch.sin(phase)
        cos_ph = torch.cos(phase)
        gate = (mag > mag.mean(dim=1, keepdim=True) * 0.1).float()
        fused_input = torch.cat([mag, sin_ph * gate, cos_ph * gate], dim=1)
        features = self.bridge(fused_input)  # (B*N, 128, 512)

        # Reshape to (B, N, 128, 512)
        features = features.reshape(B, N, -1, R)

        # Temporal fusion → (B, 128, 512)
        fused = self.temporal(features)

        # Decode (unchanged)
        pts, conf = self.decoder(fused)
        return pts, conf

    def load_single_frame_weights(self, checkpoint_path):
        """Load weights from a single-frame MagnitudePhaseFusion checkpoint.

        Handles common checkpoint formats:
        - ckpt["model_state_dict"] (our training script format)
        - ckpt (raw state_dict)
        - "module." prefix stripping (DataParallel)

        Loads beamformer, bridge, and decoder weights. Temporal block
        stays at random init (near-identity due to residual design).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state dict from various checkpoint formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            src_state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            src_state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith(("beamformer", "bridge", "decoder")) for k in ckpt):
            src_state = ckpt  # raw state dict
        else:
            raise ValueError(f"Cannot find state dict in checkpoint. Keys: {list(ckpt.keys())[:10]}")

        # Strip "module." prefix if present (DataParallel)
        src_state = {k.removeprefix("module."): v for k, v in src_state.items()}

        # Load matching keys, skip temporal.* (not in source)
        own_state = self.state_dict()
        loaded, skipped, mismatched = 0, 0, 0
        for k, v in src_state.items():
            if k not in own_state:
                skipped += 1
                continue
            if own_state[k].shape != v.shape:
                print(f"  [WARN] Shape mismatch for {k}: ckpt={v.shape} vs model={own_state[k].shape}")
                mismatched += 1
                continue
            own_state[k] = v
            loaded += 1

        self.load_state_dict(own_state)
        temporal_params = sum(1 for k in own_state if k.startswith("temporal."))
        print(
            f"[load_single_frame_weights] Loaded {loaded} params, "
            f"skipped {skipped} (not in model), mismatched {mismatched}, "
            f"temporal params (random init): {temporal_params}"
        )
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Export from __init__.py**

Add to `v2/model/__init__.py`:
```python
from v2.model.temporal import TemporalMagPhaseFusion, TemporalCrossAttention
```

- [ ] **Step 7: Commit**

```bash
git add v2/model/temporal.py v2/model/tests/test_temporal.py v2/model/__init__.py
git commit -m "feat(v2): add temporal cross-attention module for multi-frame fusion"
```

---

## Task 3: Temporal Training Script

Training script with variable-N windows, staged freeze/unfreeze, checkpoint loading.

**Files:**
- Create: `v2/train/train_temporal.py`

- [ ] **Step 1: Write training script**

Key differences from train.py:
- Model: TemporalMagPhaseFusion
- Data: build_windowed_dataloaders with variable window_size
- Loads single-frame checkpoint to initialize encoder+decoder
- Staged training:
  - Epochs 0-4: freeze encoder+decoder, train ONLY temporal block (lr=7e-5)
    - IMPORTANT: exclude N=1 during this stage (N=1 bypasses temporal block → no gradients)
    - Use window_sizes=[3, 5, 8] for epochs 0-4
  - Epochs 5+: unfreeze all, encoder+decoder at 10× lower LR (7e-6)
    - Include N=1 again for single-frame regularization
- Variable N strategy: build dataset with max window (8). At each epoch, randomly pick N from allowed set.
  Crop windows to N inside __getitem__ by taking the LAST N frames from the stored max-window.
  This avoids rebuilding dataloaders each epoch.
- Eval: run at each N separately, log val/chamfer_N1, val/chamfer_N3, etc.

```python
DEFAULT_CONFIG = {
    "batch_size": 12,
    "lr": 7e-5,
    "backbone_lr_factor": 0.1,  # 10x lower LR for encoder+decoder
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
    "pretrained_checkpoint": None,  # path to single-frame best.pt
    "window_sizes": [1, 3, 5, 8],  # variable N for training
    "eval_window_sizes": [1, 3, 5, 8],  # N values for eval Pareto curve
}
```

- [ ] **Step 2: Verify import works**

```bash
docker compose run --rm mmdar python3 -c "from v2.train.train_temporal import train; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add v2/train/train_temporal.py
git commit -m "feat(v2): add temporal training script with variable-N and staged freeze"
```

---

## Task 4: Run Experiments

- [ ] **Step 1: Train temporal model from best single-frame checkpoint**

```bash
docker compose run --rm mmdar python3 -m v2.train.train_temporal \
  --pretrained-checkpoint logs/v2_mag_phase/best.pt \
  --batch-size 12 --lr 7e-5 --epochs 50 \
  --log-dir logs/v2_temporal_xattn \
  --checkpoint-every 10
```

- [ ] **Step 2: Run control experiments** (if temporal works)

Mean pooling control:
```bash
# Modify temporal module to use mean pooling instead of cross-attention
```

Channel stacking control:
```bash
# Modify temporal module to use Conv1d(N*128, 128, k=1)
```

- [ ] **Step 3: Analyze Pareto curve and update results/README.md**

---

## Success Criteria

1. **No regression at N=1**: Chamfer ≤ 0.32m (should match ~0.309m)
2. **mod-H improvement at N=5**: mod-H < 0.35m (was 0.423m single-frame)
3. **Transformer beats channel stacking**: at same N, transformer mod-H is better
4. **Clear Pareto curve**: monotonic improvement with N, diminishing returns visible
5. **Total model < 2.5M params**
6. **Inference < 5ms at N=5** on RTX 5090
