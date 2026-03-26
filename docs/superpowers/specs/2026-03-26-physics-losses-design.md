# Physics-Informed Radar Losses — Design Spec

**Goal:** Add two physics-informed loss terms to attack the mod-Hausdorff coverage gap (0.429 vs baseline 0.189) by supervising in radar-native polar coordinates with radar resolution priors.

**Approach:** Differentiable soft-splatting of predicted/GT points into polar occupancy, then Tversky recall loss + radar positive-support loss. Applied to the single-frame MagnitudePhaseFusion model first (one variable at a time).

**Reviewed by:** Claude + Codex (5 physics corrections applied, approved on corrected formulation).

---

## Loss 1: Resolution-Aware Range-Angle Recall (Tversky)

### Physics

| Parameter | Value | Source |
|-----------|-------|--------|
| σ_r (range PSF) | 1.0 bin (~2.1cm) | c/(2B) = 3.75cm → 1.8 bins → σ = 0.75, rounded to 1.0 |
| σ_u (angular PSF in sin-theta space) | 14 bins (fixed, NOT range-dependent) | 8-element λ/2 ULA: Δu ≈ 2/N = 0.25 → 32 bins → σ = 32/2.35 ≈ 14 |
| Tversky α (FP weight) | 0.3 | Hallucinations lightly penalized |
| Tversky β (FN weight) | 0.7 | Missed GT heavily penalized — directly attacks mod-H |

**Key physics constraint:** σ_u is FIXED in sin-theta space because the array response is approximately constant in u = sin(θ). Cross-range blur grows with range only in Cartesian, not in u-space.

### Differentiable Soft-Splatting Pipeline

```
pts (B, N, 3) → bilinear soft-bin onto (B, N_az, N_r) intensity grid I
    → separable Gaussian blur (σ_u=14, σ_r=1.0) on I
    → O = 1 - exp(-I)  ← bounds to [0, 1), differentiable
```

- Bilinear soft-binning: each point distributes weight to its 4 nearest grid cells proportional to distance. Differentiable w.r.t. point positions.
- Gaussian blur: fixed kernel, separable (two 1D conv2d). Efficient, differentiable.
- Bounding: O = 1 - exp(-I) maps [0, ∞) → [0, 1). Applied AFTER blur, not before.

### Tversky Loss

```python
intersection = (O_pred * O_gt).sum(dim=(-2, -1))
FP = (O_pred * (1 - O_gt)).sum(dim=(-2, -1))
FN = ((1 - O_pred) * O_gt).sum(dim=(-2, -1))
tversky = (intersection + smooth) / (intersection + α*FP + β*FN + smooth)
L_recall = 1 - tversky.mean()
```

## Loss 2: Radar Positive-Support Loss

### Physics

The beamformer magnitude at cell (r, θ) indicates signal strength. Where the radar detects returns, at least one predicted point should cover that cell.

### Formulation

```python
# Heuristic power threshold (NOT CFAR — no guard cells)
# beamformer_power must be LINEAR power |X|², not dB or magnitude
mean_power = beamformer_power.mean(dim=1, keepdim=True)  # mean over azimuth
M_plus = (beamformer_power > mean_power * k).float()  # k=6.0

# Cross-entropy at radar-positive cells
L_support = -(M_plus * log(O_pred + ε)).sum() / (M_plus.sum() + 1.0)
```

k=6.0 gives ~0.25% per-cell false alarm under exponential noise model. Conservative — only strong detections contribute.

## Integration

```python
total = chamfer + dcd_w*dcd + 0.1*coverage + 0.01*confidence
      + 0.15 * L_recall      # Tversky in polar space
      + 0.03 * L_support     # Radar positive support
```

Starting weights conservative. Tune on val: λ_recall ∈ [0.05, 0.5], λ_support ∈ [0.01, 0.1].

## Model Changes

MagnitudePhaseFusion needs `forward_with_intermediates()` to return beamformer power:
```python
def forward_with_intermediates(self, y):
    spec = self.beamformer(y)          # (B, 256, 512) complex
    mag = safe_modulus(spec)            # (B, 256, 512) — this is |X|, need |X|²
    power = mag ** 2                    # (B, 256, 512) — linear power for support loss
    # ... bridge + decoder as usual ...
    return pts, conf, power
```

## Files

| Action | Path |
|--------|------|
| Create | `v2/train/loss_physics.py` — soft_splat, ra_recall_loss, radar_support_loss |
| Create | `v2/train/tests/test_loss_physics.py` |
| Modify | `v2/model/__init__.py` — add forward_with_intermediates to MagnitudePhaseFusion |
| Modify | `v2/train/train.py` — wire new losses for model_type "mag_phase" |

## Success Criteria

1. mod-H improves from 0.423 toward <0.35 (any improvement validates the approach)
2. Chamfer stays ≤ 0.32 (no regression)
3. Directed Chamfer (gt→pred) specifically improves (the recall direction)

## Implementation Constraints

- beamformer_power MUST be linear power (|X|²), not magnitude or dB
- O = 1 - exp(-I) applied AFTER blur, not before
- Bilinear soft-binning for gradient flow (NOT hard binning)
- Coordinate convention: u = y/r (sin_theta), tensor order (B, N_az, N_r)
- σ_u = 14 bins FIXED (do not make range-dependent)
- Effective range blur σ_r_effective ≈ sqrt(σ_r² + σ_bilinear²) ≈ sqrt(1.0 + 0.25) ≈ 1.12 — acceptable

---

## Follow-Up Steps (after this experiment)

After evaluating the single-frame model with physics losses:

### If mod-H improves:
1. Apply same physics losses to temporal cross-attention model (N=3,5,8)
2. Standardize eval: run baseline through same eval pipeline for fair comparison
3. Report directed Chamfer (pred→gt, gt→pred) separately to confirm recall improvement
4. Sweep loss weights (λ_recall, λ_support) and PSF parameters (σ_u, σ_r)

### If mod-H does NOT improve:
1. Test with stronger weights (λ_recall = 0.5, λ_support = 0.1)
2. Investigate: is the 8192-point fixed-cardinality decoder the bottleneck?
3. Consider variable-cardinality output (top-K confident points, or occupancy thresholding)
4. Run cardinality control: downsample baseline to 8192 points, re-eval mod-H

### Regardless:
1. Fix val split — use 6-8 trajectories, not 4
2. Run temporal usage controls (repeat current frame, shuffle history)
3. Channel stacking control vs transformer
4. Consider dense U-Net decoder on raw IQ if point decoder mod-H stays >0.35
