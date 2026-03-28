# Phase 9a: Beamspace → Gaussian Set Decoder

## Problem

After 8 phases, we've proven:
1. The beamformer (FFT/LISTA) is an information bottleneck — 8 antennas give ~14° resolution, no decoder can recover precision from blurred features (Phases 3-8)
2. Fixed 8192-point decoders have poor precision (nn_pred→gt = 0.322m) despite good coverage (Phases 3-7)
3. Binary occupancy on beamformed grids fails because the grid is oversampled relative to actual angular resolution (Phase 8a)

The field has moved toward: raw signal processing, Gaussian scene representations, and physics-informed output formats.

## Strategy: De-risk in stages (Codex recommendation)

Don't change everything at once. Phase 9a tests the Gaussian representation on a known-working signal path. Phase 9b swaps in the raw antenna frontend.

- **Phase 9a** (this spec): Learned beamspace → Gaussian set decoder → evaluate
- **Phase 9b**: Raw 8-antenna IQ → learned angular module → Gaussian decoder
- **Phase 9c**: End-to-end fine-tuning

If 9a can't approach the baseline, the raw-antenna version won't save it.

## Phase 9a Architecture

```
Input: Raw IQ (8 antennas × 512 range bins, complex64) per frame

Signal frontend:
  Range FFT (fixed, standard) → (8, 512) complex per frame
  Learned beamspace layer: W ∈ ℂ^(B×8), B=48
    - Initialized from ULA steering matrix (first 48 of 256 bins)
    - Trainable: learns to focus on informative angle directions
    - Output: (48, 512) complex per frame
  Channelize: [|Re|, |Im|, log_power] → (3, 48, 512) float per frame
  Shallow 2D conv: (3, 48, 512) → (64, 48, 512) features per frame

Temporal fusion:
  Stack N=8 frames of features → (N×64, 48, 512) = (512, 48, 512)
  OR cross-attention across frames (reuse v2 temporal module pattern)

Gaussian set decoder (DETR-style):
  96 learnable queries
  Cross-attention: queries attend to fused feature map
  Per-query MLP head: → (r, φ, log_σ_r, log_σ_φ, existence_logit)
  Output: up to 96 2D Gaussians in polar coordinates

Post-processing:
  Filter by existence threshold → variable-count Gaussian set
  Convert polar (r, φ) to Cartesian (x, y) for eval
  Sample points from Gaussians OR use Gaussian centers directly
```

## Key Design Decisions

### Predict in polar, not Cartesian
Radar measurement uncertainty is naturally radial (range) and azimuthal (angle). A Gaussian in polar space directly encodes this: σ_r captures range uncertainty, σ_φ captures angular uncertainty. Converting to Cartesian for eval is straightforward.

### 96 queries, not 256 or 8192
Our lidar GT has ~665 points per frame on the baseline grid. At ~8-16 points per Gaussian, that's ~40-80 active Gaussians. K_max=96 gives headroom without excessive search space. Unused queries get low existence probability.

### Learned beamspace, not FFT
A complex linear layer W ∈ ℂ^(48×8) initialized from the steering matrix is a differentiable, learnable version of beamforming. 48 output bins (vs FFT's 256) is better matched to the actual angular resolution (~14° = ~13 resolvable directions). The network can adapt the beamspace to focus on informative directions.

### DETR-style set decoder
Learnable queries + cross-attention is proven for set prediction (DETR, 2020). Each query attends to the feature map and predicts one Gaussian. Hungarian matching assigns predictions to GT during training. No grid, no fixed cardinality.

## Training

### Teacher Gaussian fitting (offline, one-time)
Before training the neural network, fit Gaussians to each lidar frame offline:
1. For each lidar frame (8192, 3): project to 2D (x, y)
2. Grid-quantize to baseline eval grid → ~665 points
3. Fit K Gaussians using the Gaussian RIO algorithm (K-Means init + Ceres optimization)
4. Save teacher Gaussians as training targets

This provides stable, high-quality supervision. The neural network learns to predict Gaussian parameters that match the teacher, not to optimize raw likelihood from scratch.

### Loss function
Composite loss (NOT Chamfer):
1. **Set prediction loss** (Hungarian matching): L1 on (r, φ) centers + L1 on (log_σ_r, log_σ_φ) between predicted and teacher Gaussians
2. **Existence loss**: focal BCE on existence logits
3. **Rendered occupancy auxiliary**: render predicted Gaussians to polar grid → BCE against lidar occupancy. This provides dense spatial supervision.
4. **Optional mixture likelihood**: -log Σ_k α_k N(p_i | μ_k, Σ_k) for each GT point. Soft assignment to predicted Gaussians.

### Training schedule
- Pretrain existence + center prediction (freeze scale)
- Unfreeze scale prediction after existence is calibrated
- End-to-end fine-tuning with all loss terms

## Evaluation

1. Filter predicted Gaussians by existence threshold
2. Convert centers from polar (r, φ) to Cartesian (x, y)
3. Option A: use Gaussian centers directly as point cloud
4. Option B: sample points from each Gaussian proportional to its area
5. Compute Chamfer + mod-H against lidar GT (same as all previous phases)
6. Threshold sweep on existence to find optimal mod-H

### Success criteria
- Chamfer ≤ 0.310 (within range of single-frame v2 results)
- mod-H ≤ 0.250 (significant improvement over v2's 0.429)
- If met: Gaussian representation + beamspace is viable, proceed to Phase 9b
- If mod-H > 0.35: representation change alone isn't enough, reassess

## Implementation Components

| File | Responsibility |
|------|---------------|
| `v2/data/fit_teacher_gaussians.py` | Offline: fit Gaussians to lidar frames, save as .pt |
| `v2/data/gaussian_dataset.py` | Dataset: load IQ + teacher Gaussians, N-frame windowing |
| `v2/model/beamspace.py` | Learned beamspace layer (complex linear, init from steering) |
| `v2/model/gaussian_decoder.py` | DETR-style set decoder: queries → Gaussian parameters |
| `v2/model/gaussian_model.py` | Full model assembly: beamspace → temporal → decoder |
| `v2/train/loss_gaussian.py` | Hungarian matching + set prediction loss + occupancy auxiliary |
| `v2/train/train_gaussian.py` | Training script |
| `v2/eval/gaussian_eval.py` | Gaussian → point cloud → Chamfer/mod-H |

## Key References

- Gaussian RIO (IEEE RA-L 2026): 3D Gaussian scene representation for radar odometry
  - Code: https://github.com/robotics-upo/gaussian-rio-cpp
  - Mahalanobis loss, K-Means init, joint optimization
- DETR (Carion et al., ECCV 2020): set prediction with learnable queries + Hungarian matching
- ADC-SR (CVPR 2023): learned virtual antenna channels from raw receivers
- 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023): differentiable Gaussian rendering
- SR-SPECNet (2024): domain-informed 1D spectral estimation per range gate

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| DETR decoder may struggle with 96 queries on small feature maps | Start with 48 queries, scale up |
| Teacher Gaussian fitting quality depends on lidar density | Use raw 8192 lidar points (before grid quantization) |
| Beamspace W may not learn beyond steering init | Add covariance branch as second input (x·x^H) |
| Temporal fusion adds complexity | Start with simple channel stacking, add attention later |
| Gaussian rendering for occupancy loss is expensive | Use sparse rendering (only around predicted centers) |
