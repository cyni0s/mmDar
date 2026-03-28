# mmDar v3: Research Direction

## Status: DRAFT — pending user review

## The Case for a Paradigm Shift

### What we learned (Phases 1-7)

The RadarHD U-Net super-resolution approach and our v2 raw-IQ point decoder both hit the same wall: **point placement precision**. After 7 phases of experiments, we proved:

- Chamfer distance can be matched (0.295m) with 8 frames, 2.2M params
- Modified Hausdorff is stuck at 0.429 vs baseline's 0.189
- The gap is NOT cardinality, NOT eval pipeline, NOT temporal, NOT angular topology, NOT loss function
- The gap IS real: the v2 decoder sprays 8192 points with good coverage but poor precision (nn_pred→gt = 0.322m)
- The baseline achieves 0.180 precision with ~2874 points from occupancy thresholding — a fundamentally more conservative approach

### Why the current paradigm is limited

Four independent analyses (survey papers, web SOTA research, two Codex sessions) converge on the same diagnosis:

1. **Radar→lidar is NOT image super-resolution.** The "degradation" is RF physics (sidelobes, multipath, specularity, limited aperture) — global, nonlinear, fundamentally different from image blur.

2. **Beamforming to a 2D image is a lossy projection.** It collapses multi-antenna phase structure and spends much of the angular information budget before the neural network even sees the data.

3. **Dense point-cloud regression is the wrong target** for a sensor whose returns are sparse, specular, multipath-heavy, and probabilistic. Chamfer can look decent while the decoder sprays plausible-but-wrong points.

4. **The angular resolution ceiling is real.** With 8 half-wavelength virtual elements at 77 GHz (~3.8mm wavelength), the aperture is ~3.5λ → ~14-15° boresight resolution → **2.7m cross-range separability at 10.8m**. No neural network creates new spatial bandwidth from a single snapshot.

### What CAN help

- **Coherent temporal integration**: improves SNR and synthesizes larger aperture from ego-motion (for static scene)
- **Sparse signal recovery**: MUSIC, ESPRIT, compressed sensing — exploit target sparsity to localize beyond beamwidth
- **Strong learned priors**: what plausible scenes look like given radar cues (completion, not hallucination)
- **Probabilistic output**: express uncertainty instead of guessing

### The right research framing

> ~~"Single-chip AWR1843 becomes lidar"~~ — **rejected**
>
> "Single-chip radar + raw-wave modeling + temporal integration + strong priors → **useful geometric occupancy in all weather**" — **defensible**

---

## Proposed v3 Architecture

### Input: Raw complex per-antenna data

```
Raw IQ (8 antennas × 512 range bins, complex64)
  → Range FFT (keep complex, per-antenna)
  → NO angle FFT / beamforming at this stage
```

**Rationale**: Angle FFT / beamforming collapses the multi-antenna phase structure into a spatial image. This is information-destroying. The network should operate on the raw per-antenna complex data and learn its own angle processing.

### Module 1: Learned angular processing (signal domain)

Two parallel branches:

**Branch A — Deep-unfolded sparse recovery:**
- LISTA/ADMM-style iterative thresholding on per-range-bin antenna data
- Produces angle-likelihood spectrum (not a hard beamformed image)
- Interpretable: each iteration refines the sparse estimate
- References: ADC-SR (CVPR 2023), SR-SPECNet (2024)

**Branch B — Raw-radar temporal transformer:**
- Cross-attention over N frames of 8-antenna complex data
- Per-range-bin processing (matches signal structure)
- Learns to exploit ego-motion for synthetic aperture on static targets
- Separate handling of dynamic objects (Doppler-aware gating)
- Reference: existing v2 temporal cross-attention architecture

**Fusion**: concatenate or cross-attend branch outputs → rich per-range-bin feature vector that preserves angular information from both approaches.

### Module 2: Ego-motion-aligned multiframe fusion

- Explicit coherent path for static structure (SAR-like integration from ego-motion)
- Separate dynamic-object path (Doppler-based segmentation)
- Ego-motion from IMU or odometry (available in dataset)

### Module 3: Probabilistic polar occupancy decoder

```
Fused features (per range-azimuth cell)
  → Polar occupancy logits + uncertainty
  → Sigmoid threshold → variable-cardinality point cloud
```

**Why polar occupancy**:
- Matches radar's native measurement geometry
- Variable cardinality via thresholding (conservative, precise — like baseline)
- Mainstream consensus in 2024-2025 SOTA (RCBEVDet, DenserRadar, RadarOcc)
- Decouples output resolution from network architecture

**With uncertainty**: per-cell predicted variance, enabling downstream tasks to weight points by confidence.

### Module 4: Optional point cloud sampler

- Sample points from occupied cells for downstream compatibility
- Positions at cell centers + optional sub-cell refinement
- Purely for evaluation and interface — not the primary output

---

## Training

### Supervision
- **Primary**: lidar occupancy in polar grid (existing, from v2/data/rasterize.py)
- **Secondary**: temporal consistency loss (adjacent frame predictions should be ego-motion-consistent)
- **Optional**: Doppler consistency, differentiable radar rendering loss (physics forward model)

### Loss functions
- Focal BCE + Dice on polar occupancy (proven in baseline)
- Point-cloud metrics (Chamfer, mod-H) as monitoring, not loss
- Calibration loss on uncertainty estimates

### Evaluation (revised)
- **Primary**: occupancy precision-recall, surface F1, calibrated uncertainty
- **Secondary**: Chamfer distance, modified Hausdorff (for backward compatibility)
- **Downstream**: SLAM/odometry compatibility, object detection from radar occupancy

---

## Research Contributions (for publication)

1. **Physics-consistent virtual aperture expansion**: learned angular processing from raw antenna data, not beamformed images
2. **Dual-branch architecture**: interpretable sparse recovery (LISTA) + data-driven temporal transformer, showing complementary benefits
3. **Probabilistic radar occupancy**: uncertainty-aware output calibrated against lidar ground truth
4. **Comprehensive evaluation**: 84K paired samples, ablation across angular processing, temporal integration, and output representation

---

## Implementation Phases

### Phase 1: Polar occupancy decoder with existing LISTA features (weeks)
- Replace point decoder with occupancy head on current LISTA output
- Proves variable-cardinality output improves mod-H
- Uses existing infrastructure (rasterizer, dataset, eval)
- **This is the minimum viable experiment**

### Phase 2: Raw antenna input (weeks)
- Remove the angle FFT / beamforming
- Feed raw 8-antenna complex data per range bin to learned angular module
- Compare with LISTA baseline to show what's gained

### Phase 3: Temporal coherent integration (weeks)
- Add ego-motion-aware temporal fusion
- Coherent integration for static targets (SAR-like)
- Measure angular resolution improvement from temporal aperture

### Phase 4: Full system + paper (months)
- Combine all modules
- Ablation study
- Write paper

---

## Key References

### Signal-domain angular SR
- ADC-SR (CVPR 2023) — hallucinate virtual antenna channels from 8 receivers
- SR-SPECNet (2024) — domain-informed 1D spectral estimation per range gate
- Radar Fields (SIGGRAPH 2024) — differentiable FMCW forward model + implicit scene

### Raw radar perception
- Towards Foundational Models for Single-Chip Radar (ICCV 2025)
- EchoFusion (NeurIPS 2023) — raw spectrum, 30+ AP improvement
- ADCNet (2023) — learnable SP with distillation

### Occupancy / BEV
- RadarOcc (NeurIPS 2024) — 4D radar tensor → occupancy
- DenserRadar (ITSC 2024) — 3D U-Net on raw 4D tensor, lidar-supervised
- RCBEVDet (CVPR 2024) — radar-camera BEV fusion

### Diffusion (second-stage refinement)
- Radar-diffusion (ICRA 2024) — SDE-based radar point cloud densification
- R2LDM (2025) — latent voxel diffusion, 6-10x densification

### Multi-frame / SAR
- DREAM-PCD (2024/25) — coherent + non-coherent accumulation + denoising
- Radar-Mamba (ACM MM 2025) — SSM-based point cloud enhancement
