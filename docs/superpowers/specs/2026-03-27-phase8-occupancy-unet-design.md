# Phase 8: LISTA + U-Net Polar Occupancy Decoder

## Problem

The v2 point decoder matches baseline Chamfer (0.295m) but has 2.3x worse mod-H (0.429 vs 0.189). Through 7 phases we proved the gap is real model quality — specifically poor point placement precision (nn_pred→gt = 0.322m vs baseline 0.180m). The point decoder sprays 8192 points with good coverage but many land in wrong locations.

The baseline succeeds because: (a) 41-frame early channel stacking, (b) occupancy thresholding → conservative/precise output, (c) 17.5M params U-Net. Our failed experiments changed temporal fusion, loss functions, decoder architecture — but never replicated the baseline's winning combination.

## Phase 8a: LISTA log_power on baseline grid (minimum viable experiment)

**Goal**: Isolate exactly ONE variable — replace raw radar magnitude PNGs with LISTA log_power features. Everything else identical to baseline.

**Data flow**:
```
Per frame: Raw IQ (8 ant, 512 range)
  → LISTA beamformer → (256_az_sintheta, 512_range) complex
  → log_power = log(|Re|² + |Im|² + eps) → (1, 256_az, 512_range)
  → Reproject: sin_theta-uniform → angle-uniform (1D interp along azimuth)
  → (1, 256_range, 512_az) — canonical baseline grid

Stack 41 frames → (41, 256, 512) — identical shape to baseline input

→ Symmetric 2D U-Net (no asymmetric azimuth upsampling — input already 512 az)
→ (1, 256, 512) polar occupancy logits → sigmoid
→ Threshold → variable-size point cloud → eval
```

**What stays identical to baseline**: canonical grid (256r × 512az, angle-uniform), labels (lidar PNGs from dataset_5), eval pipeline (legacy_cartesian), temporal stacking (41 frames), loss (BCE + Dice), output representation (polar occupancy).

**What changes**: input features only (LISTA log_power vs raw radar uint8 magnitude).

**Preprocessing**: offline — run LISTA + log_power + reproject for all frames in all trajectories, save as .pt files. Per frame: 256×512×4 bytes = 0.5MB. ~42K frames × 0.5MB ≈ 21GB float32 (11GB float16).

**U-Net architecture**: symmetric encoder-decoder since no azimuth upsampling needed. Encoder: 41→64→128→256→512 with stride-2. Decoder: 512→256→128→64 with skip connections. Head: 64→1. ~3-5M params.

**Training**: Adam, lr=7e-5, BCE + Dice, batch=12, checkpoint every 10 epochs. Sweep epochs 10-30 (baseline sweet spot).

**Success criteria**:
- Chamfer ≤ 0.310 (single-frame baseline level or better)
- mod-H ≤ 0.200 (match baseline)
- If met: LISTA log_power + occupancy is the winning combination
- If mod-H still ~0.4: LISTA's angular features are the bottleneck, not the decoder

**Codex review notes**:
- Grid mismatch is the #1 risk — must reproject to canonical grid before U-Net, not at eval
- Focal BCE not needed if using standard BCE + Dice (matching baseline)
- Validate threshold sweep: test thresholds [0.3, 0.5, 0.7] to find optimal mod-H
- log_power only first (41ch), add Re/Im later as ablation

---

## Phase 8b: Add complex features (Re, Im) — ablation

**Prerequisite**: Phase 8a achieves mod-H < 0.25

**Change**: Replace 41-channel log_power input with 123-channel (Re, Im, log_power × 41 frames).

**Question answered**: Does preserving phase information from LISTA improve occupancy prediction beyond log_power alone?

**U-Net modification**: first conv changes from 41→64 to 123→64. Everything else identical.

---

## Phase 8c: Frame count sweep

**Prerequisite**: Phase 8a or 8b achieves mod-H < 0.25

**Change**: Sweep N ∈ {1, 3, 5, 8, 16, 24, 41} frames.

**Question answered**: What is the minimum temporal context for competitive mod-H with LISTA features? The baseline needs 41 frames of raw radar — does LISTA's phase-preserving beamforming reduce this requirement?

**Significance**: If N=8 matches N=41, LISTA enables 5x reduction in temporal context → lower latency, less memory, closer to real-time streaming.

---

## Phase 9: Raw antenna input (no beamforming)

**Prerequisite**: Phase 8 series complete, understand LISTA's contribution

**Change**: Replace LISTA beamformer with learnable angular processing directly on raw 8-antenna complex data. Per range bin: (8, complex) → learned module → angular features.

**Options**:
- Deep-unfolded LISTA/ADMM on raw antenna data
- 1D spectral estimation network (SR-SPECNet style)
- Transformer on antenna array dimension

**Question answered**: Can learned angular processing from raw antennas beat LISTA's fixed beamforming?

---

## Phase 10: Ego-motion coherent integration

**Prerequisite**: Phase 9 complete

**Change**: Add ego-motion-aware temporal fusion. Coherent integration for static targets (synthetic aperture), separate path for dynamic objects (Doppler-based).

**Question answered**: Can ego-motion synthesis create a larger effective aperture, improving angular resolution beyond the 8-antenna limit for static scene elements?

---

## Phase 11: Probabilistic output + uncertainty

**Prerequisite**: Phase 8-10 results analyzed

**Change**: Add per-cell uncertainty estimation (predicted variance). Calibrate against lidar GT. Evaluation: occupancy calibration, surface precision-recall, downstream SLAM compatibility.

**Question answered**: Can the system express what it doesn't know, enabling safer downstream use?

---

## Key references

- ADC-SR (CVPR 2023) — hallucinate virtual antenna channels
- SR-SPECNet (2024) — domain-informed 1D spectral estimation
- Radar Fields (SIGGRAPH 2024) — differentiable FMCW forward model
- RadarOcc (NeurIPS 2024) — 4D radar tensor → occupancy
- DenserRadar (ITSC 2024) — lidar-supervised CFAR-free detection
- Towards Foundational Models for Single-Chip Radar (ICCV 2025)
- DREAM-PCD (2024/25) — coherent + non-coherent multi-frame integration
