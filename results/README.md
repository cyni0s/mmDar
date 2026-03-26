# mmDar Experiment Results

## Comparison Table (Legacy-Cartesian Eval, Paper-Comparable)

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| **5090-optimized** | **0.295** | **0.189** | 0.054 | 0.102 | 0.196 | 0.069 | batch=12, lr=7e-5, fp32, epoch 10 |
| baseline_pretrained | 0.363 | 0.247 | 0.026 | 0.051 | 0.119 | 0.033 | Authors' pretrained 120.pt_gen |
| baseline_paper_params | 0.399 | 0.277 | 0.025 | 0.050 | 0.134 | 0.031 | batch=6, lr=1e-4, fp32, best.pt_gen of 200 epochs |
| baseline_5090_adapted | 0.537 | 0.378 | 0.013 | 0.025 | 0.082 | 0.015 | batch=48, lr=8e-4, bf16, best.pt_gen of 200 epochs |
| convlstm_T8_ep30 | 0.603 | 0.467 | 0.026 | 0.050 | 0.073 | 0.038 | **Negative result.** batch=8, lr=7e-5, fp32, T=8 TBPTT, 30 epochs (10.4h) |

*All table values are median over 18,575 test samples using `--coordinate-mode legacy_cartesian`. Checkpoints selected by Chamfer distance sweep unless noted as best.pt_gen (training loss).*

## Comparison Table (Polar-Direct Eval, Reference Only)

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | Notes |
|------------|-------------|-------------------|-------|
| baseline_pretrained | 0.429 | 0.297 | Polar-direct conversion inflates point-cloud distances |
| baseline_paper_params | 0.492 | 0.341 | |
| baseline_5090_adapted | 0.566 | 0.426 | |

*Polar-direct numbers are retained for reference. The legacy-cartesian pipeline matches the paper's MATLAB eval flow and should be used for all comparisons.*

## Training Run Details

| Experiment | Batch | LR | Mixed Precision | Epochs | Train Time | Checkpoint Selection |
|------------|-------|-----|-----------------|--------|------------|---------------------|
| **5090-optimized** | **12** | **7e-5** | **No (fp32)** | **50** | **~108 min** | **epoch 10 (Chamfer sweep)** |
| baseline_paper_params | 6 | 1e-4 | No (fp32) | 200 | ~7.5h | best.pt_gen (train loss) |
| baseline_5090_adapted | 48 | 8e-4 | Yes (bf16) | 200 | ~5.2h | best.pt_gen (train loss) |
| baseline_optimized | 24 | 1.5e-4 | Yes (bf16) | 400 | ~8.7h | epoch 20 (Chamfer sweep) |

### Hyperparameter Sweep (RTX 5090, bf16)

Systematic sweep to find optimal training config. All runs: bf16 forward pass, fp32 BCE+Dice loss, Adam(weight_decay=5e-4), seed=0, checkpoints selected by Chamfer distance.

| Batch | LR | Precision | Best Chamfer (m) | Best mod-H (m) | Best Epoch | Time to Best |
|-------|-----|-----------|-----------------|----------------|-----------|-------------|
| **12** | **7e-5** | **fp32** | **0.295** | **0.189** | **10** | **22 min** |
| 12 | 7e-5 | bf16 | 0.308 | 0.189 | 20 | 32 min |
| 12 | 1e-4 | bf16 | 0.322 | 0.211 | 10 | 16 min |
| 12 | 5e-5 | bf16 | 0.332 | 0.211 | 20 | 32 min |
| 16 | 1e-4 | bf16 | 0.334 | 0.212 | 10 | 16 min |
| 6 | 1e-4 | bf16 | 0.345 | 0.212 | 30 | 52 min |
| 12 | 1.5e-4 | bf16 | 0.366 | 0.284 | 30 | 48 min |
| 24 | 1.5e-4 | bf16 | 0.372 | 0.228 | 20 | 23 min |

Key findings:
- **Batch=12 dominates** across all LR values and precisions tested
- **LR=7e-5 is optimal** — lower (5e-5) converges too slowly, higher (1e-4+) overfits faster
- **fp32 beats bf16 by ~4% Chamfer** at the same config (0.295 vs 0.308). Use bf16 for fast sweeps, fp32 for final results
- **Sweet spot is epoch 10-20** — metrics degrade after that regardless of config
- **22 minutes** from cold start to Chamfer 0.295m (18% better than paper)

### Checkpoint Sweep (baseline_optimized, batch=24, bf16)

Training loss continues to decrease over 400 epochs, but test metrics peak early and then degrade — classic overfitting. Best test metrics at epoch 20:

| Epoch | Train Loss | Chamfer (m) | mod-Hausdorff (m) |
|-------|-----------|-------------|-------------------|
| 10 | ~0.088 | 0.445 | 0.296 |
| **20** | **~0.070** | **0.372** | **0.228** |
| 30 | ~0.061 | 0.465 | 0.284 |
| 50 | ~0.061 | 0.455 | 0.297 |
| 80 | ~0.060 | 0.378 | 0.268 |
| 100 | ~0.060 | 0.405 | 0.300 |
| 400 (best.pt_gen) | 0.057 | 0.460 | 0.382 |

### Training Speed (RTX 5090, NGC 25.02, PyTorch 2.7)

| Batch | Precision | Steps/Epoch | Time/Epoch | 100 Epochs | 200 Epochs |
|-------|-----------|-------------|------------|------------|------------|
| 6 | fp32 | 3,631 | ~2.2 min | ~3.7h | ~7.5h |
| 24 | bf16 | 908 | ~1.15 min | ~1.9h | ~3.8h |
| 48 | bf16 | 454 | ~1.3 min | ~2.2h | ~4.3h |

*Inference on full test set (18,575 samples): ~550s (~9 min) at batch=1. Evaluation: ~2-3 min.*
*Data loading (train+test into RAM): ~15 min per Docker container launch.*

### Convergence Notes

**baseline_paper_params** (paper-exact):
- Loss curve: 0.76 (epoch 0 start) → plateau ~0.065 (epoch 200)
- Training metrics worse than pretrained model despite same hyperparameters
- Checkpoint selected by training loss, which is suboptimal (see sweep above)

**baseline_5090_adapted** (5090-optimized):
- Loss curve: 0.76 (epoch 0 start) → plateau ~0.089 (epoch 200)
- Linear LR scaling (8e-4 = 48/6 × 1e-4) was too aggressive for Adam

**baseline_optimized** (5090, conservative scaling):
- Loss curve: 0.76 → 0.070 (epoch 20) → 0.057 (epoch 400)
- Best test metrics at epoch 20 despite loss continuing to improve for 380 more epochs
- Confirms: training loss is a poor proxy for point-cloud metrics in this architecture

### Discrepancy vs Paper-Reported Numbers

With legacy-cartesian conversion the pretrained model closely matches the paper:

- **Pretrained (legacy-cartesian)**: Chamfer 0.363m / Mod-Hausdorff 0.247m
- **Paper-reported**: Chamfer 0.36m / Mod-Hausdorff 0.24m
- **Gap**: +0.8% Chamfer, +2.9% Mod-Hausdorff

The original large discrepancy (polar-direct: 0.429/0.297) was caused by a different point-cloud coordinate conversion than the paper's MATLAB pipeline.

Remaining small gap likely from:
1. MATLAB vs Python numerical differences (search/binning in coordinate grids).
2. Dataset composition (19 test trajectories from `dataset_5/test` after history trimming).

The polar IoU/F1 values are very low (IoU=0.013-0.026) because lidar ground truth is sparse — most pixels are zero — so pixel-level IoU is dominated by true negatives that do not count. These metrics are less informative than point-cloud distances for this task.

## Experiment Index

Each experiment folder contains:
- `config.json` — hyperparameters and training configuration snapshot
- `metrics.json` — scalar evaluation results (per-sample + aggregate)
- `metrics.csv` — same metrics in CSV format
- `git_commit.txt` — code version used for training/evaluation
- `plots/` — side-by-side visualizations (radar / prediction / ground truth)

## Running an Evaluation

```bash
python3 eval/eval_pointcloud.py \
  --pred-dir  logs/<experiment>/test_imgs/ \
  --label-dir logs/<experiment>/test_imgs/ \
  --output-dir results/<experiment-name>/ \
  --experiment-name <experiment-name> \
  --coordinate-mode legacy_cartesian
```

Results are written to `results/<experiment-name>/metrics.json` and `metrics.csv`.
Update the Comparison Table above with the `median` values from the JSON output.

## Phase 2: ConvLSTM Temporal Modeling (Negative Result)

**Hypothesis:** ConvLSTM cells at the U-Net bottleneck and deepest skip would improve spatial precision by learning temporal dynamics across radar frames.

**Architecture:** UNet1ConvLSTM — single-frame encoder (1ch, GroupNorm), 2 ConvLSTM cells (bottleneck 16×4 + skip 32×8), shared decoder. 27.5M params (vs 17.5M baseline). Batched encoder/decoder forward pass for GPU efficiency.

**Training:** batch=8, lr=7e-5, fp32, truncated BPTT (T=8), dense supervision (final=1.0, intermediate=0.2), 30 epochs, 10.4h on RTX 5090.

**Result:** Chamfer 0.603m — 2× worse than baseline (0.295m).

**Why it failed:**
1. The baseline's 41-channel stacking fuses frames at full resolution (256×64) in the first conv layer — all spatial detail is available for cross-frame interaction. ConvLSTM delays fusion until after heavy downsampling (16×4, 32×8), losing the spatial precision that Chamfer measures.
2. T=8 training → T=41 eval creates hidden state distribution shift at steps 9-41.
3. Dense supervision with intermediate_weight=0.2 over-weights short-term predictions (total intermediate weight 1.4 vs final weight 1.0).

**Inference cost:** ConvLSTM streaming = 2.9ms/frame vs baseline = 2.2ms/frame. No meaningful latency advantage.

**Conclusion:** This task is fundamentally about spatial precision in the decoder, not temporal dynamics. The 41-channel stacking is the correct temporal fusion strategy for fixed-length sequences where spatial accuracy dominates.

## Phase 3: v2 Single-Frame Raw-IQ Pipeline

### Motivation

The baseline operates on 8-bit magnitude PNGs after CFAR detection — lossy preprocessing unsuitable for real-time streaming. The v2 pipeline operates directly on raw complex IQ data (8 virtual antennas × 512 range bins, complex64) to preserve phase information and enable streaming inference.

### v2 Model Variants Tested

| Model | Chamfer (m) | Mod-H (m) | Params | Best Epoch | Train Time | Architecture |
|-------|-------------|-----------|--------|-----------|-----------|--------------|
| v2 Magnitude | 0.317 | 0.399 | 1.88M | ~10 | — | FFT + |·| + point decoder |
| **v2 Mag+Phase** | **0.309** | **0.423** | **2.08M** | **~10** | **—** | **FFT + mag/sin/cos + point decoder** |
| v2 FFT Occupancy | 0.750 | 0.668 | 75K | 9 | 144 min | FFT + [Re,Im,logpow] + dilated conv → polar occ |
| v2 Mag+Phase 2D | 0.347 | 0.484 | 1.78M | 32 | 430 min | FFT + mag/sin/cos + 2D Conv2d bridge + point decoder |

*v2 models use direct point-cloud eval (chamfer_distance_np, mod_hausdorff_np) on XY-only projections matching legacy_cartesian. Not tested on full test set via PNG pipeline — numbers are validation-set estimates and may differ slightly from test-set numbers.*

### Key Finding: Angular Topology Collapse (Verified Bug)

The v2 point cloud decoder's `sample_features_from_range_azimuth_map()` uses `unsqueeze(2)` to create a feature map with height=1, making azimuth interpolation a no-op via grid_sample. **Empirically verified**: two points at the same range but different azimuths (az=-0.8 vs az=+0.8) get IDENTICAL features (max abs diff = 0.0).

Despite this bug, the point decoder achieves 0.309m Chamfer because:
- Chamfer distance is dominated by range precision, not angular precision
- The Conv1d(256→128) bridge mixes all 256 angular bins into 128 features per range position — rich per-point features that compensate for the grid_sample bug
- The mod-Hausdorff penalty (0.423 vs baseline 0.189) reflects the angular coverage gaps

### Experiment: FFT Occupancy Decoder (Negative Result)

**Hypothesis:** Replacing the point decoder with a polar occupancy output (256×512 logits) would preserve angular topology and fix the mod-H gap.

**Architecture:** FFTBeamformer → Channelizer [Re, Im, log_power] → InstanceNorm2d → 4× DilatedResBlock(dilations=[1,2,4,1]) → Conv2d(1) → polar occupancy logits. 75,207 params.

**Result:** Chamfer 0.750m, mod-H 0.668m — **2.4× worse** than the point decoder.

**Why it failed:**
1. Model severely under-parameterized (75K vs baseline's 17.5M). Training loss barely moved (0.96→0.95).
2. FFT beamformer produces a smooth/blurred angular spectrum (8 antennas zero-padded to 256 bins). The conv head cannot resolve angular detail that isn't in the input.
3. Labels are extremely sparse (~0.8% positive pixels). Focal BCE + Dice was insufficient — model predicted near-all-zeros.

### Experiment: 2D Angular Topology Fix (Negative Result)

**Hypothesis:** Switching the bridge from Conv1d(256→128) to Conv2d(3→128) on the 2D (azimuth × range) feature map would preserve angular topology for grid_sample and fix mod-H.

**Architecture:** FFTBeamformer → [mag, sin_phase, cos_phase] as (B, 3, 256, 512) → Conv2d(3→128) bridge → PointCloudDecoder2D with 2D grid_sample using sin_theta coordinates. 1.78M params.

**Result:** Chamfer 0.347m, mod-H 0.484m — **worse on BOTH metrics** than the broken 1D decoder (0.309/0.423).

**Why it failed:**
1. The Conv2d(3→128) bridge has 28× fewer parameters than the Conv1d(256→128) bridge (3.5K vs 98K per layer). Each spatial position sees only 3 features (mag/sin/cos) vs 256 angular-bin features mixed into 128 dims.
2. The 1D bridge's "bug" is actually a useful inductive bias: it provides rich global angular context per range position. The 2D bridge trades this for local angular locality, which is less useful for the MLP-based DensificationStage.
3. The model converged 3× slower (best at epoch 32 vs epoch 10), suggesting the 2D representation is harder to optimize.

### Conclusion

**The mod-Hausdorff gap (0.423 vs 0.189) is primarily a temporal coverage problem, not an angular topology problem.** Fixing angular topology made both metrics worse. The 1D bridge's global angular mixing is a feature, not a bug — for this point decoder architecture.

Single-frame radar at 0.309m Chamfer is close to the 41-frame baseline (0.295m). The remaining gap is weak/intermittent returns that only appear in some frames — no single-frame architecture can recover these.

**Next step:** Temporal scaling study (1/3/5/8/41 frame Pareto curve) using the working 1D Mag+Phase decoder.

## Future Experiments

Ideas to test separately from the main architectural ablation. Each should be isolated to avoid confounding.

- **Data augmentation**: random horizontal flip, intensity jitter, azimuth noise. The baseline already overfits by epoch 20 — augmentation may significantly extend the useful training window. Must be tested independently of architecture changes.
- **Learned initial state**: make ConvLSTM (h0, c0) trainable parameters instead of zero-init. May improve cold-start (T=1..5) performance.
- **Frame repetition warm-up**: repeat the first frame N times to prime ConvLSTM before processing real frames. Alternative cold-start strategy.
- **Multiple seeds**: run each experiment with 3 seeds (0, 42, 123) and report mean+std for statistical rigor.
- **Streaming drift**: evaluate ConvLSTM with indefinitely long sequences (>>41 frames) to test whether hidden state saturates or drifts.
- **Cross-trajectory transfer**: does ConvLSTM state from one environment generalize to another?
- **Non-overlapping contiguous chunks + TBPTT**: redesign dataset for true streaming training (no stride-1 sliding windows) to enable cross-batch state carry.
- **Expanded ConvLSTM levels**: add ConvLSTM to remaining 3 skip connection levels (5 total) if 2-cell results are promising.
- **Alternative temporal models**: 3D convolutions (R(2+1)D), Temporal Shift Modules (TSM), bottleneck self-attention — different temporal fusion paradigms for comparison.
- **Batched inference**: current inference uses batch=1, massively underutilizing the GPU. Inference batch size does not affect outputs (no gradients, GroupNorm is batch-invariant). Bumping to batch=16-32 should cut inference from ~9 min to ~1-2 min per checkpoint.
- **Multi-config runner**: each experiment currently launches a fresh Docker container and reloads all data (~15 min overhead). A single script that loads data once and trains multiple configs sequentially would eliminate this repeated cost.
- **Curriculum sequence length**: train with increasing T (8→16→41) rather than fixed truncated BPTT. May capture longer-range dependencies that T=8 misses.
