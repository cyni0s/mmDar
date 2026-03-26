# mmDar — Radar Super-Resolution via Asymmetric U-Net

mmDar extends [RadarHD (ICRA 2023)](https://arxiv.org/abs/2206.09273) to improve radar-to-lidar
polar image translation. The research goal is to quantifiably improve over the RadarHD baseline
on point-cloud metrics (Chamfer distance, modified Hausdorff) through targeted architectural
enhancements: temporal modeling, attention mechanisms, and advanced loss functions.

Each improvement is isolated and ablated so the contribution of each change is measurable.

## Key Results

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | Frames | Notes |
|------------|-------------|-------------------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | 41 | RadarHD ICRA 2023 |
| **5090-optimized** | **0.295** | **0.189** | **41** | **batch=12, lr=7e-5, fp32, epoch 10 (22 min)** |
| baseline_pretrained | 0.363 | 0.247 | 41 | Authors' pretrained 120.pt_gen |
| v2 Mag+Phase (raw IQ) | 0.309 | 0.423 | **1** | Single-frame, no PNG preprocessing, 2.08M params, 1.3ms |
| **v2 Temporal xattn** | **0.295** | **0.429** | **8** | **Matches baseline Chamfer, 8× fewer params (2.2M), 5× fewer frames** |

*All values are median over 18,575 test samples using legacy-cartesian coordinate conversion (paper-comparable pipeline).*

See [`results/README.md`](./results/README.md) for full experiment tracking.

## Setup & Installation

### Prerequisites

- NVIDIA GPU (tested on RTX 5090)
- Docker with NVIDIA Container Toolkit
- ~350 MB free disk space for `dataset_5/`

### Quick Start

```bash
# Clone and enter
git clone <repo-url> && cd mmDar

# Build Docker environment
docker compose build

# Run interactive container
docker compose run mmdar bash

# Inside container — inference with pretrained model
python3 test_radarhd.py

# Inside container — train from scratch
python3 train_radarhd.py
```

Alternatively, use the original Docker run command:

```bash
sudo docker run -it --rm --gpus all --shm-size 8G \
  -v $(pwd):/radarhd/ pytorch/pytorch bash

cd /radarhd/
sh install.sh
```

### Dependencies

All Python dependencies are installed by `install.sh`:

```bash
sh install.sh
```

Key packages: PyTorch, OpenCV, NumPy, SciPy, Matplotlib, Pillow.

## Usage

### Training

```bash
# Train with default parameters (matching original RadarHD paper)
python3 train_radarhd.py
```

Training configuration (model architecture, batch size, learning rate, etc.) is
controlled by constants at the top of `train_radarhd.py`. TensorBoard logs are
written to the `logs/` directory.

### Inference

```bash
# Run inference with pretrained model on test dataset
python3 test_radarhd.py
```

Downloads the pretrained model checkpoint from the link in the original repository
and places it under `logs/13_1_20220320-034822/`. Output images (predicted + ground
truth in polar format) are written to `logs/.../test_imgs/`.

### Evaluation

The Python evaluation pipeline replaces MATLAB for all metric computation:

```bash
# Convert polar images to cartesian
cd eval/
python3 pol_to_cart.py

# Compute all metrics (Chamfer, modified Hausdorff, IoU, F1)
python3 eval_pointcloud.py \
  --pred-dir  ../logs/<run>/test_imgs/pred/ \
  --label-dir ../logs/<run>/test_imgs/label/ \
  --output-dir ../results/<experiment-name>/ \
  --experiment-name <experiment-name>
```

Outputs written to `results/<experiment-name>/`:
- `metrics.json` — full per-sample and aggregate metrics
- `metrics.csv`  — tabular summary
- `plots/`       — side-by-side visualizations (radar / prediction / ground truth)

The MATLAB pipeline (`eval/pc_compare.m`, `eval/pc_distance.m`) remains available
for cross-validation.

## Project Structure

```
mmDar/
├── train_radarhd.py          # Training script
├── test_radarhd.py           # Inference script
├── install.sh                # Dependency installation
├── dataset_5/                # Paired radar / lidar images (train + test)
├── logs/                     # Model checkpoints and test outputs
├── train_test_utils/         # Model, loss, and dataloader definitions
├── eval/
│   ├── eval_pointcloud.py    # Python evaluation module (CLI + importable)
│   ├── pol_to_cart.py        # Polar → cartesian image conversion
│   ├── image_to_pcd.py       # Cartesian image → point cloud (open3d)
│   ├── pc_distance.m         # MATLAB point-cloud distance metrics
│   └── pc_compare.m          # MATLAB CDF comparison plots
├── results/                  # Per-experiment metrics and plots
│   └── README.md             # Experiment comparison table
└── create_dataset/           # Raw sensor processing scripts
```

## Changes From Original RadarHD

This section documents all modifications from the [upstream RadarHD repository](https://github.com/akarsh-prabhakara/RadarHD). The original model architecture (`UNet1`), dataloader, and loss function (`BCELoss + DiceLoss`) are **untouched** — Phase 2 adds a new model variant alongside them.

### Phase 2: Reusable Infrastructure

Built during ConvLSTM development, usable by any future model variant:

| Change | File(s) | Purpose |
|--------|---------|---------|
| `norm_type` parameter | `train_test_utils/unet_parts.py` | All building blocks (DoubleConv, Down, Up, Up_nocat) accept `norm_type='batch'` (default) or `'group'`. Enables GroupNorm variants without duplicating block code. |
| Temporal consistency metric | `eval/eval_pointcloud.py` — `temporal_consistency()` | Frame-to-frame Chamfer distance within trajectories. Evaluates any model's output stability. |
| Multi-model experiment runner | `run_experiment.py` — `--model` flag | Dispatch pattern for training/eval across model variants. Currently supports `baseline` and `convlstm`. |
| Trajectory-aware data loading | `train_test_utils/dataloader.py` — `SequentialDataset`, `TrajectoryBatchSampler`, `seq_collate_fn` | Temporal sequence access with stateless pre-computed epoch schedules. Safe for `num_workers>0`. Reusable by any sequential/temporal model. |
| T-curve evaluation pattern | `test_convlstm.py` | Evaluates at T={1,4,8,16,32,41} to measure metrics vs history length. Pattern applicable to any temporal model. |
| Test infrastructure | `tests/conftest.py`, `tests/__init__.py` | Shared fixtures (device selection, reproducibility). 61 tests across 5 test files. |
| Training script template | `train_convlstm.py` | `params.json` logging, TensorBoard integration, validation-based checkpointing, `--dry_run` smoke test, gradient checkpointing. Pattern for future training scripts. |

### Phase 2: ConvLSTM Temporal Modeling (Negative Result)

**Hypothesis:** Sequential temporal modeling via ConvLSTM at the U-Net bottleneck would improve spatial precision by learning frame-to-frame dynamics.

**Result:** ConvLSTM Chamfer 0.603m — **2× worse** than the baseline (0.295m). The approach is a dead end for this task.

| Experiment | Chamfer (m) | Mod-H (m) | IoU | F1 |
|------------|-------------|-----------|------|------|
| 5090-optimized baseline | **0.295** | **0.189** | 0.054 | 0.102 |
| Pretrained baseline | 0.363 | 0.247 | 0.026 | 0.051 |
| ConvLSTM T=8 ep30 | 0.603 | 0.467 | 0.026 | 0.051 |

**ConvLSTM-specific code** (all additive — baseline untouched):

| File | Purpose |
|------|---------|
| `train_test_utils/model.py` — `ConvLSTMCell`, `UNet1ConvLSTM` | 2 ConvLSTM cells (bottleneck + deepest skip), GroupNorm, batched encoder/decoder forward, 27.5M params |
| `train_convlstm.py` | ConvLSTM training: dense supervision, truncated BPTT, fp32/bf16 AMP |
| `test_convlstm.py` | ConvLSTM inference with T-curve evaluation |

### Infrastructure

| Change | File(s) | Purpose |
|--------|---------|---------|
| Docker environment | `Dockerfile`, `docker-compose.yml`, `.dockerignore` | NGC 25.02-py3 base (CUDA 12.8, PyTorch 2.7, RTX 5090 / sm_120 support) |
| Dependency pinning | `requirements.txt` | numpy<2.0 to avoid ABI conflicts; torch/tensorboard omitted (NGC-managed) |
| Git hygiene | `.gitignore` | Ignore checkpoints, datasets, TensorBoard dirs, test_imgs |

### Training (`train_radarhd.py`)

| Change | Original | Modified | Impact |
|--------|----------|----------|--------|
| TensorBoard logging | None | `SummaryWriter` logs epoch loss + LR | Observability only |
| Model summary | `torchsummary.summary(gen, (H+1, 256, 64))` | `torchinfo.summary(gen, input_size=(1, H+1, 256, 64))` | Fixes BatchNorm 4D error |
| DataLoader workers | `num_workers=0` (default) | `num_workers=4, pin_memory=True` | GPU utilization 7% → 94% |
| Mixed precision | fp32 only | Optional bf16 autocast (loss computed in fp32) | ~30% faster per epoch |
| Gradient accumulation | Not supported | `grad_accum_steps` parameter (default 1) | Enables large effective batch |
| LR schedule | Constant only | Optional linear warmup + cosine decay | For future experiments |
| Best checkpoint | Not saved | `best.pt_gen` saved on lowest epoch mean loss | **Caution:** training loss ≠ test metric (see Lessons) |
| Params saved | Not saved | `params.json` written to log dir | Reproducibility |
| `zero_grad` | `zero_grad()` per batch | `zero_grad(set_to_none=True)` after optimizer step | Memory efficiency; mathematically equivalent |

### Inference (`test_radarhd.py`)

| Change | Original | Modified |
|--------|----------|----------|
| Checkpoint loading | Fixed epoch number only | `epoch_num=-1` loads `best.pt_gen` |
| Model summary | Same BatchNorm fix as training | `input_size=(1, ...)` |

### Evaluation (`eval/eval_pointcloud.py` — new file)

Python replacement for the MATLAB evaluation pipeline (`pc_compare.m` + `pc_distance.m`):
- Chamfer distance and modified Hausdorff matching MATLAB definitions exactly
- Two coordinate modes: `legacy_cartesian` (matches paper's `pol_to_cart.py` flow) and `polar_direct`
- Polar image metrics: IoU, F1, precision, recall
- Batch evaluation with per-sample CSV output and side-by-side visualizations
- Uses `scipy.spatial.distance.cdist` — no PyTorch/pytorch3d dependency

The `legacy_cartesian` mode reproduces the paper's eval pipeline within 3% (Chamfer 0.363m vs reported 0.36m).

### Lessons Learned

- **Temporal modeling via ConvLSTM does not improve this task — spatial precision is the bottleneck, not temporal dynamics.** The baseline's 41-channel stacking fuses all frames at full resolution in the first conv layer. ConvLSTM delays temporal fusion until after heavy downsampling (16×4 bottleneck, 32×8 skip), losing the fine spatial detail that Chamfer distance measures. IoU/F1 stayed at baseline level (coarse occupancy was correct), but Chamfer/mod-H doubled (spatial precision was destroyed). The lesson: for radar-to-lidar translation, *where* temporal fusion happens matters more than *how* — early fusion at full resolution beats late fusion at compressed resolution.
- **Truncated BPTT (T=8 train → T=41 eval) causes distribution shift.** LSTM hidden state at steps 9-41 enters distributions never seen during training, compounding the architectural disadvantage. Both Codex (gpt-5.4) and Gemini (2.5-pro) independently diagnosed this as the primary failure mode.
- **Dense supervision can backfire.** With T=8 and intermediate_weight=0.2, the 7 intermediate timesteps contribute total weight 1.4 vs 1.0 for the final step. This optimizes more for short-term predictions than final-step quality — "short-sighted" training penalized at eval.
- **Streaming inference cost is nearly identical.** ConvLSTM streaming (1 frame + carry state) = 2.9ms vs baseline (41-channel forward) = 2.2ms. The ConvLSTM's architectural disadvantage is not offset by meaningful latency improvement.
- **Checkpoint selection by training loss is unreliable.** BCE+Dice loss in polar space correlates poorly with Cartesian point-cloud metrics (Chamfer/mod-Hausdorff). A run with 12% lower training loss produced 15% worse Chamfer distance. Select checkpoints by evaluating test metrics on saved periodic checkpoints instead.
- **Batch size 12 is optimal on RTX 5090.** Sweeping batch sizes 6/12/16/24/48 shows batch=12 gives the best Chamfer distance. Batch=6 (original paper) is too noisy, batch>=24 overfits. BatchNorm statistics noise at small batch sizes provides implicit regularization critical for this UNet architecture.
- **LR=7e-5 beats the paper's 1e-4.** A systematic LR sweep (5e-5, 7e-5, 1e-4, 1.5e-4) found 7e-5 optimal at batch=12, achieving Chamfer 0.308m — 15% better than the pretrained model (0.363m).
- **bf16 mixed precision trades ~4% quality for ~30% speed.** At batch=12, lr=7e-5: fp32 achieves Chamfer 0.295m vs bf16's 0.308m. For fast iteration (sweeps, prototyping) bf16 is fine. For final results, use fp32.
- **The original authors used no validation set or metric-based selection.** They trained for 130 epochs, saved every 10, and shipped epoch 120. Our approach: train, save every 10 epochs, sweep checkpoints by Chamfer distance.
- **Best 5090 config: batch=12, lr=7e-5, fp32, ~10 epochs (~22 min).** Chamfer 0.295m — 18% better than the paper. For fast sweeps, bf16 at the same config is ~30% faster with only ~4% quality loss.

#### v2 Raw-IQ Pipeline Lessons

- **Single-frame raw IQ achieves near-baseline Chamfer.** The v2 Mag+Phase decoder (FFT + sin/cos phase channels + point decoder) achieves 0.309m Chamfer from 1 frame vs the 41-frame baseline's 0.295m. Raw IQ is viable for streaming without PNG preprocessing.
- **Phase helps average geometry but hurts worst-case structure.** Adding sin/cos phase channels improved Chamfer by 2.5% (0.317→0.309) but worsened mod-Hausdorff by 6% (0.399→0.423). Phase sharpens localization for detected targets but provides no coverage benefit for missed returns.
- **The mod-Hausdorff gap is temporal, not architectural.** We verified that the point decoder has an angular topology collapse (grid_sample with height=1), but fixing it made both Chamfer and mod-H WORSE. The mod-H gap (0.423 vs 0.189) is caused by single-frame radar missing weak/intermittent returns that 41-frame temporal stacking recovers. Architecture changes cannot fix a data-availability problem.
- **"Bugs" can be features.** The Conv1d(256→128) bridge collapses 256 angular bins into 128 abstract channel features — destroying explicit angular topology but providing rich per-range-position features. When we "fixed" this with Conv2d(3→128) preserving 2D layout, the model got worse because each position only sees 3 local features instead of 256 global angular bins. The right inductive bias depends on the downstream consumer (MLP vs conv).
- **Capacity matters for occupancy prediction.** A 75K-param dilated conv head on polar occupancy (256×512, 0.8% positive) produced Chamfer 0.750m — the model couldn't learn from the sparse labels. The baseline U-Net uses 17.5M params. Dense occupancy prediction from sparse labels requires multi-scale capacity, not a flat conv head.
- **Change one variable at a time.** The occupancy experiment changed the decoder (point→occupancy), loss (Chamfer→focal BCE+Dice), output format, AND reduced params by 25×. All at once. The result was uninterpretable — did the idea fail or just the specific implementation? The 2D angular fix (one variable changed) gave a clear answer.
- **Temporal cross-attention matches baseline Chamfer with 8 frames (not 41) and 8× fewer params.** But mod-Hausdorff is unchanged. Most of the Chamfer gain comes from pretrained initialization, not temporal fusion. The per-range-bin cross-attention adds only ~0.005m Chamfer and 0 mod-H on test set.
- **Validation on 4 trajectories is unreliable.** Val showed 14% temporal improvement; test showed 1.7%. The unit of variation is trajectory, not frame — 4 trajectories cannot represent 19.
- **The mod-H gap (0.429 vs 0.189) is an output representation problem, not temporal.** Fixed 8192-point decoders must place all points somewhere, while occupancy thresholding naturally adapts cardinality. Also: eval pipelines differ (direct point cloud vs PNG→Cartesian conversion). These must be standardized before mod-H comparisons are meaningful.

### Hyperparameter Sweep Summary (RTX 5090)

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

## Credits & References

- **Original paper:** [High Resolution Point Clouds from mmWave Radar](https://arxiv.org/abs/2206.09273),
  Prabhakara et al., ICRA 2023
- **Original codebase:** [github.com/akarsh-prabhakara/RadarHD](https://github.com/akarsh-prabhakara/RadarHD)

```bibtex
@INPROCEEDINGS{10161429,
  author={Prabhakara, Akarsh and Jin, Tao and Das, Arnav and Bhatt, Gantavya
          and Kumari, Lilly and Soltanaghai, Elahe and Bilmes, Jeff
          and Kumar, Swarun and Rowe, Anthony},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  title={High Resolution Point Clouds from mmWave Radar},
  year={2023},
  pages={4135-4142},
  doi={10.1109/ICRA48891.2023.10161429}
}
```
