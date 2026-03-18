# mmDar — Radar Super-Resolution via Asymmetric U-Net

mmDar extends [RadarHD (ICRA 2023)](https://arxiv.org/abs/2206.09273) to improve radar-to-lidar
polar image translation. The research goal is to quantifiably improve over the RadarHD baseline
on point-cloud metrics (Chamfer distance, modified Hausdorff) through targeted architectural
enhancements: temporal modeling, attention mechanisms, and advanced loss functions.

Each improvement is isolated and ablated so the contribution of each change is measurable.

## Key Results

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Notes |
|------------|-------------|-------------------|-----|-----|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.363** | **0.247** | 0.026 | 0.051 | Pretrained 120.pt_gen |
| baseline_optimized_ep020 | **0.372** | **0.228** | 0.027 | 0.052 | batch=24, lr=1.5e-4, bf16, epoch 20 |
| baseline_paper_params | 0.399 | 0.277 | 0.025 | 0.050 | batch=6, lr=1e-4, fp32, best.pt_gen |
| baseline_5090_adapted | 0.537 | 0.378 | 0.013 | 0.025 | batch=48, lr=8e-4, bf16, best.pt_gen |

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

This section documents all modifications from the [upstream RadarHD repository](https://github.com/akarsh-prabhakara/RadarHD). The model architecture (`UNet1`), dataloader, and loss function (`BCELoss + DiceLoss`) are **untouched**.

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

- **Checkpoint selection by training loss is unreliable.** BCE+Dice loss in polar space correlates poorly with Cartesian point-cloud metrics (Chamfer/mod-Hausdorff). A run with 12% lower training loss produced 15% worse Chamfer distance. Select checkpoints by evaluating test metrics on saved periodic checkpoints instead.
- **Batch size affects regularization.** The original batch=6 provides noisy gradients that implicitly regularize via BatchNorm statistics. Scaling to batch=24 or 48 reduces this noise, leading to sharper minima that overfit. For this architecture, batch=6 appears optimal.
- **The original authors used no validation set or metric-based selection.** They trained for 130 epochs, saved every 10, and shipped epoch 120. Matching this approach (periodic saves + metric-based selection from candidates) is more effective than our `best.pt_gen` strategy.

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
