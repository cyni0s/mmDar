# mmDar — Physics-First Radar-to-Point-Cloud

Class submission for ELEC-6970 (Applied Statistics and Machine Learning, Auburn).
Generates lidar-quality 2D point clouds from raw mmWave radar IQ (TI AWR1843,
77 GHz, 8 virtual antennas).

Starts from the [RadarHD (ICRA 2023)](https://arxiv.org/abs/2206.09273) U-Net
baseline and adds a physics-first Gaussian set prediction model that operates
directly on complex IQ data: classical FFT beamforming (fixed, 0 parameters) →
2D conv encoder → DETR-style decoder → 96 Gaussians in polar coordinates.

**See [`REPORT.pdf`](./REPORT.pdf) for the full write-up.**

## Headline results

From `REPORT.pdf` Table II, on the sealed 18,575-frame low-ID test set
(`v2` split) unless noted, val-selected checkpoints only:

| Model | Params | Chamfer (m) ↓ | mod-H (m) ↓ |
|-------|--------|---------------|-------------|
| **Physics-first Gaussian** | **3.1 M** | **0.318** | **0.230** |
| Honest baseline (UNet1) | 17.5 M | 0.406 | 0.296 |
| Physics-first Gaussian (mixed-ID split) | 3.1 M | 0.280 | 0.205 |

The physics-first model beats the honestly-evaluated baseline by 22% on both
metrics while using 5.6× fewer parameters. See `REPORT.pdf` §IV.B for why
checkpoint selection against the test set produces optimistic results, and how
the "honest" baseline number recovers from the literature's 0.295 m.

## What's in this submission

Included:

1. **All source code** — two pipelines (baseline U-Net in `baseline/`,
   physics-first Gaussian in `train/` + `model/` + `data/`).
2. **`dataset_5/`** — 336 MB of paired radar/lidar PNGs (84,232 frames).
   Sufficient to train and evaluate the baseline U-Net from scratch.
3. **`checkpoints/`** — three pretrained models (~95 MB total) that reproduce
   the three rows of the headline table. See `checkpoints/README.md`.
4. **`data/processed/radar_250.pt`, `lidar_250.pt`, `norm_250.pt`** — one
   preprocessed trajectory (366 frames, 46 MB) so the Gaussian model can be
   demonstrated end-to-end without the raw dataset.
5. **`REPORT.pdf`** — final IEEE-format report (and `reports/initial_report/`
   has the .tex source + figures).

Not included (too large for GitHub):

- The full raw RadarHD dataset (~400 GB uncompressed / 83 GB compressed as
  `RadarHD-dataset.zip`). Required **only** for retraining the Gaussian
  model from scratch. Download from
  [akarsh-prabhakara/RadarHD](https://github.com/akarsh-prabhakara/RadarHD)
  following their README.

## System requirements

- Linux, NVIDIA GPU (≥ 8 GB VRAM — tested on RTX 5090)
- Docker with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- ~1.5 GB free disk for the repo
- ~42 GB additional if retraining the Gaussian model from raw data
- Every command in this README runs inside the Docker container via
  `docker compose run --rm mmdar …`. Nothing is installed on the host.

## Setup

```bash
git clone <repo-url>
cd mmDar
docker compose build          # ~10 min first time (NGC 25.02-py3 + pytorch3d)
```

## Quick start — reproduce a headline number with a pretrained checkpoint

### Physics-first Gaussian on the shipped demo trajectory (~3 min on RTX 5090)

```bash
docker compose run --rm mmdar python3 tools/run_inference.py \
    --checkpoint checkpoints/physics_gaussian_headline.pt \
    --trajectories 250 \
    --split v2 \
    --output results/demo_gaussian/
```

Output: `results/demo_gaussian/metrics.json` with per-frame Chamfer and
modified Hausdorff. Trajectory 250 is a hard high-ID test case — its per-frame
medians will be **worse** than the aggregate 0.318 / 0.230 that the checkpoint
achieves on the full 19-trajectory test set (smoke-tested: Chamfer ≈ 0.52,
mod-H ≈ 0.55 on this trajectory alone). This run verifies the model, the
wrapper, and the preprocessed-IQ pipeline all work end-to-end on a fresh clone.

To reproduce the full headline number you need all 44 preprocessed trajectories
— see "Full reproduction" below.

### Baseline U-Net on the full sealed test set (~9 min on RTX 5090)

The baseline uses the upstream `test_radarhd.py` entry point, which expects
the checkpoint at a fixed path:

```bash
mkdir -p logs/baseline_honest/
cp checkpoints/baseline_honest.pt_gen logs/baseline_honest/best.pt_gen

docker compose run --rm mmdar python3 baseline/test_radarhd.py
docker compose run --rm mmdar python3 eval/pol_to_cart.py
docker compose run --rm mmdar python3 -m eval.eval_pointcloud \
    --pred-dir logs/baseline_honest/test_imgs/pred/ \
    --label-dir logs/baseline_honest/test_imgs/label/ \
    --output-dir results/baseline_honest_eval/
```

Expected: Chamfer ≈ 0.406 m, mod-H ≈ 0.296 m on the 18,575-frame test set.

## Reproduce the baseline U-Net from scratch (no raw dataset needed)

```bash
docker compose run --rm mmdar python3 baseline/train_honest.py
```

Trains 50 epochs (~80 min on RTX 5090) from the `dataset_5/` PNGs that ship
with the repo, selects the best epoch on validation mod-H, and writes
`logs/baseline_honest/best.pt_gen`. Then run the baseline inference + eval
commands from the Quick Start section above.

Hyperparameters (hardcoded at the top of `baseline/train_honest.py`):
batch 12, lr 7 × 10⁻⁵, fp32, 50 epochs, 41-frame history.

## Reproduce the Gaussian model from scratch (requires raw dataset)

### 1. Get the raw RadarHD dataset

Download from
[github.com/akarsh-prabhakara/RadarHD](https://github.com/akarsh-prabhakara/RadarHD).
Unpack it somewhere, then point `RADARHD_RAW` at the unpacked directory:

```bash
export RADARHD_RAW=/absolute/path/to/RadarHD-dataset
```

### 2. Preprocess raw IQ + lidar into `.pt` tensors

(~2 h on RTX 5090, outputs ~42 GB to `data/processed/`)

```bash
docker compose run --rm -e RADARHD_RAW mmdar python3 -m data.preprocess \
    --raw-dir $RADARHD_RAW --output-dir data/processed/
```

Writes 44 × {`radar_<id>.pt`, `lidar_<id>.pt`, `norm_<id>.pt`} plus
`frame_table.json`. (Trajectory 250's files ship with the repo; preprocessing
will overwrite them identically.)

### 3. Fit ground-truth prototypes (~5 min)

```bash
docker compose run --rm mmdar python3 -m train.train --fit-prototypes
```

### 4. Train (~3 h on RTX 5090, 50 epochs)

```bash
# Headline result — low-ID split → Chamfer 0.318 / mod-H 0.230
docker compose run --rm mmdar python3 -m train.train \
    --train --split v2 --augment

# Mixed-ID best → Chamfer 0.280 / mod-H 0.205
docker compose run --rm mmdar python3 -m train.train \
    --train --split mixed --augment
```

Defaults: batch 4, lr 1 × 10⁻⁴, window 41, K 96, N_az 64,
σ_r prior 0.3, Huber range weight 0.1 (matches exp4 of the report).

### 5. Evaluate the newly-trained model

```bash
docker compose run --rm mmdar python3 tools/run_inference.py \
    --checkpoint logs/v2_gaussian/best.pt \
    --split v2 --trajectories all \
    --output results/my_run/
```

## Best parameters (summary)

| Report row | Shipped checkpoint | Reproduce-from-scratch command | Wall time (RTX 5090) |
|---|---|---|---|
| Physics Gaussian headline (0.318 / 0.230) | `checkpoints/physics_gaussian_headline.pt` | `python3 -m train.train --train --split v2 --augment` | ~3 h |
| Physics Gaussian mixed-ID (0.280 / 0.205) | `checkpoints/physics_gaussian_mixed.pt` | `python3 -m train.train --train --split mixed --augment` | ~3 h |
| Honest baseline (0.406 / 0.296) | `checkpoints/baseline_honest.pt_gen` | `python3 baseline/train_honest.py` | ~80 min |

## Project layout

```
mmDar/
├── REPORT.pdf                     # Final report (copy of reports/initial_report/report.pdf)
├── README.md                      # This file
├── Dockerfile, docker-compose.yml, requirements.txt, install.sh
├── dataset_5/                     # 336 MB paired radar/lidar PNGs — baseline data
├── checkpoints/                   # Three pretrained models — see checkpoints/README.md
├── data/
│   ├── processed/                 # Demo trajectory 250 + frame_table.json
│   ├── preprocess.py              # Raw RadarHD → .pt pipeline
│   ├── windowed_dataset.py        # Dataset abstraction for 41-frame windows
│   └── split{,_v2,_mixed}.py      # Train/val/test trajectory IDs
├── model/                         # Gaussian model components
│   ├── physics_frontend.py        # Classical FFT + 2D encoder + full pipeline
│   ├── gaussian_head.py           # DETR decoder + Gaussian heads
│   ├── beamspace.py, lista.py     # Alternative beamforming stages
│   └── cvnn.py                    # Complex-valued layers
├── train/
│   ├── train.py                   # Gaussian model training entry point
│   └── loss_gaussian.py           # Hungarian NLL + auxiliary terms
├── baseline/
│   ├── train_radarhd.py           # Upstream RadarHD training (unchanged)
│   ├── train_honest.py            # Honest-evaluation variant (val-based selection)
│   └── test_radarhd.py            # Upstream inference entry point
├── eval/                          # Metric computation (Chamfer, mod-H, IoU, F1)
├── tools/run_inference.py         # End-to-end Gaussian inference CLI
├── train_test_utils/              # U-Net model, dataloader, Dice loss
├── tests/                         # pytest suite
├── create_dataset/                # Upstream raw-sensor processing (reference only)
└── reports/initial_report/        # Report .tex + figures + references.bib
```

## Citation

Upstream RadarHD:

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

For the physics-first Gaussian pipeline and the honest-evaluation findings in
this submission, see [`REPORT.pdf`](./REPORT.pdf).

## Team

Chris Turner, Andreas Zeck, Justin Palm — ELEC-6970, Auburn University.
