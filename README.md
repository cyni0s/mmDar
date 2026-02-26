# mmDar — Radar Super-Resolution via Asymmetric U-Net

mmDar extends [RadarHD (ICRA 2023)](https://arxiv.org/abs/2206.09273) to improve radar-to-lidar
polar image translation. The research goal is to quantifiably improve over the RadarHD baseline
on point-cloud metrics (Chamfer distance, modified Hausdorff) through targeted architectural
enhancements: temporal modeling, attention mechanisms, and advanced loss functions.

Each improvement is isolated and ablated so the contribution of each change is measurable.

## Key Results

Pending evaluation. Planned comparison:

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Notes |
|------------|-------------|-------------------|-----|-----|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.429** | **0.297** | 0.026 | 0.051 | Pretrained model |
| baseline_paper_params | 0.492 | 0.341 | 0.025 | 0.050 | Retrained, paper HP |
| baseline_5090_adapted | 0.566 | 0.426 | 0.013 | 0.025 | Retrained, 5090 GPU |

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
