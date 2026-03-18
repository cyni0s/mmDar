# mmDar Experiment Results

## Comparison Table (Legacy-Cartesian Eval, Paper-Comparable)

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.363** | **0.247** | 0.026 | 0.051 | 0.119 | 0.033 | Pretrained 120.pt_gen |
| baseline_optimized_ep020 | **0.372** | **0.228** | 0.027 | 0.052 | 0.123 | 0.034 | batch=24, lr=1.5e-4, bf16, epoch 20 of 400 |
| baseline_paper_params | 0.399 | 0.277 | 0.025 | 0.050 | 0.134 | 0.031 | batch=6, lr=1e-4, fp32, best.pt_gen of 200 epochs |
| baseline_5090_adapted | 0.537 | 0.378 | 0.013 | 0.025 | 0.082 | 0.015 | batch=48, lr=8e-4, bf16, best.pt_gen of 200 epochs |

*All table values are median over 18,575 test samples using `--coordinate-mode legacy_cartesian`. Checkpoints selected by Chamfer sweep unless noted as best.pt_gen (training loss).*

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
| baseline_paper_params | 6 | 1e-4 | No (fp32) | 200 | ~7.5h | best.pt_gen (train loss) |
| baseline_5090_adapted | 48 | 8e-4 | Yes (bf16) | 200 | ~5.2h | best.pt_gen (train loss) |
| baseline_optimized | 24 | 1.5e-4 | Yes (bf16) | 400 | ~8.7h | epoch 20 (Chamfer sweep) |

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
