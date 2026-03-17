# mmDar Experiment Results

## Comparison Table (Legacy-Cartesian Eval, Paper-Comparable)

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.363** | **0.247** | 0.026 | 0.051 | 0.119 | 0.033 | Pretrained 120.pt_gen |
| baseline_paper_params | 0.399 | 0.277 | 0.025 | 0.050 | 0.134 | 0.031 | Retrained 200 epochs, batch=6, lr=1e-4, adam |
| baseline_5090_adapted | 0.537 | 0.378 | 0.013 | 0.025 | 0.082 | 0.015 | Retrained 200 epochs, batch=48, lr=8e-4, bf16 |

*All table values are median over 18,575 test samples using `--coordinate-mode legacy_cartesian`. All runs use best.pt_gen (lowest training loss). IoU/F1/precision/recall are unchanged across coordinate modes because they are image-space metrics.*

## Comparison Table (Polar-Direct Eval, Reference Only)

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | Notes |
|------------|-------------|-------------------|-------|
| baseline_pretrained | 0.429 | 0.297 | Polar-direct conversion inflates point-cloud distances |
| baseline_paper_params | 0.492 | 0.341 | |
| baseline_5090_adapted | 0.566 | 0.426 | |

*Polar-direct numbers are retained for reference. The legacy-cartesian pipeline matches the paper's MATLAB eval flow and should be used for all comparisons.*

## Training Run Details

| Experiment | Batch | LR | Mixed Precision | Train Time | Best Epoch Approx |
|------------|-------|-----|-----------------|------------|-------------------|
| baseline_paper_params | 6 | 1e-4 | No (fp32) | ~7.5h | best.pt_gen |
| baseline_5090_adapted | 48 | 8e-4 | Yes (bf16) | ~5.2h | best.pt_gen |

### Convergence Notes

**baseline_paper_params** (paper-exact):
- Loss curve: 0.76 (epoch 0 start) → ~0.054 (best epoch ~101) → plateau ~0.065
- Well-converged, consistent with expected behavior for this architecture
- Training metrics slightly worse than pretrained model despite same hyperparameters
  (hypothesis: pretrained model may have used additional regularization or data augmentation)

**baseline_5090_adapted** (5090-optimized):
- Loss curve: 0.76 (epoch 0 start) → plateau ~0.084-0.094 (epochs 50-199)
- Higher loss plateau than paper-exact run despite same number of epochs
- Linear LR scaling (8e-4 = 48/6 × 1e-4) was too aggressive; model converged to a worse local minimum
- Recommendation: For future runs, use lr=2e-4 to 4e-4 with warmup for large-batch training

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
