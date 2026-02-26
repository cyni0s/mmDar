# mmDar Experiment Results

## Comparison Table

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.429** | **0.297** | 0.026 | 0.051 | 0.119 | 0.033 | Pretrained 120.pt_gen; see discrepancy note |
| baseline_paper_params | 0.492 | 0.341 | 0.025 | 0.050 | 0.134 | 0.031 | Retrained 200 epochs, batch=6, lr=1e-4, adam |
| baseline_5090_adapted | 0.566 | 0.426 | 0.013 | 0.025 | 0.082 | 0.015 | Retrained 200 epochs, batch=48, lr=8e-4, bf16 |

*Table values are median over 18,575 test samples. All runs use best.pt_gen (lowest training loss).*

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

The pretrained model produces higher (worse) point-cloud distances than the paper reports:
- Chamfer: 0.429 m measured vs 0.36 m reported (+19%)
- Mod-Hausdorff: 0.297 m measured vs 0.24 m reported (+24%)

Retrained models (paper-exact and 5090-adapted) are within 15% and 32% of pretrained respectively.

**Possible explanations (not yet isolated):**
1. **Evaluation pipeline difference**: The paper may use a different threshold value or coordinate mapping when converting polar images to point clouds. Our eval uses `threshold=1` (any non-zero pixel) matching `image_to_pcd.py`.
2. **PyTorch version**: Inference run with PyTorch 2.7.0 vs the original training environment (likely PyTorch 1.x). BatchNorm statistics or floating-point behavior may differ slightly.
3. **Dataset split**: The paper may evaluate on a different subset or use a different test/train split than `dataset_5/test/`.
4. **Metric aggregation**: The paper may report mean rather than median, or exclude certain sequences.

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
  --experiment-name <experiment-name>
```

Results are written to `results/<experiment-name>/metrics.json` and `metrics.csv`.
Update the Comparison Table above with the `median` values from the JSON output.
