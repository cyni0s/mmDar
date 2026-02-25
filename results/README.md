# mmDar Experiment Results

## Comparison Table

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | **0.429** | **0.297** | 0.026 | 0.051 | 0.119 | 0.033 | Pretrained model eval; see discrepancy note below |
| baseline_paper_params | — | — | — | — | — | — | Retrained, paper-exact HP |
| baseline_5090_adapted | — | — | — | — | — | — | Retrained, 5090-optimized |

*Table values are median over 18,575 test samples. Dashes indicate pending evaluation.*

### Discrepancy vs Paper-Reported Numbers

The pretrained model produces higher (worse) point-cloud distances than the paper reports:
- Chamfer: 0.429 m measured vs 0.36 m reported (+19%)
- Mod-Hausdorff: 0.297 m measured vs 0.24 m reported (+24%)

**Possible explanations (not yet isolated):**
1. **Evaluation pipeline difference**: The paper may use a different threshold value or coordinate mapping when converting polar images to point clouds. Our eval uses `threshold=1` (any non-zero pixel) matching `image_to_pcd.py`.
2. **PyTorch version**: Inference run with PyTorch 2.7.0 vs the original training environment (likely PyTorch 1.x). BatchNorm statistics or floating-point behavior may differ slightly.
3. **Dataset split**: The paper may evaluate on a different subset or use a different test/train split than `dataset_5/test/`.
4. **Metric aggregation**: The paper may report mean rather than median, or exclude certain sequences.

The polar IoU/F1 values are very low (IoU=0.026) because lidar ground truth is sparse — most pixels are zero — so pixel-level IoU is dominated by true negatives that do not count. These metrics are less informative than point-cloud distances for this task.

## Experiment Index

Each experiment folder contains:
- `config.json` — hyperparameters and training configuration snapshot
- `metrics.json` — scalar evaluation results
- `metrics.csv` — same metrics in CSV format
- `git_commit.txt` — code version used during training
- `plots/` — side-by-side visualizations (radar / prediction / ground truth)

## Running an Evaluation

```bash
python3 eval/eval_pointcloud.py \
  --pred-dir  <experiment-dir>/pred/ \
  --label-dir <experiment-dir>/label/ \
  --output-dir results/<experiment-name>/ \
  --experiment-name <experiment-name>
```

Results are written to `results/<experiment-name>/metrics.json` and `metrics.csv`.
Update the Comparison Table above with the `median` values from the JSON output.
