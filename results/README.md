# mmDar Experiment Results

## Comparison Table

| Experiment | Chamfer (m) | Mod-Hausdorff (m) | IoU | F1 | Precision | Recall | Notes |
|------------|-------------|-------------------|-----|-----|-----------|--------|-------|
| Paper (reported) | 0.36 | 0.24 | — | — | — | — | RadarHD ICRA 2023 |
| baseline_pretrained | — | — | — | — | — | — | Pretrained model eval |
| baseline_paper_params | — | — | — | — | — | — | Retrained, paper-exact HP |
| baseline_5090_adapted | — | — | — | — | — | — | Retrained, 5090-optimized |

*Table updated after each experiment run. Dashes indicate pending evaluation.*

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
