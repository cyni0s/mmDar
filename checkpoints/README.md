# Pretrained Checkpoints

Three checkpoints that reproduce the three headline rows of Table II in
`../REPORT.pdf`. Each row of the table maps to exactly one file here.

| Checkpoint | Model | Split | Chamfer | mod-H | Params |
|------------|-------|-------|---------|-------|--------|
| `physics_gaussian_headline.pt` | Physics-first Gaussian (augmented) | low-ID `v2` (17/8/19) | **0.318 m** | **0.230 m** | 3.1 M |
| `physics_gaussian_mixed.pt`    | Physics-first Gaussian (σ=0.3, Huber=0.1) | mixed-ID (25/6/13) | **0.280 m** | **0.205 m** | 3.1 M |
| `baseline_honest.pt_gen`       | RadarHD U-Net (UNet1)              | low-ID `v2` (17/8/19) | 0.406 m    | 0.296 m   | 17.5 M |

All three checkpoints are val-selected (not test-selected). See report §III for
the methodological discussion of checkpoint selection bias.

## Running inference

### Gaussian models

```bash
docker compose run --rm mmdar python3 tools/run_inference.py \
    --checkpoint checkpoints/physics_gaussian_headline.pt \
    --trajectories 250 \
    --output results/demo_gaussian/
```

Flags:
- `--checkpoint`: path to either Gaussian `.pt`
- `--trajectories`: comma-separated list of trajectory IDs, or `all` for the
  full test set of the split
- `--split`: `v2` (default) or `mixed` — determines which test trajectories
  are valid for `--trajectories all`
- `--output`: results directory (metrics.json + per-frame rows)

### Baseline U-Net

The baseline uses its own inference script:

```bash
# Move the checkpoint into place for test_radarhd.py
mkdir -p logs/baseline_honest/
cp checkpoints/baseline_honest.pt_gen logs/baseline_honest/best.pt_gen

docker compose run --rm mmdar python3 baseline/test_radarhd.py
```

## Checkpoint provenance

- `physics_gaussian_headline.pt` — output of
  `python3 -m train.train --train --split v2 --augment` at the code state that
  produced report Table II row 1. The hyperparameters at save time are pinned
  in `config_headline.json`.
- `physics_gaussian_mixed.pt` — output of
  `python3 -m train.train --train --split mixed --augment` with current
  defaults (σ_r prior 0.3, Huber range weight 0.1). Pinned in `config_mixed.json`.
- `baseline_honest.pt_gen` — output of `python3 baseline/train_honest.py`
  (val-selected, batch 12, lr 7e-5, fp32, 50 epochs).
