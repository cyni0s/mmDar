# GT Standardization Experiment Design

## Problem

The v2 temporal model matches baseline Chamfer (0.295m) but has 2.3x worse mod-Hausdorff (0.429 vs 0.186). Phase 6 proved the 8192-point cardinality is NOT the cause (baseline uses only ~2874 pred / ~665 GT points).

The remaining suspect is **eval pipeline mismatch**: baseline GT = ~665 grid-quantized points (4.2cm spacing, from PNG rasterization); v2 GT = 8192 continuous points (from FPS of raw lidar). The v2 model must match 12x denser GT at higher spatial precision.

## Experiment

Evaluate v2 temporal xattn (N=8 frames) predictions under four GT conditions:

| ID | Pred | GT | Tests |
|----|------|-----|-------|
| Control | 8192 continuous | 8192 continuous | Reproduce 0.295/0.429 |
| A | 8192 continuous | FPS(N_i) per-frame matched | GT density effect |
| B | 8192 continuous | Grid-quantized GT | Density + quantization |
| C | Grid-quantized pred | Grid-quantized GT | Full legacy protocol |

### Grid quantization round-trip

Bin XY points into the baseline's 256x512 Cartesian grid, then extract:
1. `x_grid = linspace(0, 10.8, 256)`, `y_grid = linspace(-10.8, 10.8, 512)` — same as eval/eval_pointcloud.py
2. For each (x, y) point: `row = searchsorted(x_grid, x)`, `col = searchsorted(y_grid, y)`
3. Set `grid[row, col] = 1` (binary occupancy)
4. Extract non-zero cells back to (x_meters, y_meters) via the same grids

This replicates the baseline's Cartesian grid quantization (4.2cm spacing). No polar intermediary needed since v2 outputs Cartesian XY.

### Condition A: per-frame matched FPS

For each frame, determine N_i = number of points the grid quantization would produce, then FPS the v2 GT to N_i points. This isolates density from quantization.

### Metrics

Per-sample: Chamfer, mod-H, nn_pred2gt (median), nn_gt2pred (median), N_pred, N_gt.
Aggregate: median, mean, std over all test samples. Report N_i/M_i distributions.

### Implementation

Extend `v2/eval/standardize_eval.py` with:
- `--v2-model` flag to load v2 temporal model instead of baseline
- `--v2-dataset` flag to use v2 processed data
- `grid_quantize(xy_points)` function for the round-trip
- Four condition evaluation loop

### Acceptance criteria

- Control reproduces 0.295/0.429 (within float32 tolerance)
- If condition C gives mod-H < 0.25: eval pipeline is a major contributor
- If condition C gives mod-H > 0.35: v2 model genuinely has worse coverage
- Report directed terms to identify precision vs coverage contributions

### Codex review notes

- Grid quantization must match legacy geometry exactly (RMAX, RBINS, ABINS, bin edges)
- Use per-frame N_i matching, not fixed count
- Report frame-valid counts and N_i/M_i distributions
- Claims should be "within v2, most of the score gap appears to come from eval representation mismatch" — not full baseline parity (that needs cross-model same-protocol comparison)
