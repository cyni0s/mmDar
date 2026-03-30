# Standardize Eval: Cardinality Impact on Modified Hausdorff

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the mod-H gap (0.42 vs 0.19) between v2 and baseline is caused by fixed 8192-point cardinality vs variable cardinality, by running the baseline's own output through FPS subsampling and measuring metric degradation.

**Architecture:** Single standalone script (`v2/eval/standardize_eval.py`) that loads the baseline checkpoint, runs inference on all 18,575 test samples, converts polar output to point clouds in-memory (replicating exact uint8 quantization), and evaluates 4 conditions × 7 cardinalities. GPU-accelerated metrics via torch.cdist.

**Tech Stack:** PyTorch, numpy, cv2 (thresholding), existing `eval/eval_pointcloud.py` (coordinate grids + `polar_image_to_pointcloud`), existing `train_test_utils/` (model + dataloader)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `v2/eval/standardize_eval.py` | Main experiment script — inference, FPS, metrics, JSON report |
| `v2/eval/fps.py` | Pure-PyTorch farthest point sampling (2D, deterministic) |
| `v2/eval/tests/test_fps.py` | Tests for FPS correctness |
| `v2/eval/tests/test_standardize_eval.py` | Integration tests: C1 parity, metric consistency |

---

### Task 1: Implement FPS in PyTorch

**Files:**
- Create: `v2/eval/fps.py`
- Create: `v2/eval/tests/test_fps.py`

- [ ] **Step 1: Write failing tests for FPS**

```python
# v2/eval/tests/test_fps.py
import torch
import numpy as np
import pytest


def test_fps_returns_correct_count():
    """FPS on 100 points requesting 10 should return exactly 10."""
    from v2.eval.fps import fps_2d
    pts = torch.randn(100, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (10, 2)


def test_fps_fewer_than_n_returns_all():
    """When K < N, fps_2d returns all K points (no padding)."""
    from v2.eval.fps import fps_2d
    pts = torch.randn(5, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (5, 2)


def test_fps_deterministic():
    """Same seed produces same output."""
    from v2.eval.fps import fps_2d
    pts = torch.randn(100, 2)
    r1 = fps_2d(pts, 20, seed=0)
    r2 = fps_2d(pts, 20, seed=0)
    assert torch.allclose(r1, r2)


def test_fps_spread():
    """FPS should spread points — min pairwise distance should be larger than random."""
    from v2.eval.fps import fps_2d
    # Grid of points: FPS should pick well-spread subset
    xs = torch.linspace(0, 1, 20)
    ys = torch.linspace(0, 1, 20)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)  # 400 pts
    result = fps_2d(grid, 20, seed=0)
    dists = torch.cdist(result, result)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min().item()
    # On a [0,1]² grid, 20 FPS points should have min spacing > 0.15
    assert min_dist > 0.15, f"FPS min spacing {min_dist} too small"


def test_fps_empty_returns_empty():
    """Empty input returns empty output."""
    from v2.eval.fps import fps_2d
    pts = torch.zeros(0, 2)
    result = fps_2d(pts, 10, seed=0)
    assert result.shape == (0, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /git/mmDar && python -m pytest v2/eval/tests/test_fps.py -v`
Expected: ImportError — `fps_2d` not found

- [ ] **Step 3: Implement FPS**

```python
# v2/eval/fps.py
"""Farthest Point Sampling on 2D point clouds (pure PyTorch, no Open3D)."""

import torch


def fps_2d(points: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point sampling on 2D points.

    Args:
        points: (K, 2) tensor of 2D points.
        n: target number of points.
        seed: deterministic start index = seed % K.

    Returns:
        (min(K, n), 2) tensor of selected points.
        Returns (0, 2) if input is empty.
    """
    K = points.shape[0]
    if K == 0:
        return points[:0]  # preserve (0, 2) shape
    if K <= n:
        return points

    selected = [seed % K]
    dists = torch.full((K,), float('inf'), device=points.device)

    for _ in range(n - 1):
        new_dists = torch.cdist(
            points, points[selected[-1]].unsqueeze(0)
        ).squeeze(1)
        dists = torch.minimum(dists, new_dists)
        selected.append(int(dists.argmax()))

    return points[selected]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /git/mmDar && python -m pytest v2/eval/tests/test_fps.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add v2/eval/fps.py v2/eval/tests/test_fps.py
git commit -m "feat(v2/eval): add pure-PyTorch 2D farthest point sampling"
```

---

### Task 2: Implement the standardize_eval script

**Files:**
- Create: `v2/eval/standardize_eval.py`
- Reference: `eval/eval_pointcloud.py` (polar_image_to_pointcloud, coordinate grids)
- Reference: `train_test_utils/model.py` (UNet1)
- Reference: `train_test_utils/dataloader.py` (Dataset)

- [ ] **Step 1: Write the script skeleton with argument parsing and config**

```python
# v2/eval/standardize_eval.py
"""Standardize eval: test cardinality impact on modified Hausdorff.

Runs baseline UNet1 inference on the full test set, converts polar output
to point clouds in-memory (replicating exact uint8 quantization), and
evaluates 4 conditions x multiple cardinalities.

Conditions:
  C1: variable pred vs variable GT       (control — reproduces baseline)
  C2: FPS(N) pred vs variable GT          (isolate pred-side cardinality)
  C3: variable pred vs FPS(N) GT          (isolate GT-side cardinality)
  C4: FPS(N) pred vs FPS(N) GT            (both sides fixed)

Run inside Docker:
  docker compose run --rm mmdar python3 v2/eval/standardize_eval.py

Uses existing eval/eval_pointcloud.py for polar→point cloud conversion
and v2/eval/fps.py for deterministic farthest point sampling.
"""

import sys
import os
import time
import json
import argparse

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.eval_pointcloud import polar_image_to_pointcloud, COORD_MODE_LEGACY
from train_test_utils.dataloader import Dataset
from train_test_utils.model import UNet1
from v2.eval.fps import fps_2d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT = 'logs/sweep_b12_lr7e-05_fp32_20260319-140647/010.pt_gen'
HISTORY = 40
CARDINALITIES = [256, 512, 1024, 2048, 4096, 8192, 16384]
PILOT_N = 2000  # quick pilot before full sweep
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=CHECKPOINT,
                   help='Path to baseline .pt_gen checkpoint')
    p.add_argument('--pilot', type=int, default=PILOT_N,
                   help='Pilot subset size (0 = skip pilot, run full)')
    p.add_argument('--full', action='store_true',
                   help='Run full test set (skip pilot)')
    p.add_argument('--cardinalities', type=int, nargs='+', default=CARDINALITIES,
                   help='Point counts to sweep')
    p.add_argument('--output', default='results/standardize_eval/',
                   help='Output directory for JSON report')
    p.add_argument('--c1-only', action='store_true',
                   help='Run only C1 (control) to verify parity')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics (GPU-accelerated, matching eval/eval_pointcloud.py definitions)
# ---------------------------------------------------------------------------

def _nn_dists_gpu(src: torch.Tensor, tgt: torch.Tensor,
                  chunk: int = 2048) -> torch.Tensor:
    """Chunked nearest-neighbor distances: src -> tgt, returns (N,) tensor."""
    nn_list = []
    for s in range(0, src.shape[0], chunk):
        e = min(s + chunk, src.shape[0])
        d = torch.cdist(src[s:e], tgt)  # (chunk, M)
        nn_list.append(d.min(dim=1).values)
    return torch.cat(nn_list)


def compute_metrics(pred_xy: torch.Tensor,
                    gt_xy: torch.Tensor) -> dict:
    """Chamfer + directed mod-H terms on 2D XY point clouds.

    Returns dict with:
      chamfer, mod_hausdorff,
      nn_pred2gt_median, nn_gt2pred_median  (directed terms)
    """
    nn_pg = _nn_dists_gpu(pred_xy, gt_xy)
    nn_gp = _nn_dists_gpu(gt_xy, pred_xy)

    chamfer = float(0.5 * nn_pg.mean() + 0.5 * nn_gp.mean())
    med_pg = float(nn_pg.median())
    med_gp = float(nn_gp.median())
    mod_h = max(med_pg, med_gp)

    return {
        'chamfer': chamfer,
        'mod_hausdorff': mod_h,
        'nn_pred2gt_median': med_pg,
        'nn_gt2pred_median': med_gp,
    }


# ---------------------------------------------------------------------------
# Core: polar image → 2D point cloud (in-memory, replicating uint8 path)
# ---------------------------------------------------------------------------

def polar_to_pc(polar_float: np.ndarray) -> np.ndarray:
    """Convert model output (float [0,1], shape 256x512) to 2D point cloud.

    Replicates exact baseline quantization: clip → uint8 → threshold → legacy_cartesian.

    Returns (N, 2) float64 array (x_meters, y_meters). N is variable.
    """
    u8 = np.clip(polar_float * 255, 0, 255).astype(np.uint8)
    return polar_image_to_pointcloud(u8, threshold=1,
                                     coordinate_mode=COORD_MODE_LEGACY)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_conditions(pred_pc_np: np.ndarray, gt_pc_np: np.ndarray,
                        cardinalities: list, device: torch.device,
                        c1_only: bool = False) -> dict:
    """Evaluate all conditions for one sample.

    Args:
        pred_pc_np: (N_pred, 2) variable-size pred point cloud
        gt_pc_np:   (N_gt, 2) variable-size GT point cloud
        cardinalities: list of target N values for FPS
        device: torch device for GPU metrics
        c1_only: if True, only compute C1

    Returns:
        dict of condition -> metrics
    """
    results = {}

    pred_var = torch.from_numpy(pred_pc_np).float().to(device)
    gt_var = torch.from_numpy(gt_pc_np).float().to(device)

    # C1: variable vs variable (control)
    if pred_var.shape[0] > 0 and gt_var.shape[0] > 0:
        results['C1'] = compute_metrics(pred_var, gt_var)
        results['C1']['n_pred'] = int(pred_var.shape[0])
        results['C1']['n_gt'] = int(gt_var.shape[0])
    else:
        results['C1'] = None

    if c1_only:
        return results

    for N in cardinalities:
        pred_fps = fps_2d(pred_var, N, seed=SEED)
        gt_fps = fps_2d(gt_var, N, seed=SEED)

        key_suffix = f'_N{N}'

        # C2: FPS(N) pred vs variable GT
        if pred_fps.shape[0] > 0 and gt_var.shape[0] > 0:
            results[f'C2{key_suffix}'] = compute_metrics(pred_fps, gt_var)
            results[f'C2{key_suffix}']['n_pred'] = int(pred_fps.shape[0])
            results[f'C2{key_suffix}']['n_gt'] = int(gt_var.shape[0])
            results[f'C2{key_suffix}']['saturated_pred'] = int(pred_var.shape[0] < N)

        # C3: variable pred vs FPS(N) GT
        if pred_var.shape[0] > 0 and gt_fps.shape[0] > 0:
            results[f'C3{key_suffix}'] = compute_metrics(pred_var, gt_fps)
            results[f'C3{key_suffix}']['n_pred'] = int(pred_var.shape[0])
            results[f'C3{key_suffix}']['n_gt'] = int(gt_fps.shape[0])
            results[f'C3{key_suffix}']['saturated_gt'] = int(gt_var.shape[0] < N)

        # C4: FPS(N) pred vs FPS(N) GT
        if pred_fps.shape[0] > 0 and gt_fps.shape[0] > 0:
            results[f'C4{key_suffix}'] = compute_metrics(pred_fps, gt_fps)
            results[f'C4{key_suffix}']['n_pred'] = int(pred_fps.shape[0])
            results[f'C4{key_suffix}']['n_gt'] = int(gt_fps.shape[0])

    return results


def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(SEED)

    # Load model
    gen = UNet1(HISTORY + 1, 1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    gen.load_state_dict(ckpt['state_dict'])
    gen.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Load test data
    test_set = Dataset('dataset_5/', 'test',
                       RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
                       RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=1024,
                       M=HISTORY)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                               shuffle=False, num_workers=0)
    n_total = len(test_loader)
    print(f'Test set: {n_total} samples')

    # Determine sample count
    if args.full or args.pilot <= 0:
        n_eval = n_total
    else:
        n_eval = min(args.pilot, n_total)
    print(f'Evaluating {n_eval} samples ({"full" if n_eval == n_total else "pilot"})')

    # Collect per-sample results
    all_results = []
    t0 = time.time()

    for idx, (radar, label) in enumerate(test_loader):
        if idx >= n_eval:
            break

        with torch.no_grad():
            pred = gen(radar.to(device))  # (1, 1, 256, 512)

        # Convert to polar images
        pred_polar = pred.squeeze().cpu().numpy()         # float [0,1], (256, 512)
        label_polar = label.squeeze().cpu().numpy()       # float {0,1}, (256, 512)

        # In-memory polar → point cloud (with uint8 quantization)
        pred_pc = polar_to_pc(pred_polar)   # (N_pred, 2)
        gt_pc = polar_to_pc(label_polar)    # (N_gt, 2)

        # Evaluate all conditions
        sample_results = evaluate_conditions(
            pred_pc, gt_pc, args.cardinalities, device, c1_only=args.c1_only
        )
        all_results.append(sample_results)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_eval - idx - 1) / rate
            print(f'  [{idx+1}/{n_eval}] {rate:.1f} samples/s, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'Done: {n_eval} samples in {elapsed:.1f}s ({n_eval/elapsed:.1f} samples/s)')

    # Aggregate results
    report = aggregate_results(all_results, args.cardinalities, args.c1_only)
    report['meta'] = {
        'checkpoint': args.checkpoint,
        'n_samples': n_eval,
        'cardinalities': args.cardinalities,
        'seed': SEED,
        'elapsed_s': round(elapsed, 1),
        'full_test_set': n_eval == n_total,
    }

    # Save
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to {out_path}')

    # Print summary
    print_summary(report)

    return report


def aggregate_results(all_results: list, cardinalities: list,
                      c1_only: bool) -> dict:
    """Aggregate per-sample metrics into dataset-level statistics."""
    report = {}

    # Collect all condition keys
    conditions = ['C1']
    if not c1_only:
        for N in cardinalities:
            conditions.extend([f'C2_N{N}', f'C3_N{N}', f'C4_N{N}'])

    for cond in conditions:
        values = {
            'chamfer': [], 'mod_hausdorff': [],
            'nn_pred2gt_median': [], 'nn_gt2pred_median': [],
            'n_pred': [], 'n_gt': [],
        }
        n_empty = 0

        for sample in all_results:
            if sample.get(cond) is None:
                n_empty += 1
                continue
            for k in ['chamfer', 'mod_hausdorff', 'nn_pred2gt_median',
                       'nn_gt2pred_median']:
                values[k].append(sample[cond][k])
            values['n_pred'].append(sample[cond]['n_pred'])
            values['n_gt'].append(sample[cond]['n_gt'])

        if not values['chamfer']:
            report[cond] = {'n_valid': 0, 'n_empty': n_empty}
            continue

        report[cond] = {
            'chamfer_median': float(np.median(values['chamfer'])),
            'chamfer_mean': float(np.mean(values['chamfer'])),
            'chamfer_std': float(np.std(values['chamfer'])),
            'mod_h_median': float(np.median(values['mod_hausdorff'])),
            'mod_h_mean': float(np.mean(values['mod_hausdorff'])),
            'mod_h_std': float(np.std(values['mod_hausdorff'])),
            'nn_pred2gt_median': float(np.median(values['nn_pred2gt_median'])),
            'nn_gt2pred_median': float(np.median(values['nn_gt2pred_median'])),
            'n_pred_median': float(np.median(values['n_pred'])),
            'n_gt_median': float(np.median(values['n_gt'])),
            'n_pred_p5': float(np.percentile(values['n_pred'], 5)),
            'n_pred_p95': float(np.percentile(values['n_pred'], 95)),
            'n_valid': len(values['chamfer']),
            'n_empty': n_empty,
        }

        # Saturation rate for C2/C3/C4
        for sat_key in ['saturated_pred', 'saturated_gt']:
            sat_vals = [s[cond].get(sat_key) for s in all_results
                       if s.get(cond) is not None and sat_key in s.get(cond, {})]
            if sat_vals:
                report[cond][f'{sat_key}_rate'] = float(np.mean(sat_vals))

    return report


def print_summary(report: dict):
    """Print a formatted summary table."""
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY')
    print('=' * 80)
    print(f"{'Condition':<16} {'Chamfer':>10} {'Mod-H':>10} "
          f"{'nn_p→g':>10} {'nn_g→p':>10} {'N_pred':>10} {'N_gt':>10}")
    print('-' * 80)

    for cond, vals in sorted(report.items()):
        if cond == 'meta' or not isinstance(vals, dict):
            continue
        if 'chamfer_median' not in vals:
            continue
        print(f"{cond:<16} {vals['chamfer_median']:>10.4f} {vals['mod_h_median']:>10.4f} "
              f"{vals['nn_pred2gt_median']:>10.4f} {vals['nn_gt2pred_median']:>10.4f} "
              f"{vals['n_pred_median']:>10.0f} {vals['n_gt_median']:>10.0f}")

    # C1 parity check
    c1 = report.get('C1', {})
    if 'chamfer_median' in c1:
        cd = c1['chamfer_median']
        mh = c1['mod_h_median']
        cd_ok = abs(cd - 0.295) < 0.005
        mh_ok = abs(mh - 0.189) < 0.005
        print(f"\nC1 parity: Chamfer {cd:.4f} ({'PASS' if cd_ok else 'FAIL'}) | "
              f"mod-H {mh:.4f} ({'PASS' if mh_ok else 'FAIL'})")
        if not cd_ok or not mh_ok:
            print("*** STOP: C1 does not reproduce baseline. Fix parity before sweep. ***")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
```

- [ ] **Step 2: Verify script syntax**

Run: `cd /git/mmDar && python -c "import ast; ast.parse(open('v2/eval/standardize_eval.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add v2/eval/standardize_eval.py
git commit -m "feat(v2/eval): add standardize_eval experiment script"
```

---

### Task 3: Run C1-only parity check (stop gate)

This is the most important step. If C1 doesn't reproduce 0.295/0.189, stop and debug.

**Files:**
- Reference: `v2/eval/standardize_eval.py`

- [ ] **Step 1: Run C1-only on pilot (2000 samples) inside Docker**

```bash
docker compose run --rm mmdar python3 v2/eval/standardize_eval.py \
    --c1-only --pilot 2000 --output results/standardize_eval_pilot/
```

Expected: ~5 min (inference + metric computation). Output shows:
- Chamfer median ≈ 0.295 ± 0.01 (pilot may differ slightly from full)
- mod-H median ≈ 0.189 ± 0.01

- [ ] **Step 2: If parity fails, debug**

Common issues:
- Quantization mismatch: check `np.clip` before `astype(np.uint8)`
- Threshold: baseline uses `cv2.THRESH_TOZERO` with threshold=1, not 0
- Coordinate mode: must be `legacy_cartesian`, not `polar_direct`
- Label format: Dataset returns float {0.0, 1.0} not uint8 {0, 255}. `polar_to_pc` handles this via `* 255 → uint8`

- [ ] **Step 3: If parity passes, run C1-only on full test set**

```bash
docker compose run --rm mmdar python3 v2/eval/standardize_eval.py \
    --c1-only --full --output results/standardize_eval_c1/
```

Expected: ~15 min (data load) + ~20 min (18,575 samples inference + metrics)
Acceptance: Chamfer 0.295 ± 0.005, mod-H 0.189 ± 0.005

- [ ] **Step 4: Commit results**

```bash
git add results/standardize_eval_c1/report.json
git commit -m "results: C1 parity verified — baseline reproduces 0.295/0.189 in-memory"
```

---

### Task 4: Run pilot sweep (2000 samples, all conditions)

- [ ] **Step 1: Run pilot with stratified subset**

```bash
docker compose run --rm mmdar python3 v2/eval/standardize_eval.py \
    --pilot 2000 --output results/standardize_eval_pilot/
```

Expected: ~30-60 min (inference once, then 4 conditions × 7 cardinalities × 2000 samples for metrics). Watch for:
- C1 matches full parity
- N-sweep trend: does mod-H degrade gradually or sharply?
- Saturation rate: how many samples have K < N (especially at N=8192, 16384)?

- [ ] **Step 2: Inspect pilot results**

```bash
cat results/standardize_eval_pilot/report.json | python3 -m json.tool
```

Look for:
- C2 (pred FPS, GT variable): if mod-H jumps → pred cardinality matters
- C3 (pred variable, GT FPS): if mod-H jumps → GT cardinality matters
- C4 (both FPS): combined effect
- Directed terms: which direction (pred→gt or gt→pred) drives degradation?
- Saturation rate: if >50% of samples have K < N, padding dominates the test

- [ ] **Step 3: Commit pilot results**

```bash
git add results/standardize_eval_pilot/report.json
git commit -m "results: pilot sweep — cardinality impact on mod-H (2000 samples)"
```

---

### Task 5: Run full sweep (if pilot is informative)

Only run if pilot results show a clear trend worth confirming on the full set.

- [ ] **Step 1: Select interesting cardinality range from pilot**

If the pilot shows a clear knee (e.g., mod-H flat for N > 4096), narrow the sweep:
```bash
docker compose run --rm mmdar python3 v2/eval/standardize_eval.py \
    --full --cardinalities 1024 2048 4096 8192 16384 \
    --output results/standardize_eval_full/
```

Expected: ~2-3 hours (inference + full metrics sweep)

- [ ] **Step 2: Analyze and report**

Compare full results against pilot. Write findings to `results/README.md` under a new section.

- [ ] **Step 3: Commit**

```bash
git add results/standardize_eval_full/report.json results/README.md
git commit -m "results: full cardinality sweep — mod-H vs point count analysis"
```

---

### Task 6: Interpret results and decide next steps

No code — analysis and decision-making.

- [ ] **Step 1: Check hypothesis A (cardinality is the bottleneck)**

If baseline mod-H jumps from 0.189 to ~0.4 at N=8192:
→ Confirmed. The 8192-point representation cannot achieve 0.189 mod-H.
→ Next: variable-cardinality decoder (polar occupancy head with proper capacity)

- [ ] **Step 2: Check hypothesis B (cardinality is NOT the bottleneck)**

If baseline mod-H stays ~0.19 even at N=8192:
→ The v2 decoder is the problem, not the fixed count.
→ Next: investigate why v2's learned 8192 points have worse coverage than FPS-subsampled baseline points. Likely the decoder's placement strategy, not the count.

- [ ] **Step 3: Check directed terms for insights**

If nn_gt→pred dominates (coverage): v2 isn't covering all GT regions.
If nn_pred→gt dominates (precision): v2 is placing points in wrong locations.

- [ ] **Step 4: Document findings in results/README.md**

Update the "Phase 6" section with the standardized eval findings and implications for the decoder design.
