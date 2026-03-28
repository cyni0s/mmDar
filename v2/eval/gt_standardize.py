"""GT Standardization: test whether mod-H gap is eval pipeline mismatch.

Evaluates v2 temporal model predictions under 4 GT conditions:
  Control: v2 pred (8192 cont) vs v2 GT (8192 cont)    — reproduce 0.295/0.429
  A:       v2 pred (8192 cont) vs FPS(N_i) of v2 GT    — GT density effect
  B:       v2 pred (8192 cont) vs grid-quantized GT     — density + quantization
  C:       grid-quantized pred vs grid-quantized GT      — full legacy protocol

Run inside Docker:
  docker compose run --rm mmdar python3 v2/eval/gt_standardize.py
"""

import sys
import os
import time
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.eval_pointcloud import (
    _x_axis_grid, _y_axis_grid, RMAX, RBINS, ABINS,
)
from v2.model.temporal import TemporalMagPhaseFusion
from v2.data.windowed_dataset import build_windowed_dataloaders
from v2.eval.fps import fps_2d

CHECKPOINT = 'logs/v2_temporal_xattn/best.pt'
WINDOW_SIZE = 8
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=CHECKPOINT)
    p.add_argument('--pilot', type=int, default=0,
                   help='Pilot subset size (0 = full test set)')
    p.add_argument('--output', default='results/gt_standardize/')
    p.add_argument('--control-only', action='store_true',
                   help='Only run Control condition to verify parity')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Grid quantization: bin XY points into baseline's 256×512 Cartesian grid
# ---------------------------------------------------------------------------

def grid_quantize(xy_points: np.ndarray) -> np.ndarray:
    """Bin continuous XY points onto the baseline's Cartesian eval grid.

    Uses the same grid constants and searchsorted convention as
    eval/eval_pointcloud.py:polar_to_cartesian_legacy.

    Args:
        xy_points: (N, 2) or (N, 3) float array. Only columns 0,1 (x,y) used.

    Returns:
        (M, 2) float64 array of grid-quantized (x_meters, y_meters).
        M = number of unique occupied cells. May be 0 if all points are
        out of range.
    """
    if len(xy_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    x = xy_points[:, 0].astype(np.float64)
    y = xy_points[:, 1].astype(np.float64)

    # Filter to grid range
    mask = (x >= 0) & (x <= RMAX) & (y >= -RMAX) & (y <= RMAX)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Bin to grid indices (same as baseline's searchsorted + clip)
    row_idx = np.searchsorted(_x_axis_grid, x, side='left')
    col_idx = np.searchsorted(_y_axis_grid, y, side='left')
    row_idx = np.clip(row_idx, 0, RBINS - 1)
    col_idx = np.clip(col_idx, 0, ABINS - 1)

    # Deduplicate: unique occupied cells
    grid = np.zeros((RBINS, ABINS), dtype=np.uint8)
    grid[row_idx, col_idx] = 1

    # Extract non-zero cells back to metric coordinates
    occupied = np.argwhere(grid > 0)  # (M, 2) with [row, col]
    if len(occupied) == 0:
        return np.empty((0, 2), dtype=np.float64)

    x_meters = _x_axis_grid[occupied[:, 0]]
    y_meters = _y_axis_grid[occupied[:, 1]]
    return np.column_stack((x_meters, y_meters)).astype(np.float64)


# ---------------------------------------------------------------------------
# Metrics (GPU, same as standardize_eval.py)
# ---------------------------------------------------------------------------

def _nn_dists_gpu(src: torch.Tensor, tgt: torch.Tensor,
                  chunk: int = 2048) -> torch.Tensor:
    nn_list = []
    for s in range(0, src.shape[0], chunk):
        e = min(s + chunk, src.shape[0])
        d = torch.cdist(src[s:e], tgt)
        nn_list.append(d.min(dim=1).values)
    return torch.cat(nn_list)


def compute_metrics(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> dict:
    nn_pg = _nn_dists_gpu(pred_xy, gt_xy)
    nn_gp = _nn_dists_gpu(gt_xy, pred_xy)
    chamfer = float(0.5 * nn_pg.mean() + 0.5 * nn_gp.mean())
    med_pg = float(nn_pg.median())
    med_gp = float(nn_gp.median())
    return {
        'chamfer': chamfer,
        'mod_hausdorff': max(med_pg, med_gp),
        'nn_pred2gt_median': med_pg,
        'nn_gt2pred_median': med_gp,
    }


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(pred_pts: np.ndarray, gt_pts: np.ndarray,
                    device: torch.device, control_only: bool) -> dict:
    """Evaluate all conditions for one sample.

    Args:
        pred_pts: (8192, 3) predicted point cloud
        gt_pts:   (8192, 3) ground truth point cloud
        device:   torch device

    Returns dict of condition → metrics.
    """
    results = {}

    # Continuous XY tensors
    pred_xy = torch.from_numpy(pred_pts[:, :2].copy()).float().to(device)
    gt_xy = torch.from_numpy(gt_pts[:, :2].copy()).float().to(device)

    # Control: continuous vs continuous
    results['Control'] = compute_metrics(pred_xy, gt_xy)
    results['Control']['n_pred'] = int(pred_xy.shape[0])
    results['Control']['n_gt'] = int(gt_xy.shape[0])

    if control_only:
        return results

    # Grid-quantize both pred and GT
    pred_grid_np = grid_quantize(pred_pts)  # (M_i, 2)
    gt_grid_np = grid_quantize(gt_pts)      # (N_i, 2)

    n_pred_grid = len(pred_grid_np)
    n_gt_grid = len(gt_grid_np)

    # Condition B: continuous pred vs grid-quantized GT
    if n_gt_grid > 0:
        gt_grid = torch.from_numpy(gt_grid_np).float().to(device)
        results['B'] = compute_metrics(pred_xy, gt_grid)
        results['B']['n_pred'] = int(pred_xy.shape[0])
        results['B']['n_gt'] = n_gt_grid

    # Condition A: continuous pred vs FPS(N_i) of continuous GT
    # N_i matched to grid-quantized GT count for this frame
    if n_gt_grid > 0:
        gt_fps = fps_2d(gt_xy, n_gt_grid, seed=SEED)
        results['A'] = compute_metrics(pred_xy, gt_fps)
        results['A']['n_pred'] = int(pred_xy.shape[0])
        results['A']['n_gt'] = int(gt_fps.shape[0])

    # Condition C: grid-quantized pred vs grid-quantized GT
    if n_pred_grid > 0 and n_gt_grid > 0:
        pred_grid = torch.from_numpy(pred_grid_np).float().to(device)
        results['C'] = compute_metrics(pred_grid, gt_grid)
        results['C']['n_pred'] = n_pred_grid
        results['C']['n_gt'] = n_gt_grid

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.manual_seed(SEED)

    # Load v2 temporal model
    model = TemporalMagPhaseFusion(N_az=256, bridge_out_ch=128, max_lag=16)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f'Loaded: {args.checkpoint}')

    # Build test dataloader
    loaders = build_windowed_dataloaders(
        processed_dir='v2/data/processed',
        window_size=WINDOW_SIZE,
        batch_size=1,
        num_workers=0,
    )
    test_loader = loaders['test']
    n_total = len(test_loader)
    n_eval = min(args.pilot, n_total) if args.pilot > 0 else n_total
    print(f'Test set: {n_total} samples, evaluating {n_eval}')

    all_results = []
    t0 = time.time()

    for idx, (radar, lidar, norm) in enumerate(test_loader):
        if idx >= n_eval:
            break

        with torch.no_grad():
            pred_pts, _conf = model(radar.to(device))

        pred_np = pred_pts[0].cpu().numpy()  # (8192, 3)
        gt_np = lidar[0].cpu().numpy()       # (8192, 3)

        sample = evaluate_sample(pred_np, gt_np, device, args.control_only)
        all_results.append(sample)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_eval - idx - 1) / rate
            print(f'  [{idx+1}/{n_eval}] {rate:.1f} samples/s, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'Done: {len(all_results)} samples in {elapsed:.1f}s')

    report = aggregate(all_results, args.control_only)
    report['meta'] = {
        'checkpoint': args.checkpoint,
        'window_size': WINDOW_SIZE,
        'n_samples': len(all_results),
        'elapsed_s': round(elapsed, 1),
        'seed': SEED,
    }

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved: {out_path}')
    print_summary(report)
    return report


def aggregate(all_results: list, control_only: bool) -> dict:
    conditions = ['Control']
    if not control_only:
        conditions += ['A', 'B', 'C']

    report = {}
    for cond in conditions:
        vals = {'chamfer': [], 'mod_hausdorff': [],
                'nn_pred2gt_median': [], 'nn_gt2pred_median': [],
                'n_pred': [], 'n_gt': []}
        n_empty = 0

        for sample in all_results:
            s = sample.get(cond)
            if s is None:
                n_empty += 1
                continue
            for k in ['chamfer', 'mod_hausdorff', 'nn_pred2gt_median', 'nn_gt2pred_median']:
                vals[k].append(s[k])
            vals['n_pred'].append(s['n_pred'])
            vals['n_gt'].append(s['n_gt'])

        if not vals['chamfer']:
            report[cond] = {'n_valid': 0, 'n_empty': n_empty}
            continue

        report[cond] = {
            'chamfer_median': float(np.median(vals['chamfer'])),
            'chamfer_mean': float(np.mean(vals['chamfer'])),
            'mod_h_median': float(np.median(vals['mod_hausdorff'])),
            'mod_h_mean': float(np.mean(vals['mod_hausdorff'])),
            'nn_pred2gt_median': float(np.median(vals['nn_pred2gt_median'])),
            'nn_gt2pred_median': float(np.median(vals['nn_gt2pred_median'])),
            'n_pred_median': float(np.median(vals['n_pred'])),
            'n_gt_median': float(np.median(vals['n_gt'])),
            'n_gt_p5': float(np.percentile(vals['n_gt'], 5)),
            'n_gt_p95': float(np.percentile(vals['n_gt'], 95)),
            'n_valid': len(vals['chamfer']),
            'n_empty': n_empty,
        }
    return report


def print_summary(report: dict):
    print('\n' + '=' * 80)
    print('GT STANDARDIZATION RESULTS')
    print('=' * 80)
    print(f"{'Condition':<12} {'Chamfer':>10} {'Mod-H':>10} "
          f"{'nn_p->g':>10} {'nn_g->p':>10} {'N_pred':>8} {'N_gt':>8}")
    print('-' * 80)

    for cond in ['Control', 'A', 'B', 'C']:
        v = report.get(cond, {})
        if 'chamfer_median' not in v:
            continue
        print(f"{cond:<12} {v['chamfer_median']:>10.4f} {v['mod_h_median']:>10.4f} "
              f"{v['nn_pred2gt_median']:>10.4f} {v['nn_gt2pred_median']:>10.4f} "
              f"{v['n_pred_median']:>8.0f} {v['n_gt_median']:>8.0f}")

    ctrl = report.get('Control', {})
    if 'chamfer_median' in ctrl:
        cd = ctrl['chamfer_median']
        mh = ctrl['mod_h_median']
        print(f"\nControl parity: Chamfer {cd:.4f} (expect ~0.295) | "
              f"mod-H {mh:.4f} (expect ~0.429)")

    c = report.get('C', {})
    if 'mod_h_median' in c and 'mod_h_median' in ctrl:
        gap_reduction = 1.0 - (c['mod_h_median'] / ctrl['mod_h_median'])
        print(f"Legacy protocol (C): mod-H {c['mod_h_median']:.4f} "
              f"({gap_reduction*100:.1f}% reduction from Control)")


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
