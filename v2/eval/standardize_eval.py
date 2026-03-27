# v2/eval/standardize_eval.py
"""Standardize eval: test cardinality impact on modified Hausdorff.

Runs baseline UNet1 inference on the full test set, converts polar output
to point clouds in-memory (replicating exact uint8 quantization), and
evaluates 4 conditions x multiple cardinalities.

Conditions:
  C1: variable pred vs variable GT       (control -- reproduces baseline)
  C2: FPS(N) pred vs variable GT          (isolate pred-side cardinality)
  C3: variable pred vs FPS(N) GT          (isolate GT-side cardinality)
  C4: FPS(N) pred vs FPS(N) GT            (both sides fixed)

Run inside Docker:
  docker compose run --rm mmdar python3 v2/eval/standardize_eval.py

Uses existing eval/eval_pointcloud.py for polar->point cloud conversion
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
PILOT_N = 2000
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
# Core: polar image -> 2D point cloud (in-memory, replicating uint8 path)
# ---------------------------------------------------------------------------

def polar_to_pc(polar_float: np.ndarray) -> np.ndarray:
    """Convert model output (float [0,1], shape 256x512) to 2D point cloud.

    Replicates exact baseline quantization: clip -> uint8 -> threshold -> legacy_cartesian.

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
    """Evaluate all conditions for one sample."""
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
                       RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=512,
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

        # In-memory polar -> point cloud (with uint8 quantization)
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
          f"{'nn_p->g':>10} {'nn_g->p':>10} {'N_pred':>10} {'N_gt':>10}")
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
