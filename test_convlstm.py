"""Inference script for UNet1ConvLSTM (ConvLSTM bottleneck model).

Usage:
  # Standard inference at T=41:
  python test_convlstm.py --checkpoint logs/convlstm_1_20260320-000000/030.pt_gen \
      --output_dir /tmp/convlstm_test --T 41 --eval

  # T-curve evaluation (T = 1,4,8,16,32,41):
  python test_convlstm.py --checkpoint logs/convlstm_1_20260320-000000/030.pt_gen \
      --output_dir /tmp/convlstm_tcurve --t_curve --eval

Output format:
  {output_dir}/{traj_id}_{frame_idx}_pred.png
  {output_dir}/{traj_id}_{frame_idx}_label.png

These filenames are compatible with eval/eval_pointcloud.py evaluate_experiment().
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from train_test_utils.dataloader import Dataset, SequentialDataset
from train_test_utils.model import UNet1ConvLSTM
from eval.eval_pointcloud import evaluate_experiment

# T values for T-curve evaluation (zero-init state per T, per sample)
T_CURVE_VALUES = [1, 4, 8, 16, 32, 41]

BASEPATH = './dataset_5/'
ORIG_SIZE = [256, 64, 512]
REQD_SIZE = [256, 64, 512]


def load_model(checkpoint_path, device):
    """Load UNet1ConvLSTM from checkpoint."""
    model = UNet1ConvLSTM(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def run_inference(model, T, output_dir, device, ordered_filename=None, seq_dataset=None):
    """Run ConvLSTM inference at a given T value.

    Loads the test set using the baseline Dataset(M=40) to get the exact same 18,575
    targets and filenames. For each target, loads the last T frames of the 41-frame
    window from SequentialDataset, zero-initialises state, and runs a forward pass.

    CRITICAL: state is zero-initialised per sample (state=None). No state carryover
    between test samples. This is required for correct T-curve evaluation (Pitfall 5).

    Args:
        model: loaded UNet1ConvLSTM in eval mode
        T: number of history frames to use (1 <= T <= 41)
        output_dir: directory to save *_pred.png and *_label.png
        device: torch device
        ordered_filename: list from Dataset.__filenames__() — reused if provided
        seq_dataset: SequentialDataset instance — reused if provided

    Returns:
        ordered_filename list (for reuse across T values)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load baseline test dataset for target pool and ordered filenames
    # Dataset(M=40) has input_data[i] = list of 41 radar file paths per sample
    if ordered_filename is None or seq_dataset is None:
        print(f'  Loading test dataset (M=40) ...')
        baseline_set = Dataset(
            BASEPATH, 'test',
            RBINS=REQD_SIZE[0], ABINS_RADAR=REQD_SIZE[1], ABINS_LIDAR=REQD_SIZE[2],
            RBINS_ORIG=ORIG_SIZE[0], ABINS_RADAR_ORIG=ORIG_SIZE[1], ABINS_LIDAR_ORIG=ORIG_SIZE[2],
            M=40,
        )
        ordered_filename = baseline_set.__filenames__()

        # Also build SequentialDataset to access per-traj file lists
        seq_dataset = SequentialDataset(
            BASEPATH, 'test', M=40,
            RBINS=REQD_SIZE[0], ABINS_RADAR=REQD_SIZE[1], ABINS_LIDAR=REQD_SIZE[2],
            RBINS_ORIG=ORIG_SIZE[0], ABINS_RADAR_ORIG=ORIG_SIZE[1], ABINS_LIDAR_ORIG=ORIG_SIZE[2],
        )
        print(f'  Test set: {len(ordered_filename)} samples')

    n_samples = len(ordered_filename)
    print(f'  Running inference: T={T}, samples={n_samples}, output={output_dir}')

    t0 = time.time()
    with torch.no_grad():
        for sample_i, (traj_id, local_target_idx) in enumerate(seq_dataset.eligible_targets):
            # Load last T radar frames of the 41-frame window ending at local_target_idx
            # CRITICAL: zero-init state per sample (state=None)
            data = seq_dataset.traj_data[traj_id]

            # The window of interest ends at local_target_idx (inclusive).
            # We want the last T frames: start = local_target_idx - T + 1
            start = local_target_idx - T + 1
            end = local_target_idx + 1  # exclusive
            radar_paths = data['radar'][start:end]
            lidar_label_path = data['lidar'][local_target_idx]

            # Load radar frames individually, stack to (T, 1, H, W)
            radar_frames = [seq_dataset.get_radar(p) for p in radar_paths]
            radar_seq = torch.stack(radar_frames, dim=0)   # (T, 1, 256, 64)
            radar_seq = radar_seq.unsqueeze(0).to(device)  # (1, T, 1, 256, 64)

            # Forward: zero-init state per sample
            pred, _ = model(radar_seq, state=None)
            # pred: (1, T, 1, 256, 512) — take final timestep
            pred_final = pred[:, -1]  # (1, 1, 256, 512)

            # Convert to uint8 PNG
            pred_np = pred_final.squeeze().cpu().numpy()   # (256, 512)
            pred_np = (pred_np * 255).astype(np.uint8)

            # Load label
            label_tensor = seq_dataset.get_lidar(lidar_label_path)  # (1, 256, 512)
            label_np = label_tensor.squeeze().numpy()                # (256, 512)
            label_np = (label_np * 255).astype(np.uint8)

            # Save PNGs: stem = ordered_filename[sample_i] = "{traj_id}_{frame_idx}"
            stem = ordered_filename[sample_i]
            Image.fromarray(pred_np).save(os.path.join(output_dir, f'{stem}_pred.png'))
            Image.fromarray(label_np).save(os.path.join(output_dir, f'{stem}_label.png'))

            if (sample_i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f'    {sample_i+1}/{n_samples}  elapsed={elapsed:.0f}s')

    elapsed = time.time() - t0
    print(f'  Inference complete: {elapsed:.1f}s ({elapsed/n_samples*1000:.1f}ms/sample)')
    return ordered_filename, seq_dataset


def run_t_curve(model, output_dir, device, args):
    """Evaluate the model at each T in T_CURVE_VALUES.

    For each T:
      1. Zero-init state for every sample (Pitfall 5 prevention)
      2. Run inference
      3. Optionally evaluate metrics
      4. Collect results

    Returns list of dicts: {T, chamfer, mod_hausdorff} (if --eval) or {T} only.
    """
    ordered_filename = None
    seq_dataset = None
    curve_results = []

    for T in T_CURVE_VALUES:
        print(f'\n--- T-curve: T={T} ---')
        t_dir = os.path.join(output_dir, f'T{T:02d}')

        ordered_filename, seq_dataset = run_inference(
            model, T, t_dir, device,
            ordered_filename=ordered_filename,
            seq_dataset=seq_dataset,
        )

        row = {'T': T}
        if args.eval:
            results_subdir = os.path.join(
                args.results_dir or output_dir,
                f'{args.experiment_name}_T{T:02d}',
            )
            agg = evaluate_experiment(
                pred_dir=t_dir,
                label_dir=t_dir,
                output_dir=results_subdir,
                experiment_name=f'{args.experiment_name}_T{T:02d}',
                coordinate_mode='legacy_cartesian',
            )
            row['chamfer']      = agg['chamfer_distance']['median']
            row['mod_hausdorff'] = agg['modified_hausdorff']['median']
            print(f'  T={T}: Chamfer={row["chamfer"]:.4f}m  mod-H={row["mod_hausdorff"]:.4f}m')

        curve_results.append(row)

    return curve_results


def print_t_curve_table(curve_results):
    """Print a formatted T-curve results table."""
    print(f'\n{"="*55}')
    print(f'T-CURVE RESULTS')
    print(f'{"T":>4}  {"Chamfer (m)":>12}  {"mod-H (m)":>12}')
    for r in curve_results:
        chamfer_str = f'{r["chamfer"]:>12.4f}' if 'chamfer' in r else f'{"N/A":>12}'
        mh_str      = f'{r["mod_hausdorff"]:>12.4f}' if 'mod_hausdorff' in r else f'{"N/A":>12}'
        print(f'{r["T"]:>4}  {chamfer_str}  {mh_str}')
    print(f'{"="*55}')


def main():
    parser = argparse.ArgumentParser(
        description='ConvLSTM inference and T-curve evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt_gen checkpoint file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output PNGs (and T-curve subdirs).')
    parser.add_argument('--T', type=int, default=41,
                        help='Number of history frames to use (standard mode).')
    parser.add_argument('--t_curve', action='store_true',
                        help=f'Run T-curve evaluation at T={T_CURVE_VALUES}.')
    parser.add_argument('--eval', action='store_true',
                        help='Run eval_pointcloud after inference.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Results output directory for eval metrics (default: output_dir).')
    parser.add_argument('--experiment_name', type=str, default='convlstm',
                        help='Experiment name embedded in eval output.')
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')

    model = load_model(args.checkpoint, device)
    print(f'Model loaded: UNet1ConvLSTM')

    if args.t_curve:
        curve_results = run_t_curve(model, args.output_dir, device, args)

        # Print table
        print_t_curve_table(curve_results)

        # Save t_curve.json
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, 't_curve.json')
        with open(json_path, 'w') as f:
            json.dump({'experiment_name': args.experiment_name,
                       'T_curve_values': T_CURVE_VALUES,
                       'results': curve_results}, f, indent=2)
        print(f'\nSaved: {json_path}')

    else:
        # Standard single-T inference
        ordered_filename, _ = run_inference(model, args.T, args.output_dir, device)

        if args.eval:
            results_dir = args.results_dir or args.output_dir
            agg = evaluate_experiment(
                pred_dir=args.output_dir,
                label_dir=args.output_dir,
                output_dir=results_dir,
                experiment_name=args.experiment_name,
                coordinate_mode='legacy_cartesian',
            )
            print(f'\nMetrics (T={args.T}):')
            for key in ['chamfer_distance', 'modified_hausdorff', 'iou', 'f1']:
                if key in agg:
                    print(f'  {key:25s}  median={agg[key]["median"]:.4f}')


if __name__ == '__main__':
    main()
