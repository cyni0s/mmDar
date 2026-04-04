"""Decompose prediction errors into range vs cross-range (angular) components.

For each predicted Gaussian center, finds nearest GT point and decomposes
the error vector into radial (range) and tangential (cross-range) components
relative to the sensor at the origin.

This tells us: is the 20cm mod-H dominated by range errors or angular errors?

Usage:
  docker compose run --rm mmdar python3 eval/error_decompose.py \
    --checkpoint logs/verify_cleanup/best.pt \
    --processed-dir data/processed \
    --split mixed
"""

import sys
import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.physics_frontend import PhysicsGaussianModel
from train.train import AugmentedGaussianDataset, get_split


def decompose_errors(pred_xy, gt_xy):
    """Decompose nearest-neighbor errors into range and cross-range.

    For each predicted point, find nearest GT point. Decompose the error
    vector into radial (range) and tangential (cross-range) relative to
    the sensor at origin.

    Args:
        pred_xy: (N, 2) predicted XY
        gt_xy: (M, 2) ground truth XY

    Returns:
        dict with range_errors, xrange_errors, total_errors (all absolute, in metres)
    """
    # Nearest neighbor: pred -> GT
    D = torch.cdist(pred_xy, gt_xy)  # (N, M)
    nn_idx = D.argmin(dim=1)  # (N,)
    nn_gt = gt_xy[nn_idx]  # (N, 2)

    # Error vectors
    err = pred_xy - nn_gt  # (N, 2) displacement

    # Radial direction from origin to predicted point
    r = torch.norm(pred_xy, dim=1, keepdim=True).clamp(min=1e-6)  # (N, 1)
    radial = pred_xy / r  # (N, 2) unit radial
    tangential = torch.stack([-radial[:, 1], radial[:, 0]], dim=1)  # (N, 2) unit tangential

    # Project error onto radial and tangential
    range_err = (err * radial).sum(dim=1)  # (N,) signed range error
    xrange_err = (err * tangential).sum(dim=1)  # (N,) signed cross-range error
    total_err = torch.norm(err, dim=1)  # (N,) total error

    # Also decompose GT -> pred (reverse direction for mod-H)
    nn_idx_rev = D.argmin(dim=0)  # (M,)
    nn_pred = pred_xy[nn_idx_rev]  # (M, 2)
    err_rev = gt_xy - nn_pred  # (M, 2)

    r_gt = torch.norm(gt_xy, dim=1, keepdim=True).clamp(min=1e-6)
    radial_gt = gt_xy / r_gt
    tangential_gt = torch.stack([-radial_gt[:, 1], radial_gt[:, 0]], dim=1)

    range_err_rev = (err_rev * radial_gt).sum(dim=1)
    xrange_err_rev = (err_rev * tangential_gt).sum(dim=1)

    return {
        # pred -> GT direction
        "range_abs": range_err.abs(),
        "xrange_abs": xrange_err.abs(),
        "total": total_err,
        "range_signed": range_err,
        "xrange_signed": xrange_err,
        # GT -> pred direction (for coverage analysis)
        "rev_range_abs": range_err_rev.abs(),
        "rev_xrange_abs": xrange_err_rev.abs(),
        # Ranges of predicted points (for error vs distance analysis)
        "pred_range": r.squeeze(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", default="mixed")
    parser.add_argument("--window-size", type=int, default=41)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Max frames to evaluate (for speed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = PhysicsGaussianModel(
        N_az=cfg.get("N_az", 64),
        T=cfg.get("window_size", args.window_size),
        K=cfg.get("K", 96),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}", flush=True)

    # Load test data
    _, _, TEST_TRAJS, _ = get_split(args.split)
    print(f"Test trajectories: {TEST_TRAJS}", flush=True)

    # Collect errors
    all_range = []
    all_xrange = []
    all_total = []
    all_rev_range = []
    all_rev_xrange = []
    all_pred_ranges = []
    n_frames = 0

    with torch.no_grad():
        for tid in TEST_TRAJS:
            proto_path = os.path.join(args.processed_dir, f"proto_{tid}.pt")
            if not os.path.exists(proto_path):
                continue
            ds = AugmentedGaussianDataset(tid, args.processed_dir,
                                           args.window_size, augment=False)
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            for radar, lidar, protos in loader:
                if n_frames >= args.max_frames:
                    break

                points = model.predict_points(radar.to(device),
                                               threshold=args.threshold)
                pred_xy = points[0]  # (N_pred, 2)
                gt_xy = lidar[0, :, :2].to(device)  # (8192, 2)

                if pred_xy.shape[0] < 2:
                    continue

                errs = decompose_errors(pred_xy, gt_xy)
                all_range.append(errs["range_abs"].cpu())
                all_xrange.append(errs["xrange_abs"].cpu())
                all_total.append(errs["total"].cpu())
                all_rev_range.append(errs["rev_range_abs"].cpu())
                all_rev_xrange.append(errs["rev_xrange_abs"].cpu())
                all_pred_ranges.append(errs["pred_range"].cpu())
                n_frames += 1

            if n_frames >= args.max_frames:
                break

    print(f"\nAnalyzed {n_frames} frames\n")

    # Aggregate
    range_err = torch.cat(all_range)
    xrange_err = torch.cat(all_xrange)
    total_err = torch.cat(all_total)
    rev_range = torch.cat(all_rev_range)
    rev_xrange = torch.cat(all_rev_xrange)
    pred_ranges = torch.cat(all_pred_ranges)

    print("=" * 60)
    print("ERROR DECOMPOSITION: pred -> GT (precision)")
    print("=" * 60)
    print(f"  {'':20s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s}")
    print(f"  {'Range (radial)':20s} {range_err.mean():.4f} {range_err.median():.4f} "
          f"{range_err.quantile(0.9):.4f} {range_err.quantile(0.95):.4f}")
    print(f"  {'Cross-range (angular)':20s} {xrange_err.mean():.4f} {xrange_err.median():.4f} "
          f"{xrange_err.quantile(0.9):.4f} {xrange_err.quantile(0.95):.4f}")
    print(f"  {'Total':20s} {total_err.mean():.4f} {total_err.median():.4f} "
          f"{total_err.quantile(0.9):.4f} {total_err.quantile(0.95):.4f}")
    print(f"\n  Angular/Total ratio: {xrange_err.mean() / total_err.mean():.1%}")

    print(f"\n{'=' * 60}")
    print("ERROR DECOMPOSITION: GT -> pred (coverage)")
    print("=" * 60)
    print(f"  {'':20s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s}")
    print(f"  {'Range (radial)':20s} {rev_range.mean():.4f} {rev_range.median():.4f} "
          f"{rev_range.quantile(0.9):.4f} {rev_range.quantile(0.95):.4f}")
    print(f"  {'Cross-range (angular)':20s} {rev_xrange.mean():.4f} {rev_xrange.median():.4f} "
          f"{rev_xrange.quantile(0.9):.4f} {rev_xrange.quantile(0.95):.4f}")

    # Error vs distance bins
    print(f"\n{'=' * 60}")
    print("ERROR vs RANGE (pred -> GT)")
    print("=" * 60)
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 11)]
    print(f"  {'Range bin':12s} {'N pts':>7s} {'Range err':>10s} {'Xrange err':>11s} {'Total':>8s} {'Ang/Tot':>8s}")
    for lo, hi in bins:
        mask = (pred_ranges >= lo) & (pred_ranges < hi)
        if mask.sum() < 10:
            continue
        re = range_err[mask]
        xe = xrange_err[mask]
        te = total_err[mask]
        print(f"  {lo:2d}-{hi:2d}m        {mask.sum():7d} {re.mean():10.4f} {xe.mean():11.4f} "
              f"{te.mean():8.4f} {xe.mean()/te.mean():8.1%}")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    ratio = xrange_err.mean() / total_err.mean()
    if ratio > 0.6:
        print("  >> Angular error DOMINATES. Improve azimuth resolution.")
    elif ratio < 0.4:
        print("  >> Range error DOMINATES. Improve range estimation.")
    else:
        print("  >> Errors roughly balanced between range and angular.")
    print(f"  >> Angular fraction: {ratio:.1%}")
    print(f"  >> Mean total error: {total_err.mean():.3f}m")
    print(f"  >> Points evaluated: {len(total_err):,}")


if __name__ == "__main__":
    main()
