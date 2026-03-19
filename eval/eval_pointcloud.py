"""
eval_pointcloud.py — Unified Python evaluation module for mmDar / RadarHD experiments.

Provides importable functions and a CLI entry point for computing point-cloud and
polar-image metrics between model predictions and ground-truth lidar images.

Metric definitions match the MATLAB reference in eval/pc_distance.m exactly:
  - Chamfer: 0.5 * mean(nn_A→B) + 0.5 * mean(nn_B→A)
  - Modified Hausdorff: max(median(nn_A→B), median(nn_B→A))

Coordinate constants match eval/image_to_pcd.py:
  RMAX=10.8, RBINS=256, ABINS=512
"""

import os
import glob
import json
import csv
import subprocess
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe inside Docker / headless
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants (must match eval/image_to_pcd.py exactly)
# ---------------------------------------------------------------------------

RMAX = 10.8
RBINS = 256
ABINS = 512
MIN_THRESHOLD = 1
MAX_THRESHOLD = 255
COORD_MODE_LEGACY = "legacy_cartesian"
COORD_MODE_DIRECT = "polar_direct"
VALID_COORD_MODES = {COORD_MODE_LEGACY, COORD_MODE_DIRECT}

# Pre-compute coordinate grids (module-level, computed once)
_x_axis_grid = np.linspace(0, RMAX, RBINS)   # shape (256,)  — range axis
_y_axis_grid = np.linspace(-RMAX, RMAX, ABINS)  # shape (512,) — azimuth axis

# Legacy pol_to_cart mapping grids (must match eval/pol_to_cart.py)
_agrid = np.linspace(-90, 90, ABINS)
_rgrid = np.linspace(0, RMAX, RBINS)
_singrid = np.sin(_agrid * np.pi / 180)
_sine_theta, _range_d = np.meshgrid(_singrid, _rgrid)
_cos_theta = np.sqrt(1 - _sine_theta**2)
_legacy_x_axis = np.multiply(_range_d, _cos_theta)
_legacy_y_axis = np.multiply(_range_d, _sine_theta)

# ---------------------------------------------------------------------------
# Coordinate-space conversion
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def polar_to_cartesian_legacy(img: np.ndarray) -> np.ndarray:
    """Apply legacy polar->cartesian remapping from eval/pol_to_cart.py."""
    img_u8 = _to_uint8(img)
    cart = np.zeros((RBINS, ABINS), dtype=np.uint8)
    nonzero_loc = np.argwhere(img_u8 > 0)
    if nonzero_loc.size == 0:
        return cart

    row_idx = nonzero_loc[:, 0]
    col_idx = nonzero_loc[:, 1]
    x_vals = _legacy_x_axis[row_idx, col_idx]
    y_vals = _legacy_y_axis[row_idx, col_idx]

    new_row_idx = np.searchsorted(_x_axis_grid, x_vals, side="left")
    new_col_idx = np.searchsorted(_y_axis_grid, y_vals, side="left")
    new_row_idx = np.clip(new_row_idx, 0, RBINS - 1)
    new_col_idx = np.clip(new_col_idx, 0, ABINS - 1)
    cart[new_row_idx, new_col_idx] = img_u8[row_idx, col_idx]
    return cart


def polar_image_to_pointcloud(img: np.ndarray,
                               threshold: int = MIN_THRESHOLD,
                               coordinate_mode: str = COORD_MODE_LEGACY) -> np.ndarray:
    """Convert a grayscale polar image (RBINS x ABINS) to a 2-D point cloud.

    Applies the same thresholding and grid mapping as image_to_pcd.py so that
    metric values are directly comparable with the MATLAB pipeline output.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image, shape (RBINS, ABINS) = (256, 512), dtype uint8 or float.
        Row dimension = range (x), column dimension = azimuth (y).
    threshold : int
        Pixels at or below this value are zeroed (THRESH_TOZERO).
    coordinate_mode : str
        - "legacy_cartesian": apply legacy pol_to_cart remap before point extraction
          (paper-comparable mode).
        - "polar_direct": map directly from image indices to metric grids.

    Returns
    -------
    np.ndarray
        Float64 array of shape (N, 2) with columns (x_meters, y_meters).
        Returns shape (0, 2) when no pixels survive thresholding.
    """
    if coordinate_mode not in VALID_COORD_MODES:
        raise ValueError(f"Unknown coordinate_mode='{coordinate_mode}'. "
                         f"Expected one of {sorted(VALID_COORD_MODES)}")

    if coordinate_mode == COORD_MODE_LEGACY:
        img_u8 = polar_to_cartesian_legacy(img)
    else:
        img_u8 = _to_uint8(img)

    _ret, thresh_img = cv2.threshold(img_u8, threshold, MAX_THRESHOLD, cv2.THRESH_TOZERO)

    pts = cv2.findNonZero(thresh_img)
    if pts is None:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.squeeze(pts)  # (N, 2) with cols [col_idx, row_idx]
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]  # single point edge case

    # pts[:, 0] = column index → azimuth → y_axis_grid
    # pts[:, 1] = row    index → range  → x_axis_grid
    col_idx = pts[:, 0]
    row_idx = pts[:, 1]

    x_meters = _x_axis_grid[row_idx]
    y_meters = _y_axis_grid[col_idx]

    return np.column_stack((x_meters, y_meters)).astype(np.float64)


# ---------------------------------------------------------------------------
# Point-cloud distance metrics (must match pc_distance.m)
# ---------------------------------------------------------------------------

def chamfer_distance(pc_a: np.ndarray, pc_b: np.ndarray) -> float:
    """Symmetric Chamfer distance matching MATLAB pc_distance.m.

    Formula: 0.5 * mean(min_{b} dist(a, b)) + 0.5 * mean(min_{a} dist(b, a))

    Parameters
    ----------
    pc_a, pc_b : np.ndarray
        Point clouds of shape (N, D) and (M, D).  Must be non-empty.

    Returns
    -------
    float
        Chamfer distance in the same units as the input coordinates (meters).
    """
    D = cdist(pc_a, pc_b)          # (N, M)
    d_a2b = D.min(axis=1).mean()   # mean nearest-neighbour dist A→B
    d_b2a = D.min(axis=0).mean()   # mean nearest-neighbour dist B→A
    return float(0.5 * d_a2b + 0.5 * d_b2a)


def modified_hausdorff(pc_a: np.ndarray, pc_b: np.ndarray) -> float:
    """Modified Hausdorff distance matching MATLAB pc_distance.m.

    Formula: max(median(nn_A→B), median(nn_B→A))

    Parameters
    ----------
    pc_a, pc_b : np.ndarray
        Point clouds of shape (N, D) and (M, D).  Must be non-empty.

    Returns
    -------
    float
        Modified Hausdorff distance in meters.
    """
    D = cdist(pc_a, pc_b)
    d_a2b = np.median(D.min(axis=1))
    d_b2a = np.median(D.min(axis=0))
    return float(max(d_a2b, d_b2a))


def precision_recall_at_threshold(pc_pred: np.ndarray,
                                   pc_label: np.ndarray,
                                   threshold_m: float = 0.1) -> dict:
    """Point-cloud precision and recall at a distance threshold.

    Precision: fraction of predicted points within `threshold_m` of any label point.
    Recall:    fraction of label points within `threshold_m` of any predicted point.

    Parameters
    ----------
    pc_pred : np.ndarray  shape (N, D)
    pc_label : np.ndarray shape (M, D)
    threshold_m : float   distance threshold in metres

    Returns
    -------
    dict with keys: precision, recall, f1, threshold_m
    """
    D = cdist(pc_pred, pc_label)
    precision = float((D.min(axis=1) <= threshold_m).mean())
    recall    = float((D.min(axis=0) <= threshold_m).mean())
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold_m": float(threshold_m),
    }


# ---------------------------------------------------------------------------
# Polar-image metrics
# ---------------------------------------------------------------------------

def polar_iou_f1(pred_float: np.ndarray,
                 label_binary: np.ndarray,
                 threshold: float = 0.5) -> dict:
    """IoU and F1 from polar images (float prediction vs binary ground truth).

    Parameters
    ----------
    pred_float : np.ndarray
        Model output image, values in [0, 1] (or [0, 255] — normalised internally).
        Shape (H, W).
    label_binary : np.ndarray
        Ground truth binary mask.  Any non-zero value is treated as positive.
        Shape (H, W).
    threshold : float
        Binarisation threshold for pred_float (applied after normalisation to [0,1]).

    Returns
    -------
    dict with keys: iou, f1, precision, recall
    """
    # Normalise prediction to [0, 1] if values exceed 1
    pf = pred_float.astype(np.float64)
    if pf.max() > 1.0:
        pf = pf / 255.0

    pred_bin  = (pf >= threshold).astype(bool)
    label_bin = (label_binary > 0).astype(bool)

    intersection = int((pred_bin & label_bin).sum())
    union        = int((pred_bin | label_bin).sum())

    iou       = intersection / (union + 1e-8)
    tp        = intersection
    fp        = int((pred_bin & ~label_bin).sum())
    fn        = int((~pred_bin & label_bin).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "iou":       float(iou),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_side_by_side(radar_img, pred_img, label_img, save_path: str,
                      title: str = "") -> None:
    """Save a 1×3 side-by-side PNG: radar input / prediction / ground truth.

    Parameters
    ----------
    radar_img : np.ndarray or None
        Radar input image.  Pass None to render a blank panel.
    pred_img : np.ndarray
        Model prediction image.
    label_img : np.ndarray
        Ground truth image.
    save_path : str
        Full path for the output PNG file.
    title : str
        Optional suptitle (e.g. sample name + metrics).
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        (radar_img if radar_img is not None else np.zeros_like(pred_img), "Radar Input"),
        (pred_img, "Prediction"),
        (label_img, "Ground Truth"),
    ]

    for ax, (img, panel_title) in zip(axes, panels):
        if img is not None:
            ax.imshow(img, cmap="gray", aspect="auto",
                      vmin=0, vmax=255 if img.max() > 1 else 1)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=9, y=1.01)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Temporal consistency metric
# ---------------------------------------------------------------------------

def _sort_key(path):
    """Sort key that extracts (traj_id, frame_idx) from prediction filenames.

    Handles two naming conventions:
      - "{traj_id}_{frame_idx}_pred.png"         (test_convlstm.py output)
      - "{epoch}_{traj_id}_{frame_idx}_pred.png"  (run_experiment.py sweep output)

    Strategy: strip the '_pred.png' suffix, then rsplit('_', 2) to get the
    rightmost two underscore-separated tokens as (traj_id, frame_idx).
    """
    stem = Path(path).name.replace('_pred.png', '')
    parts = stem.rsplit('_', 2)
    try:
        traj_id   = int(parts[-2])
        frame_idx = int(parts[-1])
    except (ValueError, IndexError):
        # Fall back to lexicographic sort for unexpected filenames
        return (0, 0, path)
    return (traj_id, frame_idx)


def temporal_consistency(pred_dir, coordinate_mode=COORD_MODE_LEGACY):
    """Compute frame-to-frame Chamfer distance between consecutive predictions.

    Groups prediction files by trajectory, then computes Chamfer distance between
    consecutive frames within each trajectory. Frames from different trajectories
    are never compared (no boundary crossings).

    Predictions must be named with sortable trajectory+frame indices:
      - "{traj_id}_{frame_idx}_pred.png"  OR
      - "{epoch}_{traj_id}_{frame_idx}_pred.png"

    Args:
        pred_dir: directory containing *_pred.png files
        coordinate_mode: coordinate conversion mode for polar_image_to_pointcloud

    Returns:
        dict with keys:
          'mean'   : float — mean frame-to-frame Chamfer distance
          'median' : float — median frame-to-frame Chamfer distance
          'std'    : float — std of frame-to-frame Chamfer distances
          'count'  : int   — number of consecutive pairs evaluated
          'scores' : list[float] — per-pair Chamfer distances
    """
    pred_files = sorted(
        glob.glob(os.path.join(pred_dir, '*_pred.png')),
        key=_sort_key,
    )

    if not pred_files:
        return {'mean': float('nan'), 'median': float('nan'),
                'std': float('nan'), 'count': 0, 'scores': []}

    # Group files by trajectory ID (parse from filename using same logic as _sort_key)
    # Use an ordered dict keyed by traj_id to preserve sort order
    from collections import defaultdict
    traj_groups = defaultdict(list)
    for p in pred_files:
        stem = Path(p).name.replace('_pred.png', '')
        parts = stem.rsplit('_', 2)
        try:
            traj_id = int(parts[-2])
        except (ValueError, IndexError):
            traj_id = -1
        traj_groups[traj_id].append(p)

    scores = []
    for traj_id in sorted(traj_groups.keys()):
        files = traj_groups[traj_id]  # already sorted by (traj_id, frame_idx) from above
        for i in range(len(files) - 1):
            img_a = cv2.imread(files[i],   cv2.IMREAD_GRAYSCALE)
            img_b = cv2.imread(files[i+1], cv2.IMREAD_GRAYSCALE)
            if img_a is None or img_b is None:
                continue
            pc_a = polar_image_to_pointcloud(img_a, coordinate_mode=coordinate_mode)
            pc_b = polar_image_to_pointcloud(img_b, coordinate_mode=coordinate_mode)
            if pc_a.shape[0] == 0 or pc_b.shape[0] == 0:
                # Skip pairs where either prediction is empty
                continue
            scores.append(chamfer_distance(pc_a, pc_b))

    if not scores:
        return {'mean': float('nan'), 'median': float('nan'),
                'std': float('nan'), 'count': 0, 'scores': []}

    arr = np.array(scores, dtype=np.float64)
    return {
        'mean':   float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std':    float(np.std(arr)),
        'count':  len(scores),
        'scores': scores,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Return short git commit hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def evaluate_experiment(pred_dir: str,
                        label_dir: str,
                        output_dir: str,
                        experiment_name: str = "experiment",
                        radar_dir: str = None,
                        threshold: float = 0.5,
                        coordinate_mode: str = COORD_MODE_LEGACY) -> dict:
    """Evaluate all prediction/label image pairs in the given directories.

    File naming convention:
      predictions: <stem>_pred.png
      labels:      <stem>_label.png
      radar input: <stem>_radar.png  (optional)

    Metrics computed per sample:
      - polar_iou, polar_f1, polar_precision, polar_recall
      - chamfer_distance (metres), modified_hausdorff (metres)
      - precision_at_0.1m, recall_at_0.1m, f1_at_0.1m

    Aggregate statistics (mean, median, std) saved to:
      <output_dir>/metrics.json
      <output_dir>/metrics.csv

    Visualisations saved to:
      <output_dir>/plots/  — first 5 samples + worst 5 by Chamfer distance

    Parameters
    ----------
    pred_dir : str
        Directory containing *_pred.png files.
    label_dir : str
        Directory containing *_label.png files.
    output_dir : str
        Destination for metrics and plots.
    experiment_name : str
        Human-readable name embedded in output JSON.
    radar_dir : str or None
        Optional directory containing *_radar.png for visualisation.
    threshold : float
        Binarisation threshold for polar IoU/F1 (default 0.5).
    coordinate_mode : str
        Point-cloud conversion mode ("legacy_cartesian" or "polar_direct").

    Returns
    -------
    dict
        Aggregate metrics dictionary (mean / median / std per metric).
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.png")))
    if not pred_files:
        raise FileNotFoundError(f"No *_pred.png files found in {pred_dir}")
    if coordinate_mode not in VALID_COORD_MODES:
        raise ValueError(f"Unknown coordinate_mode='{coordinate_mode}'. "
                         f"Expected one of {sorted(VALID_COORD_MODES)}")

    output_path = Path(output_dir)
    plots_path = output_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict] = []
    skipped = 0

    for pred_file in pred_files:
        stem = Path(pred_file).name.replace("_pred.png", "")
        label_file = os.path.join(label_dir, f"{stem}_label.png")

        if not os.path.exists(label_file):
            print(f"  [WARN] No label for {stem}, skipping.")
            skipped += 1
            continue

        pred_img  = cv2.imread(pred_file,  cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        if pred_img is None or label_img is None:
            print(f"  [WARN] Could not read image pair for {stem}, skipping.")
            skipped += 1
            continue

        radar_img = None
        if radar_dir is not None:
            radar_file = os.path.join(radar_dir, f"{stem}_radar.png")
            if os.path.exists(radar_file):
                radar_img = cv2.imread(radar_file, cv2.IMREAD_GRAYSCALE)

        # --- Polar image metrics ---
        polar_metrics = polar_iou_f1(pred_img, label_img, threshold=threshold)

        # --- Point-cloud metrics ---
        pc_pred  = polar_image_to_pointcloud(pred_img, coordinate_mode=coordinate_mode)
        pc_label = polar_image_to_pointcloud(label_img, coordinate_mode=coordinate_mode)

        sample_row: dict = {"stem": stem, **polar_metrics}

        if pc_pred.shape[0] == 0 or pc_label.shape[0] == 0:
            # Cannot compute point-cloud metrics — record NaN
            sample_row.update({
                "chamfer_distance":  float("nan"),
                "modified_hausdorff": float("nan"),
                "precision_at_0.1m": float("nan"),
                "recall_at_0.1m":    float("nan"),
                "f1_at_0.1m":        float("nan"),
            })
        else:
            cd  = chamfer_distance(pc_pred, pc_label)
            mh  = modified_hausdorff(pc_pred, pc_label)
            pr  = precision_recall_at_threshold(pc_pred, pc_label, threshold_m=0.1)
            sample_row.update({
                "chamfer_distance":  cd,
                "modified_hausdorff": mh,
                "precision_at_0.1m": pr["precision"],
                "recall_at_0.1m":    pr["recall"],
                "f1_at_0.1m":        pr["f1"],
            })

        per_sample.append(sample_row)

    if not per_sample:
        raise RuntimeError("All samples were skipped — no valid pairs found.")

    # --- Aggregate statistics ---
    metric_keys = [k for k in per_sample[0].keys() if k != "stem"]
    aggregate: dict = {}
    for key in metric_keys:
        vals = np.array([s[key] for s in per_sample], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        aggregate[key] = {
            "mean":   float(np.mean(valid))   if len(valid) else float("nan"),
            "median": float(np.median(valid)) if len(valid) else float("nan"),
            "std":    float(np.std(valid))    if len(valid) else float("nan"),
            "n_valid": int(len(valid)),
            "n_nan":   int(np.isnan(vals).sum()),
        }

    result = {
        "experiment_name": experiment_name,
        "git_commit": _get_git_commit(),
        "n_samples":  len(per_sample),
        "n_skipped":  skipped,
        "threshold":  threshold,
        "coordinate_mode": coordinate_mode,
        "aggregate":  aggregate,
        "per_sample": per_sample,
    }

    # --- Save metrics.json ---
    with open(output_path / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # --- Save metrics.csv ---
    csv_path = output_path / "metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem"] + metric_keys)
        writer.writeheader()
        writer.writerows(per_sample)

    # --- Save git commit ---
    (output_path / "git_commit.txt").write_text(result["git_commit"] + "\n")

    # --- Visualisations ---
    # First 5 samples
    viz_indices = list(range(min(5, len(per_sample))))

    # Worst 5 by Chamfer distance (excluding NaN)
    chamfer_vals = [(i, s["chamfer_distance"]) for i, s in enumerate(per_sample)
                    if not np.isnan(s["chamfer_distance"])]
    chamfer_vals.sort(key=lambda x: x[1], reverse=True)
    worst_indices = [i for i, _ in chamfer_vals[:5]]
    viz_indices = list(dict.fromkeys(viz_indices + worst_indices))  # dedup, preserve order

    for idx in viz_indices:
        s = per_sample[idx]
        stem = s["stem"]
        pred_img  = cv2.imread(os.path.join(pred_dir,  f"{stem}_pred.png"),  cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(os.path.join(label_dir, f"{stem}_label.png"), cv2.IMREAD_GRAYSCALE)
        radar_img_v = None
        if radar_dir:
            rp = os.path.join(radar_dir, f"{stem}_radar.png")
            if os.path.exists(rp):
                radar_img_v = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)

        title = (f"{stem}  |  CD={s['chamfer_distance']:.3f}m  "
                 f"MH={s['modified_hausdorff']:.3f}m  IoU={s['iou']:.3f}")
        save_side_by_side(radar_img_v, pred_img, label_img,
                          str(plots_path / f"{stem}.png"), title=title)

    print(f"\n[evaluate_experiment] {experiment_name}")
    print(f"  Samples: {len(per_sample)}  |  Skipped: {skipped}")
    print(f"  Point-cloud coordinate mode: {coordinate_mode}")
    for key in ["chamfer_distance", "modified_hausdorff", "iou", "f1"]:
        if key in aggregate:
            agg = aggregate[key]
            print(f"  {key:25s}  median={agg['median']:.4f}  mean={agg['mean']:.4f}  std={agg['std']:.4f}")
    print(f"  Output: {output_path}")

    return aggregate


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Point-cloud evaluation for mmDar experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred-dir",         required=True,
                        help="Directory containing *_pred.png prediction images.")
    parser.add_argument("--label-dir",        required=True,
                        help="Directory containing *_label.png ground-truth images.")
    parser.add_argument("--output-dir",       required=True,
                        help="Destination directory for metrics JSON/CSV and plots.")
    parser.add_argument("--experiment-name",  default="experiment",
                        help="Human-readable name for this evaluation run.")
    parser.add_argument("--radar-dir",        default=None,
                        help="Optional directory with *_radar.png radar input images "
                             "(used only for side-by-side visualisations).")
    parser.add_argument("--threshold",        type=float, default=0.5,
                        help="Binarisation threshold for polar IoU/F1 (values in [0,1]).")
    parser.add_argument("--coordinate-mode",  type=str, default=COORD_MODE_LEGACY,
                        choices=sorted(VALID_COORD_MODES),
                        help="Point-cloud conversion mode. "
                             "'legacy_cartesian' matches original RadarHD eval flow.")

    args = parser.parse_args()

    aggregate = evaluate_experiment(
        pred_dir=args.pred_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        radar_dir=args.radar_dir,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
    )

    # Print concise summary to stdout
    print("\n=== Summary ===")
    for key, stats in aggregate.items():
        print(f"  {key:25s}  median={stats['median']:.4f}  mean={stats['mean']:.4f}")


if __name__ == "__main__":
    main()
