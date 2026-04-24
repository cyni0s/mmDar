"""Gaussian-model inference CLI for class submission.

Loads a trained PhysicsGaussianModel checkpoint, runs inference on one or more
trajectories from ``data/processed/``, computes Chamfer and modified Hausdorff
distance per frame, and writes the results to an output directory.

Thin wrapper around ``eval.eval_adapter`` (which has no CLI of its own) and
``data.windowed_dataset.WindowedTrajectoryDataset`` (which takes a single trajectory
at a time). Exists so the README's quick-start command works from a fresh
clone without knowing the training entry-point API.

Usage
-----
Inside Docker:

    # Demo on trajectory 250 (ships with the repo)
    python3 tools/run_inference.py \\
        --checkpoint checkpoints/physics_gaussian_headline.pt \\
        --trajectories 250 \\
        --output results/demo_gaussian/

    # Full test set (requires all 44 preprocessed trajectories)
    python3 tools/run_inference.py \\
        --checkpoint checkpoints/physics_gaussian_mixed.pt \\
        --split mixed --trajectories all \\
        --output results/mixed_full/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.windowed_dataset import WindowedTrajectoryDataset  # noqa: E402
from eval.eval_adapter import chamfer_distance_np, mod_hausdorff_np  # noqa: E402
from model.physics_frontend import PhysicsGaussianModel  # noqa: E402

DEFAULT_CFG = {"K": 96, "N_az": 64, "window_size": 41}
EXISTENCE_THRESHOLD = 0.3  # matches the value that produced reported mod-H 0.205


def parse_trajectories(arg: str, split: str) -> list[int]:
    if arg.strip().lower() == "all":
        if split == "v2":
            from data.split_v2 import TEST_TRAJS
        elif split == "mixed":
            from data.split_mixed import TEST_TRAJS
        elif split == "original":
            from data.split import TEST_TRAJS
        else:
            raise ValueError(f"Unknown split {split!r}")
        return list(TEST_TRAJS)
    return [int(t.strip()) for t in arg.split(",") if t.strip()]


def load_cfg(ckpt_path: Path) -> dict:
    """Read hyperparameters from a sibling config_*.json if present, else defaults."""
    candidates = [
        ckpt_path.parent / f"config_{ckpt_path.stem.replace('physics_gaussian_', '')}.json",
        ckpt_path.parent / "config.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                cfg = json.load(f)
            return {**DEFAULT_CFG, **cfg}
    return dict(DEFAULT_CFG)


def build_model(cfg: dict, device: torch.device) -> PhysicsGaussianModel:
    model = PhysicsGaussianModel(
        N_az=cfg["N_az"],
        T=cfg["window_size"],
        K=cfg["K"],
    ).to(device)
    model.eval()
    return model


def load_state(model: PhysicsGaussianModel, ckpt_path: Path) -> None:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        state = blob["model"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")


def pad_to_3d(pts_2d: np.ndarray) -> np.ndarray:
    """(N, 2) -> (N, 3) with z=0 so eval helpers can consume it."""
    if pts_2d.shape[0] == 0:
        return pts_2d.reshape(0, 3)
    return np.concatenate([pts_2d, np.zeros((pts_2d.shape[0], 1), dtype=pts_2d.dtype)], axis=1)


def run_trajectory(
    model: PhysicsGaussianModel,
    traj_id: int,
    processed_dir: Path,
    window_size: int,
    device: torch.device,
) -> list[dict]:
    ds = WindowedTrajectoryDataset(
        traj_id=traj_id,
        processed_dir=str(processed_dir),
        window_size=window_size,
    )
    rows: list[dict] = []
    with torch.no_grad():
        for frame_idx in range(len(ds)):
            radar_window, lidar, _norm = ds[frame_idx]
            x = radar_window.unsqueeze(0).to(device)
            gt = lidar.numpy()

            out = model(x)
            existence_prob = torch.sigmoid(out["existence"][0])
            mu_xy = out["mu_xy"][0]
            mask = existence_prob > EXISTENCE_THRESHOLD
            pred_xy = mu_xy[mask].cpu().numpy()
            pred_pts = pad_to_3d(pred_xy)

            if pred_pts.shape[0] < 2 or gt.shape[0] < 2:
                rows.append({
                    "traj": traj_id,
                    "frame_idx": frame_idx,
                    "n_pred": int(pred_pts.shape[0]),
                    "n_gt": int(gt.shape[0]),
                    "chamfer": None,
                    "mod_h": None,
                })
                continue

            rows.append({
                "traj": traj_id,
                "frame_idx": frame_idx,
                "n_pred": int(pred_pts.shape[0]),
                "n_gt": int(gt.shape[0]),
                "chamfer": chamfer_distance_np(pred_pts, gt),
                "mod_h": mod_hausdorff_np(pred_pts, gt),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    cd = np.array([r["chamfer"] for r in rows if r["chamfer"] is not None])
    mh = np.array([r["mod_h"] for r in rows if r["mod_h"] is not None])
    return {
        "n_frames_total": len(rows),
        "n_frames_scored": int(cd.size),
        "chamfer_mean": float(cd.mean()) if cd.size else None,
        "chamfer_median": float(np.median(cd)) if cd.size else None,
        "mod_h_mean": float(mh.mean()) if mh.size else None,
        "mod_h_median": float(np.median(mh)) if mh.size else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectories", type=str, default="all",
                   help="Comma-separated trajectory IDs, or 'all' for the split's test set.")
    p.add_argument("--split", choices=["v2", "mixed", "original"], default="v2",
                   help="Only used when --trajectories=all.")
    p.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data/processed")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    trajectories = parse_trajectories(args.trajectories, args.split)
    cfg = load_cfg(args.checkpoint)
    device = torch.device(args.device)

    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Config:         K={cfg['K']}, N_az={cfg['N_az']}, window={cfg['window_size']}")
    print(f"Trajectories:   {trajectories}")
    print(f"Device:         {device}")
    print(f"Processed dir:  {args.processed_dir}")
    print(f"Existence cut:  {EXISTENCE_THRESHOLD}")
    print()

    model = build_model(cfg, device)
    load_state(model, args.checkpoint)

    all_rows: list[dict] = []
    for traj in trajectories:
        if not (args.processed_dir / f"radar_{traj}.pt").exists():
            print(f"[SKIP] trajectory {traj}: radar_{traj}.pt not found in {args.processed_dir}")
            continue
        print(f"Running trajectory {traj} ...")
        traj_rows = run_trajectory(
            model, traj, args.processed_dir, cfg["window_size"], device,
        )
        traj_summary = summarize(traj_rows)
        print(f"  {traj_summary['n_frames_scored']}/{traj_summary['n_frames_total']} frames scored"
              f"   chamfer_median={traj_summary['chamfer_median']:.3f}"
              f"   mod_h_median={traj_summary['mod_h_median']:.3f}")
        all_rows.extend(traj_rows)

    summary = summarize(all_rows)
    summary["trajectories"] = trajectories
    summary["existence_threshold"] = EXISTENCE_THRESHOLD
    summary["checkpoint"] = str(args.checkpoint)

    out_path = args.output / "metrics.json"
    out_path.write_text(json.dumps({"summary": summary, "per_frame": all_rows}, indent=2))

    print()
    print("=== Aggregate summary ===")
    for k in ("n_frames_scored", "chamfer_mean", "chamfer_median",
              "mod_h_mean", "mod_h_median"):
        v = summary[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
