"""
v2 Radar Preprocessing Pipeline
================================

Converts raw RadarHD-dataset radar pickles and lidar CSVs into training-ready .pt tensor files.

Processing steps per trajectory:
  1. Load raw radar frames: (N_frames, 192, 4, 512) complex128
  2. DC offset correction: trajectory-level per-antenna mean subtraction (NOT per-frame)
  3. Per-frame TDM-MIMO processing:
     a. Deinterleave 192 chirps into 64-chirp sequences: TX0 = chirps[0::3], TX2 = chirps[2::3]
     b. Range FFT on axis=2 (512 ADC samples -> 512 range bins)
     c. Doppler FFT on axis=0 (64 chirps -> 64 Doppler bins)
     d. TDM-MIMO Doppler phase compensation for TX2: multiply bin m by exp(-j*2*pi*2*m/3)
     e. fftshift to center zero-Doppler, take index 32 (zero-Doppler slice)
     f. Concatenate TX0[32,:,:] and TX2[32,:,:] -> (8, 512) virtual array
     g. Cast to complex64
  4. Per-frame normalization by max magnitude (DATA-08):
     - norm_factor = max(abs(varray)) per frame
     - Store normalization factor per frame in norm_{traj}.pt of shape (N,) float32
     - This allows exact denormalization if needed later
  5. Lidar alignment: apply 5h offset (trajs <= 200) or 4h (trajs > 200), match frames
  6. Lidar FPS: filter to scene volume, farthest point sample to 8192 points

Output per trajectory:
  - v2/data/processed/radar_{traj}.pt  -- (N, 8, 512) complex64 (normalized)
  - v2/data/processed/lidar_{traj}.pt  -- (N, 8192, 3) float32
  - v2/data/processed/norm_{traj}.pt   -- (N,) float32 normalization factors

Assumptions / limitations:
  - Near-field correction (<2m) not applied (targets <1.8m violate far-field assumption)
  - IQ imbalance: check only; correction deferred if imbalance < 1%
  - Trajectory-level DC correction removes static clutter but also attenuates any
    scene-wide DC component common to all frames (acceptable trade-off)
"""

import argparse
import json
import os
import pickle
import zipfile
from datetime import datetime, timezone

import numpy as np

# pandas and torch are imported lazily inside functions that need them so that
# the pure signal-processing functions (process_frame_tdm_mimo, dc_correct, etc.)
# remain testable on the host without Docker or a full Python environment.
# Functions that require these libraries import them at call time.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TX = 3
N_CHIRPS_PER_TX = 64        # 192 interleaved chirps / 3 TXs
N_RX = 4
N_RANGE = 512
N_VIRTUAL_AZ = 8            # TX0 (4 RX) + TX2 (4 RX)
N_LIDAR_PTS = 8192

# Scene volume limits (metres)
SCENE_X = (0.0, 10.0)
SCENE_Y = (-10.0, 10.0)
SCENE_Z = (-0.3, 0.3)

MAX_RANGE = 21.59           # metres per RadarHD dataset


# ---------------------------------------------------------------------------
# Core radar processing
# ---------------------------------------------------------------------------

def process_frame_tdm_mimo(frame: np.ndarray) -> np.ndarray:
    """
    Process one TDM-MIMO radar frame into an 8-element virtual array snapshot.

    Parameters
    ----------
    frame : np.ndarray, shape (192, 4, 512), complex128
        Raw ADC data: interleaved TDM chirps × RX antennas × ADC samples.
        TDM firing order: chirp 0=TX0, 1=TX1, 2=TX2, 3=TX0, 4=TX1, 5=TX2, …

    Returns
    -------
    np.ndarray, shape (8, 512), complex64
        8-element virtual azimuth array at zero-Doppler.
        Rows 0-3: TX0 × RX0-3, rows 4-7: TX2 × RX0-3.
    """
    # Step 1: Deinterleave into per-TX sequences (stride 3)
    tx0 = frame[0::3, :, :]  # (64, 4, 512) — TX0 chirps
    tx2 = frame[2::3, :, :]  # (64, 4, 512) — TX2 chirps

    # Step 2: Range FFT along ADC samples axis
    tx0_r = np.fft.fft(tx0, axis=2)   # (64, 4, 512) range domain
    tx2_r = np.fft.fft(tx2, axis=2)

    # Step 3: Doppler FFT across chirps axis
    tx0_rd = np.fft.fft(tx0_r, axis=0)  # (64, 4, 512) range-Doppler
    tx2_rd = np.fft.fft(tx2_r, axis=0)

    # Step 4: TDM-MIMO Doppler phase compensation for TX2 (k=2)
    # Formula: multiply Doppler bin m by exp(-j * 2*pi * k * m / N_TX)
    # TX0 (k=0): compensation = 1 (no-op, skip)
    # TX2 (k=2): exp(-j * 2*pi * 2 * m / 3)
    m = np.arange(N_CHIRPS_PER_TX, dtype=np.float64)
    comp_tx2 = np.exp(-1j * 2 * np.pi * 2 * m / N_TX)   # (64,)
    tx2_rd = tx2_rd * comp_tx2[:, np.newaxis, np.newaxis]

    # Step 5: fftshift to center zero-Doppler, extract zero-Doppler slice
    tx0_rd = np.fft.fftshift(tx0_rd, axes=0)
    tx2_rd = np.fft.fftshift(tx2_rd, axes=0)
    zero_dop_idx = N_CHIRPS_PER_TX // 2  # index 32 after fftshift

    tx0_zero = tx0_rd[zero_dop_idx, :, :]  # (4, 512)
    tx2_zero = tx2_rd[zero_dop_idx, :, :]  # (4, 512)

    # Step 6: Concatenate to 8-element virtual array
    varray = np.concatenate([tx0_zero, tx2_zero], axis=0)  # (8, 512)

    return varray.astype(np.complex64)


def dc_correct(frames: np.ndarray) -> np.ndarray:
    """
    Trajectory-level DC offset correction.

    Computes per-(chirp, RX, range_bin) mean across ALL frames in the trajectory
    and subtracts it. This is NOT per-frame — per-frame DC correction would remove
    valid signal together with the offset.

    Parameters
    ----------
    frames : np.ndarray, shape (N_frames, 192, 4, 512), complex128

    Returns
    -------
    np.ndarray, shape (N_frames, 192, 4, 512), complex128
        DC corrected frames.
    """
    dc_offset = np.mean(frames, axis=0, keepdims=True)  # (1, 192, 4, 512)
    return frames - dc_offset


def check_iq_balance(varray_traj: np.ndarray, traj_id: int) -> float:
    """
    Check IQ amplitude imbalance on the processed virtual array.

    Examines antenna 0 at its strongest range bin. Computes the ratio of
    real vs imaginary variance to detect IQ imbalance.

    Parameters
    ----------
    varray_traj : np.ndarray, shape (N_frames, 8, 512), complex64
    traj_id : int

    Returns
    -------
    float
        Imbalance percentage: 0 = perfect circular IQ, 100 = fully elliptical.
    """
    # Antenna 0, find strongest range bin by mean power
    ant0 = varray_traj[:, 0, :]           # (N_frames, 512)
    mean_power = np.mean(np.abs(ant0) ** 2, axis=0)  # (512,)
    best_bin = int(np.argmax(mean_power))

    samples = ant0[:, best_bin]             # (N_frames,) complex
    re_var = float(np.var(samples.real))
    im_var = float(np.var(samples.imag))
    denom = max(re_var, im_var)
    if denom < 1e-30:
        imbalance_pct = 0.0
    else:
        imbalance_pct = abs(re_var - im_var) / denom * 100.0

    level = "OK" if imbalance_pct < 1.0 else "WARNING"
    print(
        f"[traj {traj_id}] IQ imbalance: {imbalance_pct:.2f}%  [{level}]"
        f"  (ant0, range_bin={best_bin}, re_var={re_var:.3e}, im_var={im_var:.3e})"
    )
    return imbalance_pct


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------

def _parse_iso(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp string to UTC datetime."""
    # Handle both 'Z' suffix and '+00:00'
    ts_str = ts_str.strip()
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


def build_frame_table(
    traj_id: int,
    radar_data: dict,
    lidar_csv_path: str,
) -> list:
    """
    Align radar frames with lidar timestamps.

    Uses the empirically verified timezone offset rule:
      - Trajectories 1-200: lidar timestamps are 5 hours ahead of radar UTC
      - Trajectories 201+: lidar timestamps are 4 hours ahead of radar UTC

    Parameters
    ----------
    traj_id : int
    radar_data : dict
        Dictionary from the radar pickle containing 'start_time', 'end_time',
        and 'num_frames' keys.
    lidar_csv_path : str

    Returns
    -------
    list of (int, float)
        Sorted list of (radar_frame_idx, lidar_timestamp_sec) pairs.
        radar_frame_idx is sequential 0..N-1 (verified against dataset_5).
    """
    OFFSET_HOURS = 5 if traj_id <= 200 else 4

    t_start = _parse_iso(radar_data["start_time"]).timestamp()
    t_end = _parse_iso(radar_data["end_time"]).timestamp()
    n_radar = radar_data["num_frames"]
    radar_ts = np.linspace(t_start, t_end, n_radar)

    # Read lidar timestamps (column 4 = Unix seconds float)
    import pandas as pd  # noqa: lazy import
    df_ts = pd.read_csv(lidar_csv_path, header=None, usecols=[4])
    lidar_raw = df_ts.iloc[:, 0].to_numpy()
    lidar_adj = np.unique(lidar_raw - OFFSET_HOURS * 3600)  # adjust to radar UTC

    # Filter to radar recording window
    within = lidar_adj[(lidar_adj >= t_start) & (lidar_adj <= t_end)]

    # Match each lidar timestamp to nearest radar frame; deduplicate
    seen_radar = set()
    pairs = []
    for lt in within:
        ri = int(np.argmin(np.abs(radar_ts - lt)))
        if ri not in seen_radar:
            seen_radar.add(ri)
            pairs.append((ri, float(lt)))

    return sorted(pairs, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Lidar processing
# ---------------------------------------------------------------------------

def lidar_fps_fixed(pts_xyz: np.ndarray, n_pts: int = N_LIDAR_PTS) -> np.ndarray:
    """
    Scene-filter then farthest-point-sample (or pad) a lidar point cloud.

    Parameters
    ----------
    pts_xyz : np.ndarray, shape (M, 3), any float dtype
        Raw lidar XYZ points (metres).
    n_pts : int
        Target number of output points (default 8192).

    Returns
    -------
    np.ndarray, shape (n_pts, 3), float32
    """
    # Filter to scene volume BEFORE FPS (important: FPS on pre-filtered points)
    x, y, z = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
    mask = (
        (x >= SCENE_X[0]) & (x <= SCENE_X[1]) &
        (y >= SCENE_Y[0]) & (y <= SCENE_Y[1]) &
        (z >= SCENE_Z[0]) & (z <= SCENE_Z[1])
    )
    filtered = pts_xyz[mask]

    if len(filtered) == 0:
        # Edge case: no points in scene volume — return zeros
        return np.zeros((n_pts, 3), dtype=np.float32)

    if len(filtered) >= n_pts:
        # FPS subsampling via open3d (lazy import — not available on host, only in Docker)
        import open3d as o3d  # noqa: lazy import
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered.astype(np.float64))
        fps_pcd = pcd.farthest_point_down_sample(n_pts)
        return np.asarray(fps_pcd.points, dtype=np.float32)
    else:
        # Padding: tile and truncate (24/769 frames in traj 112 need this)
        n_reps = n_pts // len(filtered) + 1
        padded = np.tile(filtered, (n_reps, 1))[:n_pts]
        return padded.astype(np.float32)


def parse_lidar_csv(csv_path: str) -> dict:
    """
    Parse a lidar CSV file into a dict mapping timestamp -> (M, 3) XYZ array.

    CSV format: no header, columns [x, y, z, intensity, timestamp].
    Timestamps are Unix seconds (float64).

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    dict mapping float timestamp -> np.ndarray shape (M, 3) float64
    """
    import pandas as pd  # noqa: lazy import — only needed for CSV parsing
    df = pd.read_csv(csv_path, header=None, names=["x", "y", "z", "intensity", "ts"])
    lidar_by_ts = {}
    for ts, group in df.groupby("ts"):
        lidar_by_ts[float(ts)] = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return lidar_by_ts


# ---------------------------------------------------------------------------
# Trajectory-level processing
# ---------------------------------------------------------------------------

def process_trajectory(
    traj_id: int,
    radar_pkl_path: str,
    lidar_csv_path: str,
    output_dir: str,
    dataset5_count: int | None = None,
) -> dict:
    """
    Full preprocessing pipeline for one trajectory.

    Steps:
      1. Load radar pickle
      2. DC offset correction (trajectory-level)
      3. Build frame alignment table (5h/4h lidar offset)
      4. Per-frame TDM-MIMO processing -> (N, 8, 512) complex64
      5. Per-frame normalization by max magnitude (store norm factors)
      6. Parse lidar CSV, FPS per aligned frame -> (N, 8192, 3) float32
      7. IQ imbalance check
      8. Save .pt files

    Parameters
    ----------
    traj_id : int
    radar_pkl_path : str
    lidar_csv_path : str
    output_dir : str
    dataset5_count : int or None
        Expected frame count from dataset_5 for validation (±10% check).

    Returns
    -------
    dict with keys: traj_id, n_frames, iq_imbalance_pct, count_mismatch
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[traj {traj_id}] Loading radar pickle: {radar_pkl_path}")
    with open(radar_pkl_path, "rb") as f:
        radar_data = pickle.load(f)

    # radar_data['frames']: (N_raw_frames, 192, 4, 512) complex128
    raw_frames = radar_data["frames"]
    print(f"[traj {traj_id}] Raw frames: {raw_frames.shape}")

    # DC offset correction (trajectory-level mean)
    print(f"[traj {traj_id}] DC offset correction...")
    corrected = dc_correct(raw_frames)  # still complex128

    # Frame alignment table
    print(f"[traj {traj_id}] Building frame alignment table...")
    frame_table = build_frame_table(traj_id, radar_data, lidar_csv_path)
    n_aligned = len(frame_table)
    print(f"[traj {traj_id}] Aligned frames: {n_aligned}")

    # Validate against dataset_5 count (±10%)
    count_mismatch = False
    if dataset5_count is not None:
        ratio = n_aligned / dataset5_count
        if not (0.9 <= ratio <= 1.1):
            print(
                f"[traj {traj_id}] WARNING: aligned count {n_aligned} vs "
                f"dataset_5 count {dataset5_count} (ratio={ratio:.3f}, outside ±10%)"
            )
            count_mismatch = True
        else:
            print(f"[traj {traj_id}] Frame count validation OK ({n_aligned}/{dataset5_count})")

    # Parse lidar CSV once (all timestamps in memory)
    print(f"[traj {traj_id}] Parsing lidar CSV: {lidar_csv_path}")
    lidar_by_ts = parse_lidar_csv(lidar_csv_path)

    # Process each aligned frame
    radar_frames_list = []
    lidar_frames_list = []
    norm_factors = []

    OFFSET_HOURS = 5 if traj_id <= 200 else 4

    for radar_idx, lidar_ts_adj in frame_table:
        # Radar: TDM-MIMO processing on DC-corrected frame
        varray = process_frame_tdm_mimo(corrected[radar_idx])  # (8, 512) complex64

        # Per-frame normalization by max magnitude
        max_mag = float(np.max(np.abs(varray)))
        if max_mag > 0:
            varray_norm = varray / max_mag
        else:
            varray_norm = varray
        radar_frames_list.append(varray_norm)
        norm_factors.append(max_mag)

        # Lidar: find closest stored timestamp (raw CSV timestamp = adj_ts + offset)
        lidar_ts_raw = lidar_ts_adj + OFFSET_HOURS * 3600
        # Find nearest key in lidar_by_ts
        if lidar_by_ts:
            ts_arr = np.array(list(lidar_by_ts.keys()))
            closest_ts = float(ts_arr[np.argmin(np.abs(ts_arr - lidar_ts_raw))])
            pts_xyz = lidar_by_ts[closest_ts]
        else:
            pts_xyz = np.zeros((0, 3), dtype=np.float64)

        lidar_fp = lidar_fps_fixed(pts_xyz, n_pts=N_LIDAR_PTS)  # (8192, 3) float32
        lidar_frames_list.append(lidar_fp)

    # Stack into tensors
    radar_array = np.stack(radar_frames_list, axis=0)  # (N, 8, 512) complex64
    lidar_array = np.stack(lidar_frames_list, axis=0)  # (N, 8192, 3) float32
    norm_array = np.array(norm_factors, dtype=np.float32)  # (N,)

    # IQ imbalance check on unnormalized virtual array
    # Re-process antenna 0 statistics on normalized data (imbalance unaffected by scale)
    iq_imbalance = check_iq_balance(radar_array, traj_id)

    # Save to .pt files
    radar_pt_path = os.path.join(output_dir, f"radar_{traj_id}.pt")
    lidar_pt_path = os.path.join(output_dir, f"lidar_{traj_id}.pt")
    norm_pt_path = os.path.join(output_dir, f"norm_{traj_id}.pt")

    import torch  # noqa: lazy import — only needed at save time
    torch.save(torch.from_numpy(radar_array), radar_pt_path)
    torch.save(torch.from_numpy(lidar_array), lidar_pt_path)
    torch.save(torch.from_numpy(norm_array), norm_pt_path)

    print(
        f"[traj {traj_id}] Saved: radar {radar_array.shape}, "
        f"lidar {lidar_array.shape}, norm {norm_array.shape}"
    )

    return {
        "traj_id": traj_id,
        "n_frames": n_aligned,
        "iq_imbalance_pct": iq_imbalance,
        "count_mismatch": count_mismatch,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RadarHD-dataset into v2 .pt tensor files."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--zip-path", type=str, help="Path to RadarHD-dataset.zip (78 GB)"
    )
    source_group.add_argument(
        "--extracted-dir", type=str, help="Path to extracted RadarHD-dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="v2/data/processed/",
        help="Output directory for .pt files (default: v2/data/processed/)",
    )
    parser.add_argument(
        "--dataset5-dir",
        type=str,
        default=None,
        help="Optional path to dataset_5/ for frame count validation",
    )
    parser.add_argument(
        "--traj-ids",
        type=str,
        default=None,
        help="Comma-separated trajectory IDs to process (default: all found)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve data source
    if args.zip_path:
        zf = zipfile.ZipFile(args.zip_path, "r")
        radar_names = [n for n in zf.namelist() if n.endswith("_read.pkl")]
        lidar_names = [n for n in zf.namelist() if n.endswith("_fwd.csv")]
        data_source = "zip"
    else:
        extracted = args.extracted_dir
        import glob as _glob
        radar_names = sorted(_glob.glob(os.path.join(extracted, "radar", "*_read.pkl")))
        lidar_names = sorted(_glob.glob(os.path.join(extracted, "lidar_pcl", "*_fwd.csv")))
        data_source = "dir"

    # Determine trajectory IDs
    if data_source == "zip":
        traj_ids_found = sorted(
            int(os.path.basename(n).replace("_read.pkl", ""))
            for n in radar_names
        )
    else:
        traj_ids_found = sorted(
            int(os.path.basename(n).replace("_read.pkl", ""))
            for n in radar_names
        )

    if args.traj_ids:
        traj_ids = [int(x) for x in args.traj_ids.split(",")]
    else:
        traj_ids = traj_ids_found

    print(f"Trajectories to process: {traj_ids}")

    # Optional dataset_5 frame counts for validation
    dataset5_counts = {}
    if args.dataset5_dir:
        import glob as _glob2
        for tid in traj_ids:
            pattern = os.path.join(args.dataset5_dir, "**", f"*_{tid}_*.png")
            matches = _glob2.glob(pattern, recursive=True)
            if matches:
                dataset5_counts[tid] = len(matches)

    # Process each trajectory
    results = []
    for tid in traj_ids:
        if data_source == "zip":
            radar_pkl_path = f"radar/{tid}_read.pkl"
            lidar_csv_path = f"lidar_pcl/{tid}_fwd.csv"
            # Extract pkl to temp file
            import tempfile, shutil
            with zf.open(radar_pkl_path) as zf_pkl:
                tmp_pkl = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                shutil.copyfileobj(zf_pkl, tmp_pkl)
                tmp_pkl.close()
                radar_path = tmp_pkl.name
            with zf.open(lidar_csv_path) as zf_csv:
                tmp_csv = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode="wb"
                )
                shutil.copyfileobj(zf_csv, tmp_csv)
                tmp_csv.close()
                lidar_path = tmp_csv.name
        else:
            radar_path = os.path.join(extracted, "radar", f"{tid}_read.pkl")
            lidar_path = os.path.join(extracted, "lidar_pcl", f"{tid}_fwd.csv")

        ds5_count = dataset5_counts.get(tid, None)
        result = process_trajectory(
            traj_id=tid,
            radar_pkl_path=radar_path,
            lidar_csv_path=lidar_path,
            output_dir=args.output_dir,
            dataset5_count=ds5_count,
        )
        results.append(result)

        if data_source == "zip":
            os.unlink(radar_path)
            os.unlink(lidar_path)

    if data_source == "zip":
        zf.close()

    # Save frame table JSON
    frame_table_path = os.path.join(args.output_dir, "frame_table.json")
    with open(frame_table_path, "w") as ft:
        json.dump(results, ft, indent=2)
    print(f"\nFrame table saved: {frame_table_path}")

    # Print summary
    total_frames = sum(r["n_frames"] for r in results)
    iq_warnings = [r for r in results if r["iq_imbalance_pct"] >= 1.0]
    count_mismatches = [r for r in results if r["count_mismatch"]]

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  Trajectories processed: {len(results)}")
    print(f"  Total frames: {total_frames}")
    print(f"  IQ warnings (>=1%): {len(iq_warnings)}")
    for r in iq_warnings:
        print(f"    traj {r['traj_id']}: {r['iq_imbalance_pct']:.1f}%")
    print(f"  Count mismatches: {len(count_mismatches)}")
    for r in count_mismatches:
        print(f"    traj {r['traj_id']}: {r['n_frames']} frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
