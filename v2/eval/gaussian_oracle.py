"""Gaussian oracle test: can Gaussian representation preserve mod-H?

Fit K Gaussians to lidar GT, decode back to points, measure metrics.
No neural network — just representation fidelity.

If the oracle can't beat baseline mod-H (0.189), kill the Gaussian direction.

Run inside Docker:
  docker compose run --rm mmdar python3 v2/eval/gaussian_oracle.py
"""

import sys, os, time, json
import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from v2.data.split import TEST_TRAJS
from eval.eval_pointcloud import chamfer_distance, modified_hausdorff


def fit_gaussians_2d(points: np.ndarray, K: int) -> tuple:
    """Fit K 2D Gaussians to a point cloud via K-Means + per-cluster covariance.

    Args:
        points: (N, 2) float64
        K: number of Gaussians

    Returns:
        centers: (K, 2) — Gaussian centers
        covariances: (K, 2, 2) — per-Gaussian covariance matrices
        assignments: (N,) — cluster assignment per point
    """
    if len(points) < K:
        K = max(1, len(points))

    kmeans = KMeans(n_clusters=K, n_init=3, random_state=0, max_iter=100)
    assignments = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_  # (K, 2)

    covariances = np.zeros((K, 2, 2))
    for k in range(K):
        mask = assignments == k
        if mask.sum() < 2:
            covariances[k] = np.eye(2) * 0.01  # degenerate: small sphere
        else:
            covariances[k] = np.cov(points[mask], rowvar=False)
            # Regularize to prevent singular covariance
            covariances[k] += np.eye(2) * 1e-4

    return centers, covariances, assignments


def decode_gaussians_centers(centers: np.ndarray) -> np.ndarray:
    """Decode: just use Gaussian centers as point cloud."""
    return centers


def decode_gaussians_sample(centers: np.ndarray, covariances: np.ndarray,
                            n_per_gaussian: int = 8) -> np.ndarray:
    """Decode: sample n points from each Gaussian."""
    points = []
    for k in range(len(centers)):
        samples = np.random.multivariate_normal(centers[k], covariances[k],
                                                 size=n_per_gaussian)
        points.append(samples)
    return np.vstack(points)


def run_oracle():
    processed_dir = 'v2/data/processed'
    results_by_k = {}

    # Test on a range of K values
    K_values = [32, 48, 64, 96, 128, 256]
    # Decode methods
    decode_methods = ['centers', 'sample_4', 'sample_8', 'sample_16']

    print(f'Gaussian oracle test on {len(TEST_TRAJS)} test trajectories')
    print(f'K values: {K_values}')

    for K in K_values:
        print(f'\n=== K={K} ===')
        metrics = {m: {'chamfer': [], 'mod_h': []} for m in decode_methods}
        n_samples = 0
        t0 = time.time()

        for tid in TEST_TRAJS:
            lidar_path = os.path.join(processed_dir, f'lidar_{tid}.pt')
            if not os.path.exists(lidar_path):
                continue

            lidar = torch.load(lidar_path, weights_only=True).numpy()  # (N, 8192, 3)
            N = lidar.shape[0]

            for i in range(N):
                gt_xyz = lidar[i]  # (8192, 3)
                gt_xy = gt_xyz[:, :2].astype(np.float64)  # (8192, 2)

                # Filter to valid range (x > 0, within scene bounds)
                mask = (gt_xy[:, 0] > 0) & (gt_xy[:, 0] <= 10.8) & \
                       (np.abs(gt_xy[:, 1]) <= 10.8)
                gt_valid = gt_xy[mask]

                if len(gt_valid) < 3:
                    continue

                # Fit Gaussians
                centers, covs, _ = fit_gaussians_2d(gt_valid, K)

                # Decode and evaluate each method
                for method in decode_methods:
                    if method == 'centers':
                        pred = decode_gaussians_centers(centers)
                    elif method.startswith('sample_'):
                        n_per = int(method.split('_')[1])
                        np.random.seed(0)
                        pred = decode_gaussians_sample(centers, covs, n_per)
                    else:
                        continue

                    if len(pred) < 2 or len(gt_valid) < 2:
                        continue

                    cd = chamfer_distance(pred, gt_valid)
                    mh = modified_hausdorff(pred, gt_valid)
                    metrics[method]['chamfer'].append(cd)
                    metrics[method]['mod_h'].append(mh)

                n_samples += 1
                if n_samples % 2000 == 0:
                    elapsed = time.time() - t0
                    print(f'  [{n_samples}] {elapsed:.0f}s')

        elapsed = time.time() - t0
        print(f'  Done: {n_samples} samples in {elapsed:.0f}s')

        results_by_k[K] = {}
        for method in decode_methods:
            if metrics[method]['chamfer']:
                cd_mean = float(np.mean(metrics[method]['chamfer']))
                mh_mean = float(np.mean(metrics[method]['mod_h']))
                cd_med = float(np.median(metrics[method]['chamfer']))
                mh_med = float(np.median(metrics[method]['mod_h']))
                results_by_k[K][method] = {
                    'chamfer_mean': cd_mean, 'mod_h_mean': mh_mean,
                    'chamfer_median': cd_med, 'mod_h_median': mh_med,
                    'n_samples': len(metrics[method]['chamfer']),
                }
                print(f'  {method:>12}: Chamfer {cd_mean:.4f} (med {cd_med:.4f}), '
                      f'mod-H {mh_mean:.4f} (med {mh_med:.4f})')

    # Summary table
    print('\n' + '=' * 80)
    print('GAUSSIAN ORACLE RESULTS (mean)')
    print('=' * 80)
    print(f'{"K":>5} {"Method":>12} {"Chamfer":>10} {"Mod-H":>10}   vs baseline 0.295/0.189')
    print('-' * 60)
    for K in K_values:
        for method in decode_methods:
            r = results_by_k.get(K, {}).get(method, {})
            if r:
                print(f'{K:>5} {method:>12} {r["chamfer_mean"]:>10.4f} {r["mod_h_mean"]:>10.4f}')

    # Save
    os.makedirs('results/gaussian_oracle/', exist_ok=True)
    with open('results/gaussian_oracle/report.json', 'w') as f:
        json.dump(results_by_k, f, indent=2)
    print(f'\nSaved: results/gaussian_oracle/report.json')


if __name__ == '__main__':
    run_oracle()
