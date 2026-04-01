# Codebase Cleanup & Consolidation

## Goal

Strip the repo to two clean codepaths (baseline + physics-first Gaussian), consolidate training scripts, add thorough documentation. Preserve READMEs as experiment logs.

## What KEEPS

### Baseline (move to `baseline/`)
- `train_test_utils/model.py` (UNet1)
- `train_test_utils/unet_parts.py` (DoubleConv, Down, Up, etc.)
- `train_test_utils/dataloader.py` (PNG Dataset)
- `train_test_utils/dice_score.py`
- `train_radarhd.py` (training script)
- `test_radarhd.py` (inference script)
- `v2/train/train_baseline_honest.py` → move to `baseline/train_honest.py`

### Shared eval (stays in `eval/`)
- `eval/eval_pointcloud.py`
- `eval/pol_to_cart.py`
- `eval/image_to_pcd.py`

### Our model (stays in `v2/`)

**Model (core chain):**
- `v2/model/physics_frontend.py` → PhysicsGaussianModel, PhysicsFirstEncoder, ClassicalFFTFrontend, Deep2DEncoder
- `v2/model/gaussian_head.py` → GaussianSetDecoder, DecoderLayer
- `v2/model/beamspace.py` → LearnedBeamspace, DilatedResBlock1d (used by physics_frontend)
- `v2/model/lista.py` → FFTBeamformer, build_steering_matrix (used by beamspace)
- `v2/model/cvnn.py` → complex_soft_threshold, ComplexConv1d, etc. (used by lista)

**Data:**
- `v2/data/preprocess.py` → raw IQ preprocessing
- `v2/data/windowed_dataset.py` → WindowedTrajectoryDataset, build_windowed_dataloaders
- `v2/data/augment.py` → horizontal_flip, add_complex_noise, temporal_mask
- `v2/data/split.py` → original 21/4/19 split
- `v2/data/split_v2.py` → expanded 17/8/19 split
- `v2/data/split_mixed.py` → mixed-ID 25/6/13 split

**Training — CONSOLIDATE into ONE script:**
- `v2/train/train.py` (NEW — consolidated from train_mixed_split + train_physics_augmented + train_gaussian_radar)
- Flags: `--split original/v2/mixed`, `--augment`, `--window-size`, `--K`, `--N-az`
- Contains: AugmentedGaussianDataset, GaussianDataset, train_epoch, eval_per_trajectory, eval_points, fit_prototypes (GPU K-Means)
- `v2/train/loss_gaussian.py` → Hungarian NLL + coverage + cardinality + repulsion

**Eval:**
- `v2/eval/eval_adapter.py` → GPU-accelerated Chamfer/mod-H
- `v2/eval/fps.py` → farthest point sampling
- `v2/eval/gaussian_oracle.py` → representation ceiling test

**Tests (for kept code only):**
- `v2/eval/tests/test_fps.py`
- New integration test: smoke test for PhysicsGaussianModel forward + backward

## What GOES (delete — recoverable from git history)

### Root scripts (dead)
- `run_experiment.py`, `sweep_checkpoints.py`, `test_convlstm.py`, `train_convlstm.py`, `train_1d.py`, `auto_launch_baseline.sh`

### v2/model (failed experiments)
- `decoder.py` (point cloud decoder — Phases 3-5)
- `decoder_2d.py` (2D angular fix — Phase 3)
- `occupancy.py` (75K occupancy — Phase 3)
- `temporal.py` (cross-attention — Phase 4)
- `unet_occupancy.py` (U-Net occupancy — Phase 8a)

### v2/train (superseded)
- `loss.py` (Chamfer composite)
- `loss_physics.py` (physics losses — Phase 5)
- `loss_occupancy.py` (occupancy loss)
- `train_temporal.py`, `train_occupancy.py`, `train_occupancy_unet.py`
- `train_baseline_focal.py`, `train_fewer_points.py`
- `train_physics_gaussian.py` (superseded by consolidated train.py)
- `train_gaussian_radar.py` (helpers moved to consolidated train.py)
- `train_physics_augmented.py` (merged into consolidated train.py)
- `train_mixed_split.py` (merged into consolidated train.py)

### v2/data (dead)
- `dataset.py` (old v2 dataset)
- `lista_dataset.py` (Phase 8a)
- `preprocess_lista.py` (Phase 8a)
- `rasterize.py` (used only by Phase 8a)

### v2/eval (completed experiments)
- `standardize_eval.py` (Phase 6)
- `gt_standardize.py` (Phase 7)
- `occupancy_eval.py`, `occupancy_to_pc.py` (Phase 8a)

### Tests for removed code
- All test files for decoder, decoder_2d, occupancy, temporal, loss, loss_physics, loss_occupancy, occupancy_eval, lista_dataset, etc.

## What gets DOCUMENTED

Every kept .py file gets:
1. **Module docstring**: purpose, relationship to pipeline, inputs/outputs
2. **Class/function docstrings**: args, returns, physics notes
3. **Inline comments**: non-obvious logic, coordinate conventions, why not how

## What does NOT change
- `README.md` — project overview + lessons learned
- `results/README.md` — full experiment log (all 12+ phases)
- `docs/superpowers/specs/` — design documents
- `docs/superpowers/plans/` — implementation plans
- `.planning/` — GSD planning files
- `paper/` — reference papers

## Verification after cleanup
1. `docker compose run --rm mmdar python3 -c "from v2.model.physics_frontend import PhysicsGaussianModel; print('OK')"` — model imports
2. `docker compose run --rm mmdar python3 -c "from train_test_utils.model import UNet1; print('OK')"` — baseline imports (from baseline/ subfolder)
3. `docker compose run --rm mmdar python3 -m pytest v2/eval/tests/test_fps.py -v` — FPS tests
4. Smoke test: PhysicsGaussianModel forward + backward on one batch
5. Verify `v2/train/train.py --help` shows all flags

## File count
- Before: ~75 Python files
- After: ~25 Python files (+ baseline subfolder)
