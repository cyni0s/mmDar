"""
v2 Data Split Definitions
=========================

Trajectory-level train/validation/test split for the mmDar v2 pipeline.

Split rationale:
- TEST_TRAJS (19 trajectories): Sealed at the same 19 trajectories used for the
  0.295m Chamfer baseline, ensuring fair comparison. NEVER change.
- TRAIN_POOL (25 trajectories): All non-test trajectories.
- VAL_TRAJS (4 trajectories): Last 4 by ID carved out of TRAIN_POOL for validation.
  Chosen by ID ordering (highest IDs in pool) to mimic temporal/collection ordering.
- TRAIN_TRAJS (21 trajectories): TRAIN_POOL minus VAL_TRAJS.

Total: 19 + 4 + 21 = 44 trajectories.

Assumptions:
- Trajectory-level splitting avoids data leakage (frames from the same trajectory
  cannot appear in both train and val/test).
- The 19-trajectory test set is frozen; any future data addition must not touch it.
"""

# ---------------------------------------------------------------------------
# Split constants
# ---------------------------------------------------------------------------

# Test trajectories (sealed, 19 trajs) — matches RadarHD baseline evaluation set
TEST_TRAJS = [
    117, 124, 132, 139,
    227, 229, 230, 232, 233, 236, 237, 238,
    242, 245, 246, 247, 248, 249, 250,
]

# Validation trajectories (4 trajs carved from training pool, highest IDs by order)
VAL_TRAJS = [136, 137, 138, 140]

# Training trajectories (21 trajs — training pool minus validation)
TRAIN_TRAJS = [
    112, 113, 114, 115, 116,
    118, 119, 120, 121, 122, 123,
    125, 126, 127, 128, 129, 130, 131,
    133, 134, 135,
]

# All 44 trajectory IDs in the dataset (for completeness check)
ALL_TRAJS = sorted(set(TRAIN_TRAJS) | set(VAL_TRAJS) | set(TEST_TRAJS))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_split() -> None:
    """
    Assert correctness of the split definitions.

    Checks:
      1. No overlap between any pair of split sets.
      2. Union of all three equals exactly 44 trajectory IDs.
      3. TEST_TRAJS has exactly 19 elements.
      4. TRAIN_TRAJS has exactly 21 elements.
      5. VAL_TRAJS has exactly 4 elements.

    If v2/data/processed/ exists, also prints per-split frame counts
    by reading the saved .pt file shapes.
    """
    # Disjointness
    assert set(TRAIN_TRAJS).isdisjoint(set(VAL_TRAJS)), \
        "TRAIN and VAL overlap!"
    assert set(TRAIN_TRAJS).isdisjoint(set(TEST_TRAJS)), \
        "TRAIN and TEST overlap!"
    assert set(VAL_TRAJS).isdisjoint(set(TEST_TRAJS)), \
        "VAL and TEST overlap!"

    # Completeness (44 total trajectories in the dataset)
    all_union = set(TRAIN_TRAJS) | set(VAL_TRAJS) | set(TEST_TRAJS)
    assert len(all_union) == 44, \
        f"Expected 44 unique trajectory IDs, got {len(all_union)}"

    # Size checks
    assert len(TEST_TRAJS) == 19, \
        f"TEST_TRAJS must have exactly 19 elements, got {len(TEST_TRAJS)}"
    assert len(VAL_TRAJS) == 4, \
        f"VAL_TRAJS must have exactly 4 elements, got {len(VAL_TRAJS)}"
    assert len(TRAIN_TRAJS) == 21, \
        f"TRAIN_TRAJS must have exactly 21 elements, got {len(TRAIN_TRAJS)}"

    print("Split validation PASSED.")
    print(f"  Train: {len(TRAIN_TRAJS)} trajectories")
    print(f"  Val:   {len(VAL_TRAJS)} trajectories")
    print(f"  Test:  {len(TEST_TRAJS)} trajectories (sealed)")

    # Optional: count frames if processed dir exists
    import os
    processed_dir = os.path.join(os.path.dirname(__file__), "processed")
    if os.path.isdir(processed_dir):
        import torch
        split_map = {
            "train": TRAIN_TRAJS,
            "val": VAL_TRAJS,
            "test": TEST_TRAJS,
        }
        for split_name, traj_list in split_map.items():
            total = 0
            for tid in traj_list:
                pt_path = os.path.join(processed_dir, f"radar_{tid}.pt")
                if os.path.isfile(pt_path):
                    t = torch.load(pt_path, weights_only=True)
                    total += t.shape[0]
            print(f"  {split_name}: {total} frames (from processed/)")


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------

def get_split(split_name: str) -> list:
    """
    Return trajectory ID list for a named split.

    Parameters
    ----------
    split_name : str
        One of "train", "val", or "test".

    Returns
    -------
    list of int
        Trajectory IDs for the requested split.

    Raises
    ------
    ValueError
        If split_name is not one of the accepted values.
    """
    mapping = {
        "train": TRAIN_TRAJS,
        "val": VAL_TRAJS,
        "test": TEST_TRAJS,
    }
    if split_name not in mapping:
        raise ValueError(
            f"Unknown split '{split_name}'. Choose from: {list(mapping.keys())}"
        )
    return mapping[split_name]


if __name__ == "__main__":
    validate_split()
