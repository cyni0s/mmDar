"""
v2 Data Split — Expanded Validation
====================================

Expands val from 4 to 8 trajectories for more reliable model selection.
Test set remains sealed (19 trajectories, same as baseline).

Split rationale:
- TEST_TRAJS (19): sealed, never change. Mix of low-ID (easy) and high-ID (hard).
- VAL_TRAJS (8): expanded from 4 to 8. Spread across ID range for better coverage.
- TRAIN_TRAJS (17): remaining non-test trajectories.

NOTE: all train+val trajectories are IDs 112-140 (low-ID).
      High-ID trajectories (227-250) are ALL in test.
      Data augmentation is needed to bridge this distribution gap.
"""

# Test trajectories (sealed, 19 trajs) — NEVER CHANGE
TEST_TRAJS = [
    117, 124, 132, 139,
    227, 229, 230, 232, 233, 236, 237, 238,
    242, 245, 246, 247, 248, 249, 250,
]

# Validation trajectories (8 trajs — spread across range)
VAL_TRAJS = [113, 119, 125, 131, 135, 136, 138, 140]

# Training trajectories (17 trajs — remaining)
TRAIN_TRAJS = [
    112, 114, 115, 116,
    118, 120, 121, 122, 123,
    126, 127, 128, 129, 130,
    133, 134, 137,
]

ALL_TRAJS = sorted(set(TRAIN_TRAJS) | set(VAL_TRAJS) | set(TEST_TRAJS))

# Validation
assert len(TEST_TRAJS) == 19
assert len(VAL_TRAJS) == 8
assert len(TRAIN_TRAJS) == 17
assert len(ALL_TRAJS) == 44
assert set(TRAIN_TRAJS).isdisjoint(set(VAL_TRAJS))
assert set(TRAIN_TRAJS).isdisjoint(set(TEST_TRAJS))
assert set(VAL_TRAJS).isdisjoint(set(TEST_TRAJS))
