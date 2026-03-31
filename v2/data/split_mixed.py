"""
Mixed-ID Split — High-ID trajectories in train+val for capacity test.

PURPOSE: Prove whether the architecture can learn high-ID environments
when given the data. NOT for final publication metrics.

Takes 6 high-ID trajectories from the sealed test set into train+val.
Remaining 13 high-ID + 4 low-ID stay in test for evaluation.

Split: 21 train (mixed), 6 val (mixed), 17 test (mixed)
"""

# High-ID trajectories moved to train/val (6 of 15)
# Pick a spread: 2 for val, 4 for train
HIGH_ID_TO_TRAIN = [229, 233, 238, 246]
HIGH_ID_TO_VAL = [230, 242]

# Remaining test: 13 high-ID + 4 low-ID = 17
TEST_TRAJS = [
    117, 124, 132, 139,                          # low-ID (easy)
    227, 232, 236, 237, 245, 247, 248, 249, 250, # high-ID (hard)
]

# Val: 4 low-ID + 2 high-ID = 6
VAL_TRAJS = [119, 131, 136, 140] + HIGH_ID_TO_VAL  # [119, 131, 136, 140, 230, 242]

# Train: 17 low-ID + 4 high-ID = 21
TRAIN_TRAJS = [
    112, 113, 114, 115, 116,
    118, 120, 121, 122, 123,
    125, 126, 127, 128, 129, 130,
    133, 134, 135, 137, 138,
] + HIGH_ID_TO_TRAIN  # + [229, 233, 238, 246]

# Remove val trajs from train
TRAIN_TRAJS = [t for t in TRAIN_TRAJS if t not in VAL_TRAJS]

ALL_TRAJS = sorted(set(TRAIN_TRAJS) | set(VAL_TRAJS) | set(TEST_TRAJS))

# Validation
assert len(set(TRAIN_TRAJS) & set(VAL_TRAJS)) == 0, "Train/val overlap"
assert len(set(TRAIN_TRAJS) & set(TEST_TRAJS)) == 0, "Train/test overlap"
assert len(set(VAL_TRAJS) & set(TEST_TRAJS)) == 0, "Val/test overlap"
assert len(ALL_TRAJS) == 44, f"Expected 44, got {len(ALL_TRAJS)}"

print(f"Train: {len(TRAIN_TRAJS)} trajs ({sorted(TRAIN_TRAJS)})")
print(f"Val:   {len(VAL_TRAJS)} trajs ({sorted(VAL_TRAJS)})")
print(f"Test:  {len(TEST_TRAJS)} trajs ({sorted(TEST_TRAJS)})")
print(f"High-ID in train: {HIGH_ID_TO_TRAIN}")
print(f"High-ID in val:   {HIGH_ID_TO_VAL}")
print(f"High-ID in test:  {[t for t in TEST_TRAJS if t > 200]}")
