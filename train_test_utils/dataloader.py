# Creates a dataloader in batches for training and testing

import os
from PIL import Image
import numpy as np
import torch
import glob

class Dataset(torch.utils.data.Dataset):

    def __init__(self, basepath, sub,
                RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
                RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=1024, M=0):

        self.basepath = basepath
        self.lidar_path = self.basepath + sub + '/lidar/*'
        self.radar_path = self.basepath + sub + '/radar/*'

        self.RBINS = RBINS
        self.ABINS_RADAR = ABINS_RADAR
        self.ABINS_LIDAR = ABINS_LIDAR
        self.RBINS_ORIG = RBINS_ORIG
        self.ABINS_RADAR_ORIG = ABINS_RADAR_ORIG
        self.ABINS_LIDAR_ORIG = ABINS_LIDAR_ORIG
        self.history = M

        lidar_files = sorted(glob.glob(self.lidar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        radar_files = sorted(glob.glob(self.radar_path), key=lambda x: (int(os.path.basename(x).split('_')[1]), int(os.path.basename(x).split('_')[2].split('.')[0])))
        
        if self.history == 0:
            self.labels = lidar_files
            self.input_data = radar_files
        else:
            traj = [int(os.path.basename(x).split('_')[1]) for x in lidar_files]
            time_st = [int(os.path.basename(x).split('_')[2].split('.')[0]) for x in lidar_files]
            self.labels = []
            self.input_data = []

            for i in np.unique(traj):
                start_idx = np.where(traj==i)[0][0]
                end_idx = np.where(traj==i)[0][-1]+1
                print("Traj ", i, "Time ", time_st[start_idx], " ", time_st[end_idx-1])
                radar_files_time = radar_files[start_idx:end_idx]
                lidar_files_time = lidar_files[start_idx:end_idx]

                x_local = []
                for j in range(self.history, len(radar_files_time)):
                    x_local.append(radar_files_time[j-self.history:j+1])
                y_local = lidar_files_time[self.history:]
                
                self.labels.extend(y_local)
                self.input_data.extend(x_local)

    def __len__(self):
        return len(self.input_data)

    def __filenames__(self):
        return [os.path.basename(x).split('.')[0].split('L_')[1] for x in self.labels]

    def get_lidar(self, label_filename):

        a = Image.open(label_filename)
        y = torch.Tensor(np.reshape(np.asarray(a,dtype=np.bool_), (1,self.RBINS_ORIG,self.ABINS_LIDAR_ORIG)))
        y = y[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_LIDAR_ORIG/self.ABINS_LIDAR)]

        return y

    def get_radar(self, input_filename):

        a = Image.open(input_filename)
        X = torch.Tensor(np.reshape(np.asarray(a)/255.0, (1,self.RBINS_ORIG,self.ABINS_RADAR_ORIG)))
        X = X[:,0::int(self.RBINS_ORIG/self.RBINS),0::int(self.ABINS_RADAR_ORIG/self.ABINS_RADAR)]

        return X

    def __getitem__(self, index):

        # Select sample
        if self.history == 0:
            input_filename = self.input_data[index]
            label_filename = self.labels[index]
            X, y = self.get_radar(input_filename), self.get_lidar(label_filename)

        else:
            X = torch.Tensor([])
            input_filenames = self.input_data[index]
            label_filename = self.labels[index]
            for i in input_filenames:
                xx = self.get_radar(i)
                X = torch.cat((X, xx), dim=0)
            y = self.get_lidar(label_filename)

        return X, y


# ---------------------------------------------------------------------------
# Sequential data loading for ConvLSTM training (Phase 2)
# ---------------------------------------------------------------------------
# NOTE: The Dataset class above is completely unchanged.
# The following classes add trajectory-sequential access for ConvLSTM.
# ---------------------------------------------------------------------------

def _traj_sort_key(path):
    """Sorting key: (traj_id, frame_idx) extracted from R_TID_FID.png or L_TID_FID.png."""
    base = os.path.basename(path)
    parts = base.split('_')
    return (int(parts[1]), int(parts[2].split('.')[0]))


class SequentialDataset(torch.utils.data.Dataset):
    """Trajectory-sequential dataset for ConvLSTM training.

    Each sample is a contiguous clip of T frames from a single trajectory.
    Items are indexed by (traj_id, target_frame_idx, T) tuples, where
    target_frame_idx is the LOCAL index within the trajectory's file list.

    Eligible targets: frames with >= M preceding frames in the same trajectory
    (identical pool to Dataset(M=40) when M=40).

    Args:
        basepath: path to dataset root (e.g. 'dataset_5/')
        sub: split subdirectory ('train' or 'test')
        M: minimum number of preceding frames required (default 40)
        RBINS, ABINS_RADAR, ABINS_LIDAR: output spatial resolution
        RBINS_ORIG, ABINS_RADAR_ORIG, ABINS_LIDAR_ORIG: source PNG resolution
        exclude_traj_ids: list of trajectory IDs to exclude (training mode)
        include_traj_ids: if set, load ONLY these trajectory IDs (val mode)
    """

    def __init__(self, basepath, sub, M=40,
                 RBINS=256, ABINS_RADAR=64, ABINS_LIDAR=512,
                 RBINS_ORIG=256, ABINS_RADAR_ORIG=64, ABINS_LIDAR_ORIG=1024,
                 exclude_traj_ids=None, include_traj_ids=None):

        self.basepath = basepath
        self.sub = sub
        self.M = M

        self.RBINS = RBINS
        self.ABINS_RADAR = ABINS_RADAR
        self.ABINS_LIDAR = ABINS_LIDAR
        self.RBINS_ORIG = RBINS_ORIG
        self.ABINS_RADAR_ORIG = ABINS_RADAR_ORIG
        self.ABINS_LIDAR_ORIG = ABINS_LIDAR_ORIG

        radar_path = os.path.join(basepath, sub, 'radar', '*')
        lidar_path = os.path.join(basepath, sub, 'lidar', '*')

        radar_files = sorted(glob.glob(radar_path), key=_traj_sort_key)
        lidar_files = sorted(glob.glob(lidar_path), key=_traj_sort_key)

        # Group files by trajectory ID
        # traj_data[traj_id] = {'radar': [paths], 'lidar': [paths], 'eligible_targets': [local_idxs]}
        self.traj_data = {}
        for rf, lf in zip(radar_files, lidar_files):
            tid = int(os.path.basename(rf).split('_')[1])
            if tid not in self.traj_data:
                self.traj_data[tid] = {'radar': [], 'lidar': [], 'eligible_targets': []}
            self.traj_data[tid]['radar'].append(rf)
            self.traj_data[tid]['lidar'].append(lf)

        # Compute eligible target frames (those with >= M preceding frames)
        for tid, data in self.traj_data.items():
            n_frames = len(data['radar'])
            data['eligible_targets'] = list(range(M, n_frames))

        # Apply trajectory filter
        if include_traj_ids is not None:
            include_set = set(include_traj_ids)
            self.traj_data = {tid: data for tid, data in self.traj_data.items()
                              if tid in include_set}
        elif exclude_traj_ids is not None:
            exclude_set = set(exclude_traj_ids)
            self.traj_data = {tid: data for tid, data in self.traj_data.items()
                              if tid not in exclude_set}

        # Flat list of (traj_id, local_target_idx) for all eligible targets
        self.eligible_targets = []
        for tid in sorted(self.traj_data.keys()):
            for local_idx in self.traj_data[tid]['eligible_targets']:
                self.eligible_targets.append((tid, local_idx))

    def __len__(self):
        return len(self.eligible_targets)

    def get_radar(self, path):
        """Load a single radar frame. Returns (1, RBINS, ABINS_RADAR) tensor."""
        a = Image.open(path)
        X = torch.Tensor(np.reshape(np.asarray(a) / 255.0,
                                    (1, self.RBINS_ORIG, self.ABINS_RADAR_ORIG)))
        X = X[:, 0::int(self.RBINS_ORIG / self.RBINS),
                 0::int(self.ABINS_RADAR_ORIG / self.ABINS_RADAR)]
        return X

    def get_lidar(self, path):
        """Load a single lidar frame. Returns (1, RBINS, ABINS_LIDAR) tensor."""
        a = Image.open(path)
        y = torch.Tensor(np.reshape(np.asarray(a, dtype=np.bool_),
                                    (1, self.RBINS_ORIG, self.ABINS_LIDAR_ORIG)))
        y = y[:, 0::int(self.RBINS_ORIG / self.RBINS),
                 0::int(self.ABINS_LIDAR_ORIG / self.ABINS_LIDAR)]
        return y

    def __getitem__(self, item):
        """Load a T-frame clip ending at target_frame_idx.

        Args:
            item: tuple (traj_id, target_frame_idx, T)
                  target_frame_idx is the LOCAL index within the trajectory.
                  T is the clip length (number of frames to load).

        Returns:
            radar_seq: (T, 1, RBINS, ABINS_RADAR) float32 tensor
            lidar_seq: (T, 1, RBINS, ABINS_LIDAR) float32 tensor
            meta: dict with 'traj_id' and 'target_idx'
        """
        traj_id, target_frame_idx, T = item
        data = self.traj_data[traj_id]

        # Slice: last T frames of the window ending at target_frame_idx (inclusive)
        start = target_frame_idx - T + 1
        end = target_frame_idx + 1  # exclusive

        radar_paths = data['radar'][start:end]
        lidar_paths = data['lidar'][start:end]

        radar_frames = [self.get_radar(p) for p in radar_paths]
        lidar_frames = [self.get_lidar(p) for p in lidar_paths]

        # Stack to (T, 1, H, W)
        radar_seq = torch.stack(radar_frames, dim=0)
        lidar_seq = torch.stack(lidar_frames, dim=0)

        meta = {'traj_id': traj_id, 'target_idx': target_frame_idx}
        return radar_seq, lidar_seq, meta


class TrajectoryBatchSampler:
    """Pre-computes all batches for an epoch. Stateless after __init__ — safe with num_workers>0.

    Strategy (option b from RESEARCH.md): at set_epoch / init, pre-compute the full
    epoch schedule as a list of batch tuples. Each batch has batch_size trajectory
    slots, advanced sequentially through their eligible targets. When a slot exhausts
    its trajectory's targets, the next trajectory starts in that slot.

    For variable_t: T ~ Uniform(1, max_T) is sampled once per batch; all slots in
    a batch use the same T.

    Args:
        dataset: SequentialDataset instance
        batch_size: number of trajectory slots per batch (N)
        max_T: maximum sequence length (default 41)
        variable_t: if True, sample T per batch from Uniform(1, max_T) inclusive
        seed: base random seed; epoch seed = seed + epoch
    """

    def __init__(self, dataset, batch_size=4, max_T=41, variable_t=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_T = max_T
        self.variable_t = variable_t
        self.seed = seed
        self._current_epoch = 0
        self._batches = None
        self._build_epoch(epoch=0)

    def _build_epoch(self, epoch):
        """Pre-compute all batches for the given epoch."""
        rng = np.random.RandomState(self.seed + epoch)

        # Shuffle trajectory IDs for this epoch
        traj_ids = sorted(self.dataset.traj_data.keys())
        rng.shuffle(traj_ids)

        # Build flat list of eligible targets per trajectory in epoch order
        # Each trajectory contributes its eligible targets in sequential order
        traj_queues = []
        for tid in traj_ids:
            targets = self.dataset.traj_data[tid]['eligible_targets']
            traj_queues.append(list(targets))  # local indices, sequential

        # Fill batch_size slots by cycling through queues
        # Assign trajectories to slots in round-robin; advance in lockstep
        # When a slot's trajectory ends, pop the next available trajectory
        available = list(traj_queues)  # list of per-trajectory target lists
        traj_order = list(traj_ids)    # matching traj_ids for each queue

        batches = []
        # Work through all targets in a single pass.
        # We maintain batch_size active "slots", each draining one trajectory.
        # When a slot empties, we pull the next trajectory from the remaining pool.

        # Pull up to batch_size initial trajectories into slots
        slot_traj_ids = []
        slot_queues = []
        slot_pos = []  # current position within each slot's target list

        # Initialize slots
        src_idx = 0  # next trajectory index to pull from traj_order

        def _pull_next_traj():
            nonlocal src_idx
            if src_idx < len(traj_order):
                tid = traj_order[src_idx]
                q = available[src_idx]
                src_idx += 1
                return tid, q
            return None, None

        for _ in range(self.batch_size):
            tid, q = _pull_next_traj()
            if tid is not None:
                slot_traj_ids.append(tid)
                slot_queues.append(q)
                slot_pos.append(0)
            else:
                slot_traj_ids.append(None)
                slot_queues.append([])
                slot_pos.append(0)

        # Generate batches until all slots are exhausted
        while True:
            # Check if any slot has remaining targets
            active_slots = [i for i in range(self.batch_size)
                            if slot_traj_ids[i] is not None and slot_pos[i] < len(slot_queues[i])]
            if not active_slots:
                break

            # Sample T for this batch
            if self.variable_t:
                T = int(rng.randint(1, self.max_T + 1))
            else:
                T = self.max_T

            batch = []
            for slot in range(self.batch_size):
                if slot_traj_ids[slot] is None or slot_pos[slot] >= len(slot_queues[slot]):
                    # Slot is exhausted — try to pull next trajectory
                    tid, q = _pull_next_traj()
                    if tid is None:
                        # No more trajectories; this slot is permanently done
                        continue
                    slot_traj_ids[slot] = tid
                    slot_queues[slot] = q
                    slot_pos[slot] = 0

                if slot_pos[slot] < len(slot_queues[slot]):
                    local_idx = slot_queues[slot][slot_pos[slot]]
                    batch.append((slot_traj_ids[slot], local_idx, T))
                    slot_pos[slot] += 1

            if batch:
                batches.append(batch)

            # Advance any exhausted slots immediately for next iteration check
            for slot in range(self.batch_size):
                if (slot_traj_ids[slot] is not None and
                        slot_pos[slot] >= len(slot_queues[slot])):
                    tid, q = _pull_next_traj()
                    if tid is not None:
                        slot_traj_ids[slot] = tid
                        slot_queues[slot] = q
                        slot_pos[slot] = 0
                    else:
                        slot_traj_ids[slot] = None

        self._batches = batches

    def set_epoch(self, epoch):
        """Rebuild epoch schedule with new random seed for trajectory shuffling."""
        self._current_epoch = epoch
        self._build_epoch(epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def seq_collate_fn(batch):
    """Collate function for SequentialDataset.

    Args:
        batch: list of (radar_seq, lidar_seq, meta_dict) from SequentialDataset.__getitem__

    Returns:
        radar_batch: (B, T, 1, H_radar, W_radar) float32 tensor
        lidar_batch: (B, T, 1, H_lidar, W_lidar) float32 tensor
        traj_ids: list of int, one per batch slot
        reset_masks: list of bool (placeholder — reset logic computed in training loop)
    """
    radar_seqs, lidar_seqs, metas = zip(*batch)

    # Stack along new batch dimension: each seq is (T, 1, H, W) -> (B, T, 1, H, W)
    radar_batch = torch.stack(radar_seqs, dim=0)
    lidar_batch = torch.stack(lidar_seqs, dim=0)

    traj_ids = [m['traj_id'] for m in metas]
    # reset_masks are determined externally by the training loop based on
    # trajectory boundary crossings tracked by TrajectoryBatchSampler
    reset_masks = [False] * len(batch)

    return radar_batch, lidar_batch, traj_ids, reset_masks
