"""Dataset for 41-frame stacked LISTA log_power features + occupancy labels."""
import os
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from v2.data.split import TRAIN_TRAJS, VAL_TRAJS, TEST_TRAJS


class LISTAOccDataset(Dataset):
    """Single-trajectory dataset for LISTA log_power + occupancy labels.

    Args:
        traj_id: trajectory ID
        processed_dir: directory containing lista_logpow_{tid}.pt and lista_label_{tid}.pt
        history: number of history frames (M). Total input channels = history + 1.
    """
    def __init__(self, traj_id: int, processed_dir: str, history: int = 40):
        self.history = history
        self.features = torch.load(
            os.path.join(processed_dir, f'lista_logpow_{traj_id}.pt'),
            weights_only=True,
        ).float()  # (N, 256, 512) float32
        self.labels = torch.load(
            os.path.join(processed_dir, f'lista_label_{traj_id}.pt'),
            weights_only=True,
        ).float()  # (N, 256, 512) float32
        assert self.features.shape[0] == self.labels.shape[0]
        self.n_frames = self.features.shape[0]

    def __len__(self):
        return max(0, self.n_frames - self.history)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.history + 1
        x = self.features[start:end]  # (history+1, 256, 512)
        y = self.labels[end - 1].unsqueeze(0)  # (1, 256, 512)
        return x, y


def build_lista_dataloaders(
    processed_dir: str,
    history: int = 40,
    batch_size: int = 12,
    num_workers: int = 4,
) -> dict:
    """Build train/val/test DataLoaders."""
    split_configs = {
        'train': (TRAIN_TRAJS, True),
        'val': (VAL_TRAJS, False),
        'test': (TEST_TRAJS, False),
    }
    loaders = {}
    for split, (trajs, shuffle) in split_configs.items():
        datasets = []
        for tid in trajs:
            feat_path = os.path.join(processed_dir, f'lista_logpow_{tid}.pt')
            if os.path.exists(feat_path):
                datasets.append(LISTAOccDataset(tid, processed_dir, history))
        if datasets:
            combined = ConcatDataset(datasets)
            loaders[split] = DataLoader(
                combined, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True,
            )
        else:
            loaders[split] = None
    return loaders
