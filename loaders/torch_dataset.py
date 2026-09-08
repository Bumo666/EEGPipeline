from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_DIR_NAMES = {
    "2a": "BCICIV_2a",
    "2b": "BCICIV_2b",
    "BCICIV_2a": "BCICIV_2a",
    "BCICIV_2b": "BCICIV_2b",
}

CHANNEL_FIRST_2D_FORMATS = {"eegnet", "eegnex"}


class EEGTorchDataset(Dataset):
    """PyTorch Dataset for processed EEG NPZ files."""

    def __init__(self, npz_path, model_format="native"):
        """Load X and y from a processed train/test NPZ file."""

        self.npz_path = Path(npz_path)
        self.model_format = model_format
        if self.model_format not in {"native", *CHANNEL_FIRST_2D_FORMATS}:
            raise ValueError("model_format must be 'native', 'eegnet', or 'eegnex'.")
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Processed NPZ file not found: {self.npz_path}")

        data = np.load(self.npz_path)
        self.X = data["X"].astype(np.float32, copy=False)
        self.y = data["y"].astype(np.int64, copy=False)

        if self.X.ndim != 3:
            raise ValueError(f"X must have shape [N, C, T], got {self.X.shape}.")
        if self.y.ndim != 1:
            raise ValueError(f"y must have shape [N], got {self.y.shape}.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have the same N, got {self.X.shape[0]} and {self.y.shape[0]}."
            )

    def __len__(self):
        """Return the number of EEG trials."""

        return int(self.y.shape[0])

    def __getitem__(self, index):
        """Return one EEG trial and its label."""

        x = self.X[index]
        if self.model_format in CHANNEL_FIRST_2D_FORMATS:
            x = x[None, :, :]
        return torch.from_numpy(x), torch.tensor(self.y[index], dtype=torch.long)


def get_dataloader(
    dataset,
    subject,
    split,
    batch_size,
    model_format="native",
    processed_root="processed",
    shuffle=None,
    num_workers=0,
):
    """Create a DataLoader for processed EEG data."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    if shuffle is None:
        shuffle = split == "train"

    npz_path = build_processed_npz_path(
        dataset=dataset,
        subject=subject,
        split=split,
        processed_root=processed_root,
    )
    eeg_dataset = EEGTorchDataset(npz_path=npz_path, model_format=model_format)
    return DataLoader(
        eeg_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_processed_npz_path(dataset, subject, split, processed_root="processed"):
    """Build processed/<dataset>/subject_xx/<split>.npz path."""

    dataset_dir = normalize_dataset_dir(dataset)
    subject_dir = f"subject_{int(subject):02d}"
    return Path(processed_root) / dataset_dir / subject_dir / f"{split}.npz"


def normalize_dataset_dir(dataset):
    """Map dataset key to processed dataset directory name."""

    try:
        return DATASET_DIR_NAMES[str(dataset)]
    except KeyError as exc:
        raise ValueError("dataset must be '2a', '2b', 'BCICIV_2a', or 'BCICIV_2b'.") from exc


def create_dataloader(dataset, batch_size, shuffle):
    """Create a PyTorch DataLoader from an EEGTorchDataset instance."""

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
