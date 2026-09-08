"""DataLoader interfaces for processed EEG datasets."""

from .torch_dataset import EEGTorchDataset, get_dataloader

__all__ = ["EEGTorchDataset", "get_dataloader"]
