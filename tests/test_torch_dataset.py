from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.torch_dataset import EEGTorchDataset, get_dataloader


def test_eeg_torch_dataset_native_shape(tmp_path):
    """Native samples should have shape [C, T]."""

    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        X=np.zeros((5, 3, 10), dtype=np.float32),
        y=np.arange(5, dtype=np.int64),
    )

    dataset = EEGTorchDataset(path)
    x, y = dataset[0]

    assert tuple(x.shape) == (3, 10)
    assert y.dtype == torch.long


def test_eeg_torch_dataset_eegnet_shape(tmp_path):
    """EEGNet samples should have shape [1, C, T]."""

    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        X=np.zeros((5, 3, 10), dtype=np.float32),
        y=np.arange(5, dtype=np.int64),
    )

    dataset = EEGTorchDataset(path, model_format="eegnet")
    x, _ = dataset[0]

    assert tuple(x.shape) == (1, 3, 10)


def test_eeg_torch_dataset_eegnex_shape(tmp_path):
    """EEGNeX samples should use the same [1, C, T] convention."""

    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        X=np.zeros((5, 3, 10), dtype=np.float32),
        y=np.arange(5, dtype=np.int64),
    )

    dataset = EEGTorchDataset(path, model_format="eegnex")
    x, _ = dataset[0]

    assert tuple(x.shape) == (1, 3, 10)


def test_get_dataloader_from_processed_root(tmp_path):
    """get_dataloader should read processed/<dataset>/subject_xx/<split>.npz."""

    out = tmp_path / "BCICIV_2b" / "subject_01"
    out.mkdir(parents=True)
    np.savez_compressed(
        out / "train.npz",
        X=np.zeros((6, 3, 10), dtype=np.float32),
        y=np.arange(6, dtype=np.int64),
    )

    loader = get_dataloader(
        dataset="2b",
        subject=1,
        split="train",
        batch_size=4,
        model_format="eegnet",
        processed_root=tmp_path,
        shuffle=False,
    )
    x, y = next(iter(loader))

    assert tuple(x.shape) == (4, 1, 3, 10)
    assert tuple(y.shape) == (4,)


def test_get_dataloader_eegnex_from_processed_root(tmp_path):
    """get_dataloader should support model_format='eegnex'."""

    out = tmp_path / "BCICIV_2a" / "subject_01"
    out.mkdir(parents=True)
    np.savez_compressed(
        out / "train.npz",
        X=np.zeros((6, 22, 10), dtype=np.float32),
        y=np.arange(6, dtype=np.int64),
    )

    loader = get_dataloader(
        dataset="2a",
        subject=1,
        split="train",
        batch_size=4,
        model_format="eegnex",
        processed_root=tmp_path,
        shuffle=False,
    )
    x, y = next(iter(loader))

    assert tuple(x.shape) == (4, 1, 22, 10)
    assert tuple(y.shape) == (4,)
