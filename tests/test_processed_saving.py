from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_dataset import save_processed_dataset


def test_save_processed_dataset_layout(tmp_path):
    """Processed data must be saved as train.npz, test.npz, and meta.json."""

    X_train = np.zeros((8, 3, 10), dtype=np.float32)
    y_train = np.array([0, 1] * 4, dtype=np.int64)
    X_test = np.ones((2, 3, 10), dtype=np.float32)
    y_test = np.array([0, 1], dtype=np.int64)
    meta = {
        "source": "local_gdf",
        "files": ["B0101T.gdf"],
        "ignored_unlabeled_files": ["B0104E.gdf"],
        "skipped_epochs": 0,
        "fs": 250.0,
        "channels": ["C3", "Cz", "C4"],
        "classes": {"left_hand": 0, "right_hand": 1},
        "time_window": [0.0, 4.0],
        "bandpass": [4.0, 38.0],
    }

    output_dir = save_processed_dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta=meta,
        dataset="2b",
        subject=1,
        output_root=tmp_path,
        crop_info={
            "enabled": True,
            "original_timepoints": 11,
            "target_timepoints": 10,
        },
        artifact_info={
            "enabled": True,
            "method": "max_abs_and_finite_check",
            "max_abs": 100.0,
            "n_before": 10,
            "n_after": 10,
            "n_rejected": 0,
            "rejected_indices": [],
        },
        channel_quality_info={
            "enabled": True,
            "method": "train_channel_std_ratio",
            "fit_on": "train_only",
            "bad_channels": [],
        },
        standardization_info={
            "enabled": True,
            "method": "global_zscore",
            "fit_on": "train_only",
            "mean": 0.0,
            "std": 1.0,
        },
        split_info={
            "strategy": "official_session",
            "train_sessions": ["0train"],
            "test_sessions": ["1test"],
            "n_train": 8,
            "n_test": 2,
        },
    )

    assert output_dir == tmp_path / "BCICIV_2b" / "subject_01"
    assert (output_dir / "train.npz").exists()
    assert (output_dir / "test.npz").exists()
    assert (output_dir / "meta.json").exists()

    train = np.load(output_dir / "train.npz")
    test = np.load(output_dir / "test.npz")
    assert set(train.files) == {"X", "y"}
    assert set(test.files) == {"X", "y"}
    np.testing.assert_array_equal(train["X"], X_train)
    np.testing.assert_array_equal(train["y"], y_train)
    np.testing.assert_array_equal(test["X"], X_test)
    np.testing.assert_array_equal(test["y"], y_test)

    saved_meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert saved_meta == {
        "dataset": "BCICIV_2b",
        "subject": 1,
        "source": "local_gdf",
        "raw_files": ["B0101T.gdf"],
        "ignored_unlabeled_files": ["B0104E.gdf"],
        "skipped_epochs": 0,
        "fs": 250.0,
        "channels": ["C3", "Cz", "C4"],
        "classes": ["left_hand", "right_hand"],
        "time_window": [0.0, 4.0],
        "bandpass": [4.0, 38.0],
        "label_map": {"left_hand": 0, "right_hand": 1},
        "split": {
            "strategy": "official_session",
            "train_sessions": ["0train"],
            "test_sessions": ["1test"],
            "n_train": 8,
            "n_test": 2,
        },
        "X_train_shape": [8, 3, 10],
        "X_test_shape": [2, 3, 10],
        "preprocessing": {
            "crop": {
                "enabled": True,
                "original_timepoints": 11,
                "target_timepoints": 10,
            },
            "artifact_rejection": {
                "enabled": True,
                "method": "max_abs_and_finite_check",
                "max_abs": 100.0,
                "n_before": 10,
                "n_after": 10,
                "n_rejected": 0,
                "rejected_indices": [],
            },
            "channel_quality": {
                "enabled": True,
                "method": "train_channel_std_ratio",
                "fit_on": "train_only",
                "bad_channels": [],
            },
            "standardization": {
                "enabled": True,
                "method": "global_zscore",
                "fit_on": "train_only",
                "mean": 0.0,
                "std": 1.0,
            },
        },
    }
