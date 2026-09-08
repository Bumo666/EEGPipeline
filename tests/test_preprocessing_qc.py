from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.artifact_rejection import reject_bad_trials
from preprocessing.channel_quality import analyze_channel_quality
from preprocessing.epoching import crop_timepoints


def test_crop_timepoints_to_fixed_length():
    """crop_timepoints should trim only the time dimension."""

    X = np.zeros((2, 3, 1001), dtype=np.float32)
    cropped, info = crop_timepoints(X, target_timepoints=1000)

    assert cropped.shape == (2, 3, 1000)
    assert info == {
        "enabled": True,
        "original_timepoints": 1001,
        "target_timepoints": 1000,
    }


def test_reject_bad_trials_by_amplitude():
    """reject_bad_trials should remove only trials above threshold."""

    X = np.zeros((3, 2, 5), dtype=np.float32)
    y = np.array([0, 1, 1], dtype=np.int64)
    X[1, 0, 0] = 200.0

    clean_X, clean_y, info = reject_bad_trials(X, y, max_abs=100.0)

    assert clean_X.shape == (2, 2, 5)
    np.testing.assert_array_equal(clean_y, np.array([0, 1], dtype=np.int64))
    assert info["n_before"] == 3
    assert info["n_after"] == 2
    assert info["n_rejected"] == 1
    assert info["rejected_indices"] == [1]


def test_channel_quality_detects_low_variance_channel():
    """channel quality report should be fitted on train data only."""

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(8, 3, 20)).astype(np.float32)
    X_train[:, 1, :] = 0.0

    info = analyze_channel_quality(
        X_train=X_train,
        channel_names=["C3", "Cz", "C4"],
    )

    assert info["fit_on"] == "train_only"
    assert info["bad_channels"] == ["Cz"]
    assert "near_zero_variance" in info["bad_channel_reasons"]["Cz"]
