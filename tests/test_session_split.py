from __future__ import annotations

import numpy as np
import pytest

from splits.session_split import make_session_split


def test_make_session_split_uses_official_session_labels():
    """Official split must use train/test session labels, not random sampling."""

    X = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    trial_metadata = {
        "session": ["0train", "0train", "1test", "1test", "1test", "0train"]
    }

    X_train, y_train, X_test, y_test, split_info = make_session_split(
        X=X,
        y=y,
        trial_metadata=trial_metadata,
    )

    np.testing.assert_array_equal(X_train, X[[0, 1, 5]])
    np.testing.assert_array_equal(y_train, y[[0, 1, 5]])
    np.testing.assert_array_equal(X_test, X[[2, 3, 4]])
    np.testing.assert_array_equal(y_test, y[[2, 3, 4]])
    assert split_info["strategy"] == "official_session"
    assert split_info["train_sessions"] == ["0train"]
    assert split_info["test_sessions"] == ["1test"]


def test_make_session_split_requires_test_session():
    """Local training-only data cannot be used for official session split."""

    X = np.zeros((2, 1, 3), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="No official test sessions"):
        make_session_split(
            X=X,
            y=y,
            trial_metadata={"session": ["0train", "0train"]},
        )
