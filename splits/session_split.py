from __future__ import annotations

import numpy as np


def make_session_split(X, y, trial_metadata, train_token="train", test_token="test"):
    """Split EEG trials by official train/test session labels."""

    X = np.asarray(X)
    y = np.asarray(y)
    sessions = np.asarray(trial_metadata.get("session", []), dtype=str)
    if X.ndim != 3:
        raise ValueError(f"X must have shape [N, C, T], got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must have shape [N], got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same N, got {X.shape[0]} and {y.shape[0]}.")
    if sessions.shape[0] != y.shape[0]:
        raise ValueError(
            f"trial_metadata['session'] must have length N={y.shape[0]}, got {sessions.shape[0]}."
        )

    lower_sessions = np.char.lower(sessions)
    train_mask = np.char.find(lower_sessions, train_token.lower()) >= 0
    test_mask = np.char.find(lower_sessions, test_token.lower()) >= 0
    if np.any(train_mask & test_mask):
        raise ValueError("A session cannot be both train and test.")
    if not np.any(train_mask):
        raise ValueError("No official train sessions were found.")
    if not np.any(test_mask):
        raise ValueError("No official test sessions were found.")

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    split_info = {
        "strategy": "official_session",
        "train_token": train_token,
        "test_token": test_token,
        "train_sessions": sorted(set(sessions[train_indices].tolist())),
        "test_sessions": sorted(set(sessions[test_indices].tolist())),
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
    }
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices], split_info
