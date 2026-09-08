from __future__ import annotations

import numpy as np


def make_within_subject_split(X, y, test_size=0.2, random_state=42):
    """Create a stratified within-subject train/test split."""

    X = np.asarray(X)
    y = np.asarray(y)
    train_indices, test_indices = stratified_split_indices(
        y=y,
        test_size=test_size,
        random_state=random_state,
    )
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def stratified_split_indices(y, test_size=0.2, random_state=42):
    """Create stratified train/test indices from labels."""

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must have shape [N], got {y.shape}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)

        if label_indices.size < 2:
            raise ValueError(
                f"Class {label} has fewer than 2 samples; cannot create train/test split."
            )

        n_test = int(round(label_indices.size * test_size))
        n_test = min(max(n_test, 1), label_indices.size - 1)
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices
