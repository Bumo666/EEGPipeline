from __future__ import annotations

import numpy as np


def reject_bad_trials(X, y, max_abs=100.0):
    """Reject trials whose absolute amplitude exceeds max_abs."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 3:
        raise ValueError(f"X must have shape [N, C, T], got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must have shape [N], got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same N, got {X.shape[0]} and {y.shape[0]}.")

    finite_mask = np.isfinite(X).all(axis=(1, 2))
    if max_abs is None:
        amplitude_mask = np.ones(X.shape[0], dtype=bool)
    else:
        amplitude_mask = np.max(np.abs(X), axis=(1, 2)) <= float(max_abs)
    keep_mask = finite_mask & amplitude_mask
    rejected_indices = np.flatnonzero(~keep_mask).astype(int).tolist()

    info = {
        "enabled": True,
        "method": "max_abs_and_finite_check",
        "max_abs": None if max_abs is None else float(max_abs),
        "n_before": int(X.shape[0]),
        "n_after": int(np.sum(keep_mask)),
        "n_rejected": int(np.sum(~keep_mask)),
        "rejected_indices": rejected_indices,
    }
    return X[keep_mask], y[keep_mask], info
