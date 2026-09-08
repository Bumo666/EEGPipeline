from __future__ import annotations


import numpy as np


def extract_epochs(raw, events, time_window):
    """Cut continuous EEG data into trial epochs according to event markers."""


def map_labels(events, event_mapping):
    """Map dataset-specific event codes to unified class labels."""


def crop_timepoints(X, target_timepoints):
    """Crop EEG epochs to a fixed number of time points."""

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must have shape [N, C, T], got {X.shape}.")
    if target_timepoints is None:
        return X, {
            "enabled": False,
            "original_timepoints": int(X.shape[-1]),
            "target_timepoints": None,
        }

    target_timepoints = int(target_timepoints)
    if target_timepoints <= 0:
        raise ValueError("target_timepoints must be positive.")
    if X.shape[-1] < target_timepoints:
        raise ValueError(
            f"Cannot crop from {X.shape[-1]} to {target_timepoints} time points."
        )

    return X[..., :target_timepoints], {
        "enabled": X.shape[-1] != target_timepoints,
        "original_timepoints": int(X.shape[-1]),
        "target_timepoints": target_timepoints,
    }
