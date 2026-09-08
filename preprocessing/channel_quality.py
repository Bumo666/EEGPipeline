from __future__ import annotations

import numpy as np


def analyze_channel_quality(
    X_train,
    channel_names,
    min_std=1e-6,
    min_std_ratio=0.1,
    max_std_ratio=10.0,
):
    """Report bad/noisy channels using train data only."""

    X_train = np.asarray(X_train, dtype=np.float32)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must have shape [N, C, T], got {X_train.shape}.")
    if len(channel_names) != X_train.shape[1]:
        raise ValueError(
            f"channel_names length must match C={X_train.shape[1]}, got {len(channel_names)}."
        )

    channel_std = np.std(X_train, axis=(0, 2))
    finite_mask = np.isfinite(X_train).all(axis=(0, 2))
    finite_std = channel_std[np.isfinite(channel_std)]
    median_std = float(np.median(finite_std)) if finite_std.size else 0.0

    bad_channels = []
    reasons = {}
    for idx, channel in enumerate(channel_names):
        channel_reasons = []
        std = float(channel_std[idx])
        if not bool(finite_mask[idx]) or not np.isfinite(std):
            channel_reasons.append("non_finite_values")
        if std < float(min_std):
            channel_reasons.append("near_zero_variance")
        if median_std > 0:
            if std < median_std * float(min_std_ratio):
                channel_reasons.append("low_variance_ratio")
            if std > median_std * float(max_std_ratio):
                channel_reasons.append("high_variance_ratio")
        if channel_reasons:
            bad_channels.append(str(channel))
            reasons[str(channel)] = channel_reasons

    return {
        "enabled": True,
        "method": "train_channel_std_ratio",
        "fit_on": "train_only",
        "min_std": float(min_std),
        "min_std_ratio": float(min_std_ratio),
        "max_std_ratio": float(max_std_ratio),
        "median_std": median_std,
        "channel_std": {
            str(channel): float(std) for channel, std in zip(channel_names, channel_std)
        },
        "bad_channels": bad_channels,
        "bad_channel_reasons": reasons,
    }
