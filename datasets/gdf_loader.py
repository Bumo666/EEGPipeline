from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GDFEpochResult:
    """Container for epochs loaded from local GDF files."""

    X: np.ndarray
    y: np.ndarray
    fs: float
    files: list[str]
    skipped_epochs: int


def load_labeled_gdf_epochs(
    gdf_paths,
    event_id,
    channel_picks,
    fmin,
    fmax,
    tmin,
    tmax,
):
    """Load labeled EEG epochs from local GDF files without network access."""

    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "MNE is required for local GDF loading. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    X_parts = []
    y_parts = []
    used_files = []
    skipped_epochs = 0
    expected_fs = None

    for gdf_path in [Path(path) for path in gdf_paths]:
        if not gdf_path.exists():
            raise FileNotFoundError(f"Local GDF file not found: {gdf_path}")

        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if expected_fs is None:
            expected_fs = fs
        elif fs != expected_fs:
            raise ValueError(f"Sampling rate mismatch: {gdf_path} has {fs}, expected {expected_fs}.")

        picks = resolve_channel_picks(raw.ch_names, channel_picks)
        raw.filter(l_freq=fmin, h_freq=fmax, picks=picks, verbose="ERROR")

        n_times = int(round((tmax - tmin) * fs))
        for onset, description in zip(raw.annotations.onset, raw.annotations.description):
            label = event_id.get(str(description))
            if label is None:
                continue

            start = int(round((float(onset) + tmin) * fs))
            stop = start + n_times
            if start < 0 or stop > raw.n_times:
                skipped_epochs += 1
                continue

            epoch = raw.get_data(picks=picks, start=start, stop=stop)
            if epoch.shape[1] != n_times:
                skipped_epochs += 1
                continue

            X_parts.append(epoch)
            y_parts.append(label)

        used_files.append(str(gdf_path))

    if not X_parts:
        raise ValueError("No labeled epochs were found in the provided GDF files.")

    X = np.asarray(X_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    return GDFEpochResult(
        X=X,
        y=y,
        fs=float(expected_fs),
        files=used_files,
        skipped_epochs=skipped_epochs,
    )


def resolve_channel_picks(raw_channel_names, channel_picks):
    """Resolve channel names or integer indices into MNE-compatible picks."""

    if channel_picks == "first_22":
        if len(raw_channel_names) < 22:
            raise ValueError("Expected at least 22 channels for Dataset 2a.")
        return list(range(22))

    missing = [name for name in channel_picks if name not in raw_channel_names]
    if missing:
        raise ValueError(f"Missing channels in GDF file: {missing}")
    return list(channel_picks)
