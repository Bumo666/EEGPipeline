from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from datasets.gdf_loader import load_labeled_gdf_epochs


BNCI_2B_CHANNELS = ("C3", "Cz", "C4")
BNCI_2B_GDF_CHANNELS = ("EEG:C3", "EEG:Cz", "EEG:C4")

BNCI_2B_CLASSES = ("left_hand", "right_hand")
BNCI_2B_GDF_EVENT_ID = {
    "769": 0,
    "770": 1,
}

DEFAULT_MOABB_DATA_DIR = Path("moabb_data")
DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "BCICIV_2b_gdf"


@dataclass(frozen=True)
class BNCI2bConfig:
    """Minimal configuration for BNCI2014_004 Dataset 2b loading."""

    subject: int = 1
    fmin: float = 4.0
    fmax: float = 38.0
    tmin: float = 0.0
    tmax: float = 4.0
    sampling_rate: float = 250.0
    classes: Sequence[str] = BNCI_2B_CLASSES
    source: str = "moabb"
    data_root: Path | None = None


class BNCI2bDataset:
    """Minimal reader for BCI Competition IV Dataset 2b."""

    def __init__(self, config: BNCI2bConfig | None = None):
        """Create a Dataset 2b reader."""

        self.config = config or BNCI2bConfig()

    def load_subject(self, subject: int = 1):
        """Load one Dataset 2b subject and return X, y, meta."""

        if self.config.source == "local":
            return self._load_subject_from_gdf(subject=subject)
        if self.config.source == "moabb":
            return self._load_subject_from_moabb(subject=subject)
        raise ValueError("source must be 'local' or 'moabb'.")

    def _load_subject_from_gdf(self, subject: int = 1):
        """Load one Dataset 2b subject from local GDF files."""

        data_root = Path(self.config.data_root or DEFAULT_LOCAL_DATA_DIR)
        subject_prefix = f"B{int(subject):02d}"
        train_paths = [
            data_root / f"{subject_prefix}01T.gdf",
            data_root / f"{subject_prefix}02T.gdf",
            data_root / f"{subject_prefix}03T.gdf",
        ]
        eval_paths = [
            data_root / f"{subject_prefix}04E.gdf",
            data_root / f"{subject_prefix}05E.gdf",
        ]

        result = load_labeled_gdf_epochs(
            gdf_paths=train_paths,
            event_id=BNCI_2B_GDF_EVENT_ID,
            channel_picks=BNCI_2B_GDF_CHANNELS,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            tmin=self.config.tmin,
            tmax=self.config.tmax,
        )

        meta = self._build_meta(
            X=result.X,
            y=result.y,
            subject=subject,
            metadata=None,
            source="local_gdf",
            fs=result.fs,
            files=result.files,
            skipped_epochs=result.skipped_epochs,
            ignored_files=[str(path) for path in eval_paths if path.exists()],
        )
        return result.X, result.y, meta

    def _load_subject_from_moabb(self, subject: int = 1):
        """Load one Dataset 2b subject through MOABB and return X, y, meta."""

        DEFAULT_MOABB_DATA_DIR.mkdir(exist_ok=True)
        os.environ.setdefault("MNE_DATA", str(DEFAULT_MOABB_DATA_DIR))
        os.environ.setdefault("MNE_DATASETS_BNCI_PATH", str(DEFAULT_MOABB_DATA_DIR))

        try:
            import moabb
            from moabb.datasets import BNCI2014_004
            from moabb.paradigms import MotorImagery
        except ImportError as exc:
            raise ImportError(
                "MOABB is required for Dataset 2b loading. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        moabb.set_log_level("warning")

        dataset = BNCI2014_004()
        paradigm = MotorImagery(
            events=list(self.config.classes),
            n_classes=len(self.config.classes),
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            tmin=self.config.tmin,
            tmax=self.config.tmax,
        )

        X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        X = np.asarray(X, dtype=np.float32)
        y = self._encode_labels(labels)
        meta = self._build_meta(
            X=X,
            y=y,
            subject=subject,
            metadata=metadata,
            source="MOABB",
            fs=self.config.sampling_rate,
        )
        return X, y, meta

    def get_event_mapping(self):
        """Return Dataset 2b class-name to integer-label mapping."""

        return {name: index for index, name in enumerate(self.config.classes)}

    def get_channel_names(self):
        """Return Dataset 2b EEG channel names."""

        return list(BNCI_2B_CHANNELS)

    def _encode_labels(self, labels):
        """Convert MOABB string labels to integer labels."""

        mapping = self.get_event_mapping()
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)

    def _build_meta(
        self,
        X,
        y,
        subject,
        metadata,
        source,
        fs,
        files=None,
        skipped_epochs=0,
        ignored_files=None,
    ):
        """Build a minimal metadata dictionary for downstream code."""

        class_counts = {
            class_name: int(np.sum(y == class_id))
            for class_name, class_id in self.get_event_mapping().items()
        }
        return {
            "dataset": "2b",
            "moabb_dataset": "BNCI2014_004",
            "subject": int(subject),
            "source": source,
            "shape": list(X.shape),
            "fs": float(fs),
            "channels": self.get_channel_names(),
            "n_channels": int(X.shape[1]),
            "classes": self.get_event_mapping(),
            "n_classes": len(self.config.classes),
            "class_counts": class_counts,
            "time_window": [float(self.config.tmin), float(self.config.tmax)],
            "bandpass": [float(self.config.fmin), float(self.config.fmax)],
            "metadata_columns": list(getattr(metadata, "columns", [])),
            "trial_metadata": serialize_trial_metadata(metadata, X.shape[0]),
            "files": files or [],
            "ignored_unlabeled_files": ignored_files or [],
            "skipped_epochs": int(skipped_epochs),
        }


def load_bnci_2b_subject_1():
    """Load Dataset 2b subject 1 and return X, y, meta."""

    return BNCI2bDataset().load_subject(subject=1)


def serialize_trial_metadata(metadata, n_trials):
    """Convert MOABB metadata into plain Python lists."""

    if metadata is None:
        return {"session": ["0train"] * int(n_trials)}
    return {
        column: [to_python_scalar(value) for value in metadata[column].tolist()]
        for column in metadata.columns
    }


def to_python_scalar(value):
    """Convert NumPy scalar values into JSON-friendly Python values."""

    if hasattr(value, "item"):
        return value.item()
    return value
