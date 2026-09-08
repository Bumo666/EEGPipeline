"""Check summary counts against arrays rather than metadata assumptions."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_dataset_summary import generate_summary, summarize_subject


@pytest.fixture
def subject_folder(tmp_path):
    folder = tmp_path / "BCICIV_2b/subject_01"
    folder.mkdir(parents=True)
    meta = {"dataset": "BCICIV_2b", "subject": 1, "source": "MOABB", "fs": 250,
            "channels": ["C3", "Cz", "C4"], "classes": ["right_hand", "left_hand"],
            "label_map": {"left_hand": 1, "right_hand": 0},
            "time_window": [0, 4], "bandpass": [4, 38],
            "split": {"strategy": "official_session"},
            "X_train_shape": [3, 3, 10], "X_test_shape": [3, 3, 10],
            "preprocessing": {"standardization": {}, "artifact_rejection": {}, "channel_quality": {}}}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for split in ("train", "test"):
        np.savez(folder / f"{split}.npz", X=np.zeros((3, 3, 10), dtype=np.float32),
                 y=np.array([1, 1, 0], dtype=np.int64))
    return folder


def test_summary_uses_label_map(subject_folder):
    row = summarize_subject(subject_folder)
    assert row["train_left_hand_count"] == 2
    assert row["train_right_hand_count"] == 1
    assert row["train_feet_count"] is None
    assert row["n_train"] == 3


def test_summary_rejects_unknown_label(subject_folder):
    np.savez(subject_folder / "test.npz", X=np.zeros((3, 3, 10)), y=np.array([0, 1, 9]))
    with pytest.raises(ValueError, match="Unknown label"):
        summarize_subject(subject_folder)


def test_summary_rejects_metadata_mismatch(subject_folder):
    np.savez(subject_folder / "train.npz", X=np.zeros((3, 2, 10)), y=np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="mismatch"):
        summarize_subject(subject_folder)


def test_summary_does_not_silently_skip_missing_subjects(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_summary(tmp_path, tmp_path / "summary.csv")
    assert not (tmp_path / "summary.csv").exists()
