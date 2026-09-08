from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.bnci_2a import BNCI2aConfig, BNCI2aDataset
from datasets.bnci_2b import BNCI2bConfig, BNCI2bDataset
from preprocessing.artifact_rejection import reject_bad_trials
from preprocessing.channel_quality import analyze_channel_quality
from preprocessing.epoching import crop_timepoints
from preprocessing.normalization import EEGStandardizer
from splits.session_split import make_session_split
from splits.within_subject import make_within_subject_split


DATASET_DIR_NAMES = {
    "2a": "BCICIV_2a",
    "2b": "BCICIV_2b",
}


def parse_args():
    """Parse command-line arguments for dataset preparation."""

    parser = argparse.ArgumentParser(description="Prepare one EEG dataset subject.")
    parser.add_argument("--dataset", choices=["2a", "2b"], default="2a")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--source", choices=["local", "moabb"], default="moabb")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default="processed")
    parser.add_argument(
        "--split-strategy",
        choices=["official", "within_subject"],
        default="official",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-timepoints", type=int, default=1000)
    parser.add_argument("--artifact-threshold", type=float, default=100.0)
    parser.add_argument("--no-artifact-rejection", action="store_true")
    parser.add_argument("--channel-min-std-ratio", type=float, default=0.1)
    parser.add_argument("--channel-max-std-ratio", type=float, default=10.0)
    parser.add_argument("--no-channel-quality", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def main():
    """Run the selected dataset loading entry point."""

    os.chdir(ROOT)
    args = parse_args()
    reader = get_dataset_reader(
        dataset=args.dataset,
        source=args.source,
        data_root=args.data_root,
    )

    X, y, meta = reader.load_subject(subject=args.subject)
    X, crop_info = crop_timepoints(X, target_timepoints=args.target_timepoints)
    if args.no_artifact_rejection:
        artifact_info = {
            "enabled": False,
            "method": None,
            "max_abs": None,
            "n_before": int(X.shape[0]),
            "n_after": int(X.shape[0]),
            "n_rejected": 0,
            "rejected_indices": [],
        }
    else:
        X, y, artifact_info = reject_bad_trials(
            X=X,
            y=y,
            max_abs=args.artifact_threshold,
        )
        meta["trial_metadata"] = filter_trial_metadata(
            trial_metadata=meta.get("trial_metadata", {}),
            n_before=artifact_info["n_before"],
            rejected_indices=artifact_info["rejected_indices"],
        )

    X_train, y_train, X_test, y_test, split_info = make_train_test_split(
        X=X,
        y=y,
        meta=meta,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.no_channel_quality:
        channel_quality_info = {"enabled": False}
    else:
        channel_quality_info = analyze_channel_quality(
            X_train=X_train,
            channel_names=meta["channels"],
            min_std_ratio=args.channel_min_std_ratio,
            max_std_ratio=args.channel_max_std_ratio,
        )

    if args.no_standardize:
        standardization_info = {
            "enabled": False,
            "method": None,
            "fit_on": None,
            "mean": None,
            "std": None,
        }
    else:
        standardizer = EEGStandardizer()
        X_train = standardizer.fit_transform(X_train)
        X_test = standardizer.transform(X_test)
        standardization_info = {
            "enabled": True,
            "method": "global_zscore",
            "fit_on": "train_only",
            "mean": float(standardizer.mean_),
            "std": float(standardizer.std_),
        }

    output_dir = save_processed_dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta=meta,
        dataset=args.dataset,
        subject=args.subject,
        output_root=args.output_root,
        crop_info=crop_info,
        artifact_info=artifact_info,
        channel_quality_info=channel_quality_info,
        standardization_info=standardization_info,
        split_info=split_info,
    )

    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"channels: {meta['channels']}")
    print(f"classes: {meta['classes']}")
    print(f"saved: {output_dir}")
    return X_train, y_train, X_test, y_test, meta


def get_dataset_reader(dataset, source="local", data_root=None):
    """Return the MOABB dataset reader for a supported dataset key."""

    data_root = Path(data_root) if data_root else None
    if dataset == "2a":
        return BNCI2aDataset(BNCI2aConfig(source=source, data_root=data_root))
    if dataset == "2b":
        return BNCI2bDataset(BNCI2bConfig(source=source, data_root=data_root))
    raise ValueError(f"Unsupported dataset: {dataset}")


def make_train_test_split(X, y, meta, split_strategy, test_size, random_state):
    """Create train/test arrays and a split metadata record."""

    if split_strategy == "official":
        X_train, y_train, X_test, y_test, split_info = make_session_split(
            X=X,
            y=y,
            trial_metadata=meta.get("trial_metadata", {}),
        )
        return X_train, y_train, X_test, y_test, split_info

    if split_strategy == "within_subject":
        X_train, y_train, X_test, y_test = make_within_subject_split(
            X=X,
            y=y,
            test_size=test_size,
            random_state=random_state,
        )
        split_info = {
            "strategy": "within_subject",
            "test_size": float(test_size),
            "random_state": int(random_state),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        return X_train, y_train, X_test, y_test, split_info

    raise ValueError("split_strategy must be 'official' or 'within_subject'.")


def filter_trial_metadata(trial_metadata, n_before, rejected_indices):
    """Remove rejected trials from trial-level metadata."""

    import numpy as np

    keep_mask = np.ones(int(n_before), dtype=bool)
    keep_mask[np.asarray(rejected_indices, dtype=np.int64)] = False
    filtered = {}
    for key, values in trial_metadata.items():
        values = np.asarray(values, dtype=object)
        if values.shape[0] != int(n_before):
            continue
        filtered[key] = values[keep_mask].tolist()
    return filtered


def save_processed_dataset(
    X_train,
    y_train,
    X_test,
    y_test,
    meta,
    dataset,
    subject,
    output_root="processed",
    crop_info=None,
    artifact_info=None,
    channel_quality_info=None,
    standardization_info=None,
    split_info=None,
):
    """Save train/test NPZ files and metadata JSON for one subject."""

    output_dir = build_processed_subject_dir(
        output_root=output_root,
        dataset=dataset,
        subject=subject,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np_savez_path(output_dir / "train.npz", X=X_train, y=y_train)
    np_savez_path(output_dir / "test.npz", X=X_test, y=y_test)

    output_meta = build_processed_meta(
        meta=meta,
        dataset=dataset,
        subject=subject,
        X_train_shape=X_train.shape,
        X_test_shape=X_test.shape,
        crop_info=crop_info,
        artifact_info=artifact_info,
        channel_quality_info=channel_quality_info,
        standardization_info=standardization_info,
        split_info=split_info,
    )
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2, ensure_ascii=False)

    return output_dir


def build_processed_subject_dir(output_root, dataset, subject):
    """Build processed/<dataset>/subject_xx output path."""

    dataset_dir = DATASET_DIR_NAMES[dataset]
    subject_dir = f"subject_{int(subject):02d}"
    return Path(output_root) / dataset_dir / subject_dir


def np_savez_path(path, X, y):
    """Save X and y arrays into one compressed NPZ file."""

    import numpy as np

    np.savez_compressed(path, X=X, y=y)


def build_processed_meta(
    meta,
    dataset,
    subject,
    X_train_shape,
    X_test_shape,
    crop_info=None,
    artifact_info=None,
    channel_quality_info=None,
    standardization_info=None,
    split_info=None,
):
    """Create the meta.json payload for processed data."""

    label_map = dict(meta["classes"])
    classes = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    return {
        "dataset": DATASET_DIR_NAMES[dataset],
        "subject": int(subject),
        "source": meta.get("source"),
        "raw_files": list(meta.get("files", [])),
        "ignored_unlabeled_files": list(meta.get("ignored_unlabeled_files", [])),
        "skipped_epochs": int(meta.get("skipped_epochs", 0)),
        "fs": float(meta["fs"]),
        "channels": list(meta["channels"]),
        "classes": classes,
        "time_window": list(meta["time_window"]),
        "bandpass": list(meta["bandpass"]),
        "label_map": label_map,
        "split": split_info or {"strategy": None},
        "X_train_shape": list(X_train_shape),
        "X_test_shape": list(X_test_shape),
        "preprocessing": {
            "crop": crop_info or {"enabled": False},
            "artifact_rejection": artifact_info or {"enabled": False},
            "channel_quality": channel_quality_info or {"enabled": False},
            "standardization": standardization_info or {"enabled": False},
        },
    }


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
