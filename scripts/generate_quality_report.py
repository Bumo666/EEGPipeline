from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_DIRS = {
    "2a": "BCICIV_2a",
    "2b": "BCICIV_2b",
}


def main():
    """Generate visual quality report from processed EEG data."""

    import matplotlib.pyplot as plt

    output_dir = ROOT / "reports" / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records()
    write_summary_csv(records, output_dir / "quality_summary.csv")
    make_trial_count_plot(records, output_dir / "trial_counts.png", plt)
    make_rejected_trials_plot(records, output_dir / "rejected_trials.png", plt)
    make_class_balance_plot(records, "2a", output_dir / "class_balance_2a.png", plt)
    make_class_balance_plot(records, "2b", output_dir / "class_balance_2b.png", plt)
    make_channel_std_heatmap(records, "2a", output_dir / "channel_std_2a.png", plt)
    make_channel_std_heatmap(records, "2b", output_dir / "channel_std_2b.png", plt)
    make_report_markdown(records, output_dir / "quality_report.md")

    print(f"quality report: {output_dir}")


def collect_records():
    """Collect metadata and label statistics for all processed subjects."""

    records = []
    for dataset, dataset_dir in DATASET_DIRS.items():
        for subject_dir in sorted((ROOT / "processed" / dataset_dir).glob("subject_*")):
            meta_path = subject_dir / "meta.json"
            train_path = subject_dir / "train.npz"
            test_path = subject_dir / "test.npz"
            if not meta_path.exists() or not train_path.exists() or not test_path.exists():
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            train = np.load(train_path)
            test = np.load(test_path)
            y_train = train["y"]
            y_test = test["y"]
            record = {
                "dataset": dataset,
                "subject": int(meta["subject"]),
                "source": meta["source"],
                "split": meta["split"]["strategy"],
                "X_train_shape": meta["X_train_shape"],
                "X_test_shape": meta["X_test_shape"],
                "classes": meta["classes"],
                "channels": meta["channels"],
                "train_counts": label_counts(y_train, meta["classes"]),
                "test_counts": label_counts(y_test, meta["classes"]),
                "n_rejected": meta["preprocessing"]["artifact_rejection"]["n_rejected"],
                "bad_channels": meta["preprocessing"]["channel_quality"]["bad_channels"],
                "channel_std_ratio": channel_std_ratio(meta),
                "standardization": meta["preprocessing"]["standardization"],
            }
            records.append(record)
    return records


def label_counts(y, classes):
    """Count labels using class names."""

    return {
        class_name: int(np.sum(y == class_id))
        for class_id, class_name in enumerate(classes)
    }


def channel_std_ratio(meta):
    """Return channel std divided by train median std."""

    channel_quality = meta["preprocessing"]["channel_quality"]
    median_std = float(channel_quality.get("median_std", 1.0))
    channel_std = channel_quality.get("channel_std", {})
    if median_std == 0:
        return {name: np.nan for name in channel_std}
    return {name: float(value) / median_std for name, value in channel_std.items()}


def write_summary_csv(records, path):
    """Write one row per processed subject."""

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "subject",
                "source",
                "split",
                "X_train_shape",
                "X_test_shape",
                "n_rejected",
                "bad_channels",
                "train_counts",
                "test_counts",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset": record["dataset"],
                    "subject": record["subject"],
                    "source": record["source"],
                    "split": record["split"],
                    "X_train_shape": record["X_train_shape"],
                    "X_test_shape": record["X_test_shape"],
                    "n_rejected": record["n_rejected"],
                    "bad_channels": record["bad_channels"],
                    "train_counts": record["train_counts"],
                    "test_counts": record["test_counts"],
                }
            )


def make_trial_count_plot(records, path, plt):
    """Plot train/test trial counts per subject."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, dataset in zip(axes, ["2a", "2b"]):
        ds_records = sorted(filter_dataset(records, dataset), key=lambda r: r["subject"])
        subjects = [record["subject"] for record in ds_records]
        train_counts = [record["X_train_shape"][0] for record in ds_records]
        test_counts = [record["X_test_shape"][0] for record in ds_records]
        x = np.arange(len(subjects))
        width = 0.38
        ax.bar(x - width / 2, train_counts, width, label="train", color="#3b82f6")
        ax.bar(x + width / 2, test_counts, width, label="test", color="#f97316")
        ax.set_title(f"{dataset} official split trial counts")
        ax.set_xlabel("subject")
        ax.set_ylabel("trials")
        ax.set_xticks(x, [f"{subject:02d}" for subject in subjects])
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def make_rejected_trials_plot(records, path, plt):
    """Plot rejected trial counts per subject."""

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    labels = []
    values = []
    colors = []
    for dataset in ["2a", "2b"]:
        for record in sorted(filter_dataset(records, dataset), key=lambda r: r["subject"]):
            labels.append(f"{dataset}-{record['subject']:02d}")
            values.append(record["n_rejected"])
            colors.append("#ef4444" if record["n_rejected"] else "#94a3b8")
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_title("Rejected trials after artifact QC")
    ax.set_xlabel("subject")
    ax.set_ylabel("rejected trials")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def make_class_balance_plot(records, dataset, path, plt):
    """Plot class counts for train/test splits."""

    ds_records = sorted(filter_dataset(records, dataset), key=lambda r: r["subject"])
    classes = ds_records[0]["classes"]
    fig, axes = plt.subplots(
        len(classes),
        1,
        figsize=(11, max(2.2 * len(classes), 3)),
        sharex=True,
        constrained_layout=True,
    )
    if len(classes) == 1:
        axes = [axes]

    subjects = [record["subject"] for record in ds_records]
    x = np.arange(len(subjects))
    width = 0.38
    for ax, class_name in zip(axes, classes):
        train_counts = [record["train_counts"][class_name] for record in ds_records]
        test_counts = [record["test_counts"][class_name] for record in ds_records]
        ax.bar(x - width / 2, train_counts, width, label="train", color="#22c55e")
        ax.bar(x + width / 2, test_counts, width, label="test", color="#a855f7")
        ax.set_ylabel(class_name)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title(f"{dataset} class balance")
    axes[0].legend(frameon=False, ncols=2)
    axes[-1].set_xlabel("subject")
    axes[-1].set_xticks(x, [f"{subject:02d}" for subject in subjects])
    save_figure(fig, path)


def make_channel_std_heatmap(records, dataset, path, plt):
    """Plot train-only channel std ratios per subject."""

    ds_records = sorted(filter_dataset(records, dataset), key=lambda r: r["subject"])
    channels = ds_records[0]["channels"]
    matrix = np.asarray(
        [
            [record["channel_std_ratio"].get(channel, np.nan) for channel in channels]
            for record in ds_records
        ],
        dtype=float,
    )
    fig_width = max(10, 0.45 * len(channels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.5)
    ax.set_title(f"{dataset} train-only channel std ratio")
    ax.set_xlabel("channel")
    ax.set_ylabel("subject")
    ax.set_xticks(np.arange(len(channels)), channels, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(ds_records)), [f"{r['subject']:02d}" for r in ds_records])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("std / subject median std")
    save_figure(fig, path)


def make_report_markdown(records, path):
    """Write a short markdown summary of the generated figures."""

    total_subjects = len(records)
    total_rejected = sum(record["n_rejected"] for record in records)
    bad_channel_records = [
        f"{record['dataset']}-{record['subject']:02d}: {record['bad_channels']}"
        for record in records
        if record["bad_channels"]
    ]
    lines = [
        "# EEG Data Quality Report",
        "",
        f"- Subjects checked: {total_subjects}",
        f"- Total rejected trials: {total_rejected}",
        f"- Subjects with bad channel candidates: {len(bad_channel_records)}",
        "- Source: MOABB",
        "- Split: official_session",
        "- Standardization: train_only",
        "",
        "## Figures",
        "",
        "- trial_counts.png",
        "- rejected_trials.png",
        "- class_balance_2a.png",
        "- class_balance_2b.png",
        "- channel_std_2a.png",
        "- channel_std_2b.png",
        "",
    ]
    if bad_channel_records:
        lines.extend(["## Bad Channel Candidates", ""])
        lines.extend(f"- {item}" for item in bad_channel_records)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def filter_dataset(records, dataset):
    """Return records for one dataset."""

    return [record for record in records if record["dataset"] == dataset]


def save_figure(fig, path):
    """Save and close a matplotlib figure."""

    fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
