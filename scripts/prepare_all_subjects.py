from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_dataset.py"
DEFAULT_SUBJECTS = list(range(1, 10))


def parse_args():
    """Parse batch preparation arguments."""

    parser = argparse.ArgumentParser(description="Prepare all selected EEG subjects.")
    parser.add_argument("--datasets", nargs="+", choices=["2a", "2b"], default=["2a", "2b"])
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument("--source", choices=["local", "moabb"], default="moabb")
    parser.add_argument("--data-root-2a", default=None)
    parser.add_argument("--data-root-2b", default=None)
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
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main():
    """Run the single-subject preparation script for every selected subject."""

    args = parse_args()
    summaries = []
    failures = []

    for dataset in args.datasets:
        for subject in args.subjects:
            result = run_one_subject(args=args, dataset=dataset, subject=subject)
            summaries.append(result)
            print_summary_line(result)

            if result["status"] != "ok":
                failures.append(result)
                if not args.continue_on_error:
                    break

        if failures and not args.continue_on_error:
            break

    summary_path = write_summary(args.output_root, summaries)
    print(f"summary: {summary_path}")

    if failures:
        failed_items = ", ".join(
            f"{item['dataset']}-subject_{item['subject']:02d}" for item in failures
        )
        raise SystemExit(f"Failed subjects: {failed_items}")


def run_one_subject(args, dataset, subject):
    """Prepare one dataset subject through scripts/prepare_dataset.py."""

    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--dataset",
        dataset,
        "--subject",
        str(subject),
        "--source",
        args.source,
        "--output-root",
        args.output_root,
        "--split-strategy",
        args.split_strategy,
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--target-timepoints",
        str(args.target_timepoints),
        "--artifact-threshold",
        str(args.artifact_threshold),
    ]
    data_root = get_data_root(args=args, dataset=dataset)
    if data_root:
        command.extend(["--data-root", data_root])

    completed = None
    for attempt in range(1, args.retries + 2):
        print(
            f"running: {dataset} subject {subject:02d} "
            f"(attempt {attempt}/{args.retries + 1})",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode == 0:
            break
        if attempt <= args.retries:
            print(
                f"retrying: {dataset} subject {subject:02d} "
                f"after {args.retry_delay:g}s",
                flush=True,
            )
            time.sleep(args.retry_delay)

    result = {
        "dataset": dataset,
        "subject": int(subject),
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "attempts": int(attempt),
    }

    if completed.returncode == 0:
        result.update(read_subject_meta(args.output_root, dataset, subject))

    return result


def read_subject_meta(output_root, dataset, subject):
    """Read shape and metadata summary for one processed subject."""

    dataset_dir = {"2a": "BCICIV_2a", "2b": "BCICIV_2b"}[dataset]
    meta_path = ROOT / output_root / dataset_dir / f"subject_{subject:02d}" / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    preprocessing = meta.get("preprocessing", {})
    artifact_info = preprocessing.get("artifact_rejection", {})
    channel_info = preprocessing.get("channel_quality", {})

    return {
        "meta_path": str(meta_path.relative_to(ROOT)),
        "source": meta.get("source"),
        "raw_files": meta.get("raw_files"),
        "ignored_unlabeled_files": meta.get("ignored_unlabeled_files"),
        "skipped_epochs": meta.get("skipped_epochs"),
        "split": meta.get("split"),
        "X_train_shape": meta.get("X_train_shape"),
        "X_test_shape": meta.get("X_test_shape"),
        "channels": meta.get("channels"),
        "classes": meta.get("classes"),
        "n_rejected": artifact_info.get("n_rejected"),
        "bad_channels": channel_info.get("bad_channels"),
    }


def get_data_root(args, dataset):
    """Return an optional dataset-specific local data root."""

    if dataset == "2a":
        return args.data_root_2a
    if dataset == "2b":
        return args.data_root_2b
    raise ValueError(f"Unsupported dataset: {dataset}")


def print_summary_line(result):
    """Print one compact status line for the current subject."""

    if result["status"] != "ok":
        print(f"failed: {result['dataset']} subject {result['subject']:02d}")
        return

    print(
        "ok: "
        f"{result['dataset']} subject {result['subject']:02d} "
        f"source={result['source']} "
        f"split={result['split'].get('strategy')} "
        f"train={result['X_train_shape']} "
        f"test={result['X_test_shape']} "
        f"rejected={result['n_rejected']} "
        f"bad_channels={result['bad_channels']}"
    )


def write_summary(output_root, summaries):
    """Write a JSON summary for all processed subjects."""

    output_dir = ROOT / output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    return summary_path


if __name__ == "__main__":
    main()
