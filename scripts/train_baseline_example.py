"""Minimal train-only integration example; not a paper benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.torch_dataset import DATASET_DIR_NAMES, get_dataloader
from models.baselines import build_baseline, train_step


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_subject(dataset, subject, model_name, processed_root, batch_size=8,
                epochs=1, max_batches=1, seed=42, device="cpu", learning_rate=0.001):
    """Create independent model/optimizer per subject and train on train.npz only."""
    if subject not in range(1, 10) or min(batch_size, epochs) < 1 or max_batches < 0:
        raise ValueError("subject must be 1-9; batch_size/epochs positive; max_batches >= 0.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    folder = Path(processed_root) / DATASET_DIR_NAMES[dataset] / f"subject_{subject:02d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if (meta["source"] != "MOABB" or meta["split"]["strategy"] != "official_session"
            or meta["preprocessing"]["standardization"].get("fit_on") != "train_only"):
        raise ValueError("Example requires MOABB official_session train-only processed data.")
    n_classes = len(meta["label_map"])
    if set(meta["label_map"].values()) != set(range(n_classes)):
        raise ValueError("Expected contiguous class IDs starting at zero.")
    loader = get_dataloader(dataset, subject, "train", batch_size, model_name,
                            processed_root=processed_root, shuffle=True, num_workers=0)
    shape = loader.dataset.X.shape
    if list(shape) != meta["X_train_shape"] or shape[1] != len(meta["channels"]):
        raise ValueError("Train array and metadata shapes disagree.")
    model = build_baseline(model_name, shape[1], shape[2], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0
    first_shape = None
    for _ in range(epochs):
        for batch_index, (x, y) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            if first_shape is None:
                first_shape = list(x.shape)
            last = train_step(model, optimizer, x.to(device), y.to(device), n_classes)
            steps += 1
    return {"dataset": dataset, "subject": subject, "model": model_name,
            "implementation": f"braindecode.models.{type(model.model).__name__}",
            "constructor_args": {"n_chans": shape[1], "n_times": shape[2], "n_outputs": n_classes},
            "architecture_options": "Pinned backend defaults",
            "seed": seed, "device": device, "optimizer": "Adam", "learning_rate": learning_rate,
            "epochs": epochs, "max_batches_per_epoch": max_batches,
            "steps": steps, "batch_shape": first_shape,
            "backend_input_shape": [first_shape[0], shape[1], shape[2]],
            "n_classes": n_classes, "train_trials": shape[0],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_sha256": file_hash(folder / "train.npz"),
            "meta_sha256": file_hash(folder / "meta.json"),
            "test_used": False, "validation_used": False, "status": "passed", **last}


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["2a", "2b", "both"], default="both")
    parser.add_argument("--model", choices=["eegnet", "eegnex", "both"], default="both")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--subjects", nargs="+", type=int, choices=range(1, 10), default=[1])
    group.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "processed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def run_cases(args, epochs=1, max_batches=1, learning_rate=0.001):
    """Run the requested subjects without sharing trained state across subjects."""
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(args.threads)
    datasets = ["2a", "2b"] if args.dataset == "both" else [args.dataset]
    models = ["eegnet", "eegnex"] if args.model == "both" else [args.model]
    subjects = list(range(1, 10)) if args.all_subjects else list(dict.fromkeys(args.subjects))
    records = []
    for dataset in datasets:
        for subject in subjects:
            for name in models:
                result = run_subject(dataset, subject, name, args.processed_root,
                                     args.batch_size, epochs, max_batches,
                                     args.seed, args.device, learning_rate)
                records.append(result)
                print(f"PASS {dataset} subject={subject:02d} {name}: "
                      f"{result['batch_shape']} -> {result['logits_shape']} "
                      f"steps={result['steps']} loss={result['loss']:.6f}", flush=True)
    return records


def write_report(path, records, purpose):
    """Record implementation versions and data hashes alongside check results."""
    path = Path(path)
    report = {"purpose": purpose, "paper_accuracy_reproduced": False,
              "python": platform.python_version(),
              "versions": {name: version(name) for name in ("torch", "numpy", "braindecode")},
              "cases_passed": len(records), "cases": records}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {path}")


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1, help="Per epoch; 0 runs all train batches.")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/integration/training_example.json")
    args = parser.parse_args()
    records = run_cases(args, args.epochs, args.max_batches, args.learning_rate)
    write_report(args.output, records, "Train-only usage example; no validation or test evaluation")


if __name__ == "__main__":
    main()
