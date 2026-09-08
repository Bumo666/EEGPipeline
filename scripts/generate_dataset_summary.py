"""Generate one auditable CSV row per subject from processed arrays and metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.torch_dataset import DATASET_DIR_NAMES


def summarize_subject(folder):
    """Read actual arrays, validate their contract, and count labels by label_map."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    label_map = meta["label_map"]
    if set(label_map.values()) != set(range(len(label_map))):
        raise ValueError(f"Non-contiguous label_map: {folder}")
    prep = meta["preprocessing"]
    row = {"dataset": meta["dataset"], "subject": meta["subject"],
           "source": meta["source"], "split_strategy": meta["split"]["strategy"],
           "fs_hz": meta["fs"], "n_channels": len(meta["channels"]),
           "n_classes": len(label_map), "channels": meta["channels"],
           "classes": meta["classes"], "label_map": label_map,
           "time_window_s": meta["time_window"], "bandpass_hz": meta["bandpass"],
           "train_sessions": meta["split"].get("train_sessions"),
           "test_sessions": meta["split"].get("test_sessions"),
           "standardization_method": prep["standardization"].get("method"),
           "standardization_fit_on": prep["standardization"].get("fit_on"),
           "standardization_mean": prep["standardization"].get("mean"),
           "standardization_std": prep["standardization"].get("std"),
           "qc_enabled": prep["artifact_rejection"].get("enabled"),
           "qc_max_abs": prep["artifact_rejection"].get("max_abs"),
           "n_rejected_total": prep["artifact_rejection"].get("n_rejected"),
           "bad_channel_candidates": prep["channel_quality"].get("bad_channels"),
           "meta_path": f"{folder.parent.name}/{folder.name}/meta.json"}
    for split in ("train", "test"):
        with np.load(folder / f"{split}.npz", allow_pickle=False) as data:
            x, y = data["X"], data["y"]
            if x.ndim != 3 or y.shape != (len(x),) or len(x) == 0:
                raise ValueError(f"Invalid array shapes in {folder}/{split}")
            if list(x.shape) != meta[f"X_{split}_shape"] or x.shape[1] != len(meta["channels"]):
                raise ValueError(f"Array/metadata mismatch in {folder}/{split}")
            if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"Non-finite data or invalid labels in {folder}/{split}")
            if not np.isin(y, list(label_map.values())).all():
                raise ValueError(f"Unknown label in {folder}/{split}")
            row[f"n_{split}"] = len(y)
            row[f"X_{split}_shape"] = list(x.shape)
            row[f"X_{split}_dtype"] = str(x.dtype)
            row[f"y_{split}_dtype"] = str(y.dtype)
            for name in ("left_hand", "right_hand", "feet", "tongue"):
                row[f"{split}_{name}_count"] = (
                    int(np.sum(y == label_map[name])) if name in label_map else None
                )
    return row


def generate_summary(processed_root, output):
    """Require all 18 subjects; never silently skip missing data."""
    rows = [summarize_subject(Path(processed_root) / DATASET_DIR_NAMES[dataset] / f"subject_{s:02d}")
            for dataset in ("2a", "2b") for s in range(1, 10)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=ROOT / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dataset_summary.csv")
    args = parser.parse_args()
    rows = generate_summary(args.processed_root, args.output)
    print(f"Saved {len(rows)} subjects: {args.output}")
    for dataset in ("BCICIV_2a", "BCICIV_2b"):
        selected = [r for r in rows if r["dataset"] == dataset]
        print(dataset, "train=", sum(r["n_train"] for r in selected),
              "test=", sum(r["n_test"] for r in selected))


if __name__ == "__main__":
    main()
