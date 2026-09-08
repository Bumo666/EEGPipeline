from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.torch_dataset import get_dataloader


def main():
    """Load BCICIV 2a subject 1 train split and print one batch shape."""

    loader = get_dataloader(
        dataset="2a",
        subject=1,
        split="train",
        batch_size=8,
        model_format="native",
    )
    x, y = next(iter(loader))
    print(f"x.shape: {tuple(x.shape)}")
    print(f"y.shape: {tuple(y.shape)}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nRun first: python scripts\\prepare_dataset.py --dataset 2a --subject 1"
        ) from exc
