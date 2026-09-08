"""Test adapter and update checks without requiring optional Braindecode."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baselines import BaselineAdapter, train_step
from scripts import train_baseline_example as example


def tiny_model():
    return BaselineAdapter(nn.Sequential(nn.Flatten(), nn.Linear(30, 2)), 3, 10)


def test_adapter_preserves_batch_and_signal_axes():
    x = torch.randn(1, 3, 10)
    model = BaselineAdapter(nn.Identity(), 3, 10)
    assert torch.equal(model(x), x)
    assert torch.equal(model(x[:, None]), x)


@pytest.mark.parametrize("shape", [(2, 2, 3, 10), (2, 1, 10, 3)])
def test_adapter_rejects_wrong_axes(shape):
    with pytest.raises(ValueError, match="Expected"):
        tiny_model()(torch.zeros(shape))


def test_train_step_updates_weights():
    model = tiny_model()
    before = model.model[1].weight.detach().clone()
    report = train_step(model, torch.optim.Adam(model.parameters()),
                        torch.randn(4, 1, 3, 10), torch.tensor([0, 1, 0, 1]), 2)
    assert report["logits_shape"] == [4, 2]
    assert report["gradient_norm"] > 0
    assert not torch.equal(before, model.model[1].weight)


def test_train_step_rejects_nonfinite_input():
    model = tiny_model()
    with pytest.raises(ValueError, match="Non-finite logits"):
        train_step(model, torch.optim.Adam(model.parameters()),
                   torch.full((2, 3, 10), float("nan")), torch.tensor([0, 1]), 2)


def test_train_step_detects_no_update():
    model = tiny_model()
    with pytest.raises(ValueError, match="did not update"):
        train_step(model, torch.optim.SGD(model.parameters(), lr=0),
                   torch.randn(2, 3, 10), torch.tensor([0, 1]), 2)


def test_example_uses_only_train_and_resets_state(tmp_path, monkeypatch):
    folder = tmp_path / "BCICIV_2b/subject_01"
    folder.mkdir(parents=True)
    np.savez(folder / "train.npz", X=np.arange(240, dtype=np.float32).reshape(8, 3, 10) / 100,
             y=np.array([0, 1] * 4, dtype=np.int64))
    meta = {"source": "MOABB", "split": {"strategy": "official_session"},
            "preprocessing": {"standardization": {"fit_on": "train_only"}},
            "channels": ["C3", "Cz", "C4"], "X_train_shape": [8, 3, 10],
            "label_map": {"left_hand": 0, "right_hand": 1}}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(example, "build_baseline", lambda *args: tiny_model())
    first = example.run_subject("2b", 1, "eegnet", tmp_path, batch_size=2, epochs=2, max_batches=2)
    second = example.run_subject("2b", 1, "eegnet", tmp_path, batch_size=2, epochs=2, max_batches=2)
    assert first["steps"] == 4
    assert first["loss"] == second["loss"]
    assert first["test_used"] is False
    assert not (folder / "test.npz").exists()
