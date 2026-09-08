"""Versioned Braindecode adapters, not independent model reproductions."""

from importlib.metadata import version

import torch
from torch import nn


BACKEND_VERSION = "1.5.1"


class BaselineAdapter(nn.Module):
    """Accept pipeline [B, 1, C, T] or native [B, C, T] tensors."""

    def __init__(self, model, n_chans, n_times):
        super().__init__()
        self.model = model
        self.n_chans = n_chans
        self.n_times = n_times

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected [B, {self.n_chans}, {self.n_times}] or "
                f"[B, 1, {self.n_chans}, {self.n_times}], got {tuple(x.shape)}."
            )
        return self.model(x)


def build_baseline(name, n_chans, n_times, n_classes):
    """Build a fresh model with the pinned backend's default architecture."""
    if name not in {"eegnet", "eegnex"}:
        raise ValueError("name must be 'eegnet' or 'eegnex'.")
    try:
        installed = version("braindecode")
    except ModuleNotFoundError as exc:
        raise ImportError("Install requirements-models.txt first.") from exc
    if installed != BACKEND_VERSION:
        raise RuntimeError(f"Expected braindecode=={BACKEND_VERSION}, got {installed}.")
    from braindecode.models import EEGNet, EEGNeX

    constructor = {"eegnet": EEGNet, "eegnex": EEGNeX}[name]
    model = constructor(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    return BaselineAdapter(model, n_chans, n_times)


def train_step(model, optimizer, x, y, n_classes):
    """Run one update and fail on invalid logits, loss, gradients or weights."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if tuple(logits.shape) != (len(y), n_classes):
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits.")
    loss = nn.functional.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss.")
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("Missing or non-finite gradients.")
    grad_norm = torch.stack([p.grad.detach().double().norm() for p in parameters]).norm()
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError("Invalid or zero gradient norm.")
    # Snapshot after forward: max-norm layers may also change weights in forward.
    before = [p.detach().clone() for p in parameters]
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("Non-finite parameters after optimizer step.")
    if not any(not torch.equal(old, new) for old, new in zip(before, parameters)):
        raise ValueError("Optimizer did not update any parameter.")
    return {"loss": float(loss.detach()), "gradient_norm": float(grad_norm),
            "logits_shape": list(logits.shape), "parameters_updated": True}
