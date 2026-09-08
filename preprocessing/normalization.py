from __future__ import annotations

import numpy as np


class EEGStandardizer:
    """Train-only EEG standardizer for arrays shaped [N, C, T]."""

    def __init__(self, eps: float = 1e-8):
        """Create a global standardizer with numerical stability epsilon."""

        self.eps = float(eps)
        self.mean_: float | None = None
        self.std_: float | None = None
        self.n_train_trials_: int | None = None

    def fit(self, train_X):
        """Compute mean/std using training data only."""

        train_X = self._validate_X(train_X, name="train_X")
        self.mean_ = float(np.mean(train_X))
        self.std_ = float(np.std(train_X))
        if self.std_ < self.eps:
            self.std_ = self.eps
        self.n_train_trials_ = int(train_X.shape[0])
        return self

    def transform(self, X):
        """Standardize X using mean/std learned from training data."""

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("EEGStandardizer must be fitted before transform().")
        X = self._validate_X(X, name="X")
        return ((X - self.mean_) / self.std_).astype(np.float32, copy=False)

    def fit_transform(self, train_X):
        """Fit on training data only, then return standardized training data."""

        return self.fit(train_X).transform(train_X)

    @staticmethod
    def _validate_X(X, name: str):
        """Validate that EEG input has shape [N, C, T]."""

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"{name} must have shape [N, C, T], got {X.shape}.")
        if min(X.shape) <= 0:
            raise ValueError(f"{name} must not contain empty dimensions, got {X.shape}.")
        return X


TrainOnlyStandardizer = EEGStandardizer
