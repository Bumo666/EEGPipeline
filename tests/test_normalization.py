from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.normalization import EEGStandardizer


def test_standardizer_fits_train_only_mean():
    """standardizer.mean_ must come from train_X only, not train+test."""

    train_X = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    test_X = np.ones((2, 3, 4), dtype=np.float32) * 100.0

    standardizer = EEGStandardizer().fit(train_X)

    assert standardizer.mean_ == float(train_X.mean())
    assert standardizer.mean_ != float(np.concatenate([train_X, test_X], axis=0).mean())


def test_transform_uses_train_statistics():
    """test_X must be transformed with train_X mean/std."""

    train_X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    test_X = np.ones((1, 3, 4), dtype=np.float32) * 100.0

    standardizer = EEGStandardizer().fit(train_X)
    transformed = standardizer.transform(test_X)
    expected = (test_X - train_X.mean()) / train_X.std()

    np.testing.assert_allclose(transformed, expected.astype(np.float32))
