"""Tests for the pure-numpy/pure-Python utility helpers."""

import numpy as np
import pytest

from medembed.utils import batched, l2_normalize


def test_l2_normalize_unit_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    y = l2_normalize(x)
    # First row: (3,4) / 5 = (0.6, 0.8)
    assert np.allclose(y[0], [0.6, 0.8])
    assert np.allclose(y[1], [1.0, 0.0])
    # All-zero row should stay safely bounded (not NaN).
    assert not np.isnan(y[2]).any()


def test_l2_normalize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((3,)))


def test_batched_exact_chunks():
    assert list(batched(range(6), 2)) == [[0, 1], [2, 3], [4, 5]]


def test_batched_trailing_partial():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


def test_batched_rejects_nonpositive():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], 0))
