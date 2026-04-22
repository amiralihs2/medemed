"""Tests for the FAISS-backed VectorIndex.

Skipped automatically if `faiss` is not installed in the current env.
Uses tiny random vectors so it runs in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from medembed.index import VectorIndex  # noqa: E402


def _random_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_add_and_search_returns_self_as_top_hit():
    d = 16
    vecs = _random_unit_vectors(20, d)
    idx = VectorIndex(dim=d)
    idx.add(vecs, [{"i": i} for i in range(len(vecs))])

    assert len(idx) == 20

    # Querying with an indexed vector should rank it #1 with score ~1.
    hits = idx.search(vecs[7], k=3)[0]
    assert len(hits) == 3
    assert hits[0].metadata["i"] == 7
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].rank == 0


def test_search_validates_dim():
    idx = VectorIndex(dim=8)
    idx.add(_random_unit_vectors(4, 8), [{} for _ in range(4)])
    with pytest.raises(ValueError):
        idx.search(np.zeros((1, 9), dtype=np.float32), k=2)


def test_add_validates_metadata_alignment():
    idx = VectorIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add(_random_unit_vectors(3, 4), metadata=[{"a": 1}])  # wrong length


def test_search_on_empty_index_raises():
    idx = VectorIndex(dim=4)
    with pytest.raises(RuntimeError):
        idx.search(np.zeros(4, dtype=np.float32), k=1)


def test_save_and_load_roundtrip(tmp_path):
    d = 12
    vecs = _random_unit_vectors(15, d)
    meta = [{"label": i % 3, "idx": i} for i in range(len(vecs))]

    idx = VectorIndex(dim=d)
    idx.add(vecs, meta)
    idx.save(tmp_path)

    restored = VectorIndex.load(tmp_path)
    assert len(restored) == 15
    assert restored.dim == d

    before = idx.search(vecs[0], k=3)[0]
    after = restored.search(vecs[0], k=3)[0]
    assert [h.metadata["idx"] for h in before] == [h.metadata["idx"] for h in after]
    for b, a in zip(before, after, strict=True):
        assert b.score == pytest.approx(a.score, abs=1e-5)
