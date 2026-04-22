"""Thin wrapper over FAISS for building, saving, and querying a vector index.

Uses an inner-product index on L2-normalized vectors, which is equivalent to
cosine similarity and is the right choice for CLIP-style embeddings.
Metadata (filenames, labels, free-form dicts) is persisted alongside the raw
FAISS index so a single `VectorIndex.load(path)` restores everything.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from medembed.utils import ensure_dir, get_logger, l2_normalize

logger = get_logger(__name__)


@dataclass
class SearchHit:
    rank: int
    score: float               # cosine similarity in [-1, 1]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """In-memory FAISS index with aligned metadata."""

    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, Any]] = []

    # --------------------------------------------------------------- insert
    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add a batch of vectors + aligned metadata dicts."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected (N, {self.dim}) vectors, got {vectors.shape}"
            )
        if len(metadata) != vectors.shape[0]:
            raise ValueError(
                f"Metadata length {len(metadata)} != num vectors {vectors.shape[0]}"
            )
        # Defensive: re-normalize in case caller forgot.
        vectors = l2_normalize(vectors.astype(np.float32))
        self._index.add(vectors)
        self.metadata.extend(metadata)
        logger.info("Index now holds %d vectors", self._index.ntotal)

    # ---------------------------------------------------------------- size
    def __len__(self) -> int:
        return int(self._index.ntotal)

    # --------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int = 5) -> list[list[SearchHit]]:
        """Search top-k for each query vector.

        `query` can be shape (D,) or (N, D). Returns a list of length N,
        each a list of `SearchHit` sorted by descending similarity.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if self._index.ntotal == 0:
            raise RuntimeError("Index is empty. Call .add() first.")

        if query.ndim == 1:
            query = query[None, :]
        if query.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {query.shape[1]} != index dim {self.dim}"
            )
        query = l2_normalize(query.astype(np.float32))

        k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(query, k)

        results: list[list[SearchHit]] = []
        for row_scores, row_idxs in zip(scores, idxs, strict=True):
            hits: list[SearchHit] = []
            for rank, (score, idx) in enumerate(zip(row_scores, row_idxs, strict=True)):
                if idx == -1:       # FAISS sentinel for "no result"
                    continue
                hits.append(SearchHit(
                    rank=rank,
                    score=float(score),
                    metadata=self.metadata[int(idx)],
                ))
            results.append(hits)
        return results

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Persist index + metadata to `path/` (creates the directory)."""
        d = ensure_dir(path)
        self._faiss.write_index(self._index, str(d / "vectors.faiss"))
        with open(d / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(d / "info.json", "w") as f:
            json.dump({"dim": self.dim, "size": len(self)}, f, indent=2)
        logger.info("Saved index (%d vectors) to %s", len(self), d)

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Load a previously-saved index directory."""
        import faiss

        d = Path(path)
        info_path = d / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"No index found at {d}")
        with open(info_path) as f:
            info = json.load(f)

        obj = cls(dim=int(info["dim"]))
        obj._index = faiss.read_index(str(d / "vectors.faiss"))
        with open(d / "metadata.pkl", "rb") as f:
            obj.metadata = pickle.load(f)
        logger.info("Loaded index (%d vectors) from %s", len(obj), d)
        return obj
