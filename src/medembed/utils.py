"""Shared helpers: logging, image I/O, L2 normalization, batching."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

T = TypeVar("T")


def get_logger(name: str = "medembed", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a single stdout handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2-normalize a 2D array. Safe for all-zero rows."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most `batch_size` items from `iterable`."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB PIL.Image. Raises on missing file / bad format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return Image.open(p).convert("RGB")


def ensure_dir(path: str | Path) -> Path:
    """Create `path` as a directory if missing; return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
