"""MedEmbed: visual similarity search for medical images using ViT embeddings."""

from medembed.embedder import Embedder
from medembed.index import VectorIndex

__version__ = "0.1.0"
__all__ = ["Embedder", "VectorIndex"]
