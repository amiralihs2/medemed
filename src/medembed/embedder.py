"""Image and text embedding via pretrained vision-language transformers.

Default backbone is BiomedCLIP (Microsoft), a CLIP variant pretrained on 15M
biomedical image-text pairs from PubMed Central. A plain OpenAI CLIP backbone
is supported as a fallback for domains outside biomedicine. No fine-tuning is
required for v1 — we use the frozen backbone purely as a feature extractor.

All returned embeddings are L2-normalized, so inner products == cosine sim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from medembed.utils import batched, get_logger, l2_normalize

logger = get_logger(__name__)

# Model registry. Each entry specifies an open_clip-compatible model id.
MODEL_REGISTRY: dict[str, str] = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "clip-vit-b32": "ViT-B-32",          # generic CLIP, small + fast
    "clip-vit-b16": "ViT-B-16",          # generic CLIP, sharper
}
DEFAULT_MODEL = "biomedclip"


@dataclass
class EmbedderConfig:
    model: str = DEFAULT_MODEL
    device: str | None = None          # autodetect if None
    batch_size: int = 32
    pretrained: str | None = "openai"  # only used for generic CLIP


class Embedder:
    """Wraps a frozen vision-language backbone behind a tiny API.

    Example:
        embedder = Embedder()
        vecs = embedder.embed_images([img1, img2])           # (2, D)
        qvec = embedder.embed_text(["chest x-ray, pneumonia"])  # (1, D)
    """

    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()

        # Import lazily so the test suite doesn't require torch installed.
        import torch

        if self.config.model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.config.model}'. "
                f"Available: {sorted(MODEL_REGISTRY)}"
            )

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._torch = torch
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        import open_clip

        model_id = MODEL_REGISTRY[self.config.model]
        logger.info("Loading backbone '%s' on device '%s'...", model_id, self.device)

        if model_id.startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
            tokenizer = open_clip.get_tokenizer(model_id)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=self.config.pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_id)

        model.eval().to(self.device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # Probe embedding dim with a dummy forward pass.
        with self._torch.inference_mode():
            dummy = self._torch.zeros(1, 3, 224, 224, device=self.device)
            self.embed_dim = int(self.model.encode_image(dummy).shape[1])
        logger.info("Backbone ready. Embedding dim = %d", self.embed_dim)

    # ---------------------------------------------------------------- images
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return L2-normalized image embeddings with shape (N, D)."""
        if len(images) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        all_vecs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for batch in batched(images, self.config.batch_size):
                tensors = self._torch.stack([self.preprocess(img) for img in batch])
                tensors = tensors.to(self.device)
                feats = self.model.encode_image(tensors)
                all_vecs.append(feats.detach().cpu().numpy().astype(np.float32))
        return l2_normalize(np.concatenate(all_vecs, axis=0))

    # ----------------------------------------------------------------- text
    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized text embeddings with shape (N, D).

        Enables multimodal retrieval: embed a natural-language query and
        search over an image index (CLIP-style).
        """
        if len(texts) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        with self._torch.inference_mode():
            tokens = self.tokenizer(list(texts)).to(self.device)
            feats = self.model.encode_text(tokens)
        return l2_normalize(feats.detach().cpu().numpy().astype(np.float32))
