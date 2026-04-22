"""Build a FAISS index of embeddings over a MedMNIST dataset.

Example:
    python scripts/build_index.py \\
        --dataset pathmnist --split test --max-samples 2000 \\
        --model biomedclip --out artifacts/pathmnist_test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from medembed.data import load_medmnist
from medembed.embedder import Embedder, EmbedderConfig
from medembed.index import VectorIndex
from medembed.utils import get_logger

logger = get_logger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="pathmnist",
                   help="MedMNIST flag (pathmnist, dermamnist, pneumoniamnist, ...)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Cap the number of images indexed (None = full split)")
    p.add_argument("--model", default="biomedclip",
                   help="Backbone id from MODEL_REGISTRY")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, default=Path("artifacts/index"),
                   help="Output directory for the saved index")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    logger.info("Collecting samples from %s[%s]", args.dataset, args.split)
    samples = list(load_medmnist(
        name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        download_root=args.data_root,
    ))
    logger.info("Collected %d samples", len(samples))

    embedder = Embedder(EmbedderConfig(model=args.model, batch_size=args.batch_size))

    logger.info("Embedding images...")
    vecs = embedder.embed_images([s.image for s in samples])
    logger.info("Embeddings shape=%s", vecs.shape)

    index = VectorIndex(dim=embedder.embed_dim)
    index.add(vecs, [s.as_metadata() for s in samples])
    index.save(args.out)

    elapsed = time.time() - start
    logger.info("Done in %.1fs. Index at %s", elapsed, args.out)


if __name__ == "__main__":
    main()
