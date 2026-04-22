"""Query a prebuilt index with an image file or a text prompt.

Examples:
    # Image query
    python scripts/query.py --index artifacts/pathmnist_test \\
        --image path/to/tile.png --k 5

    # Text query (multimodal; requires a CLIP-like backbone)
    python scripts/query.py --index artifacts/pathmnist_test \\
        --text "tumor epithelium with high nuclear density" --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medembed.embedder import Embedder, EmbedderConfig
from medembed.index import VectorIndex
from medembed.utils import get_logger, load_image

logger = get_logger("query")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True,
                   help="Directory of a saved VectorIndex")
    p.add_argument("--image", type=Path, help="Path to a query image")
    p.add_argument("--text", type=str, help="Natural-language query")
    p.add_argument("--model", default="biomedclip",
                   help="Must match the backbone used to build the index")
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.image is None) == (args.text is None):
        raise SystemExit("Provide exactly one of --image or --text")

    index = VectorIndex.load(args.index)
    embedder = Embedder(EmbedderConfig(model=args.model))
    if embedder.embed_dim != index.dim:
        raise SystemExit(
            f"Backbone produced dim {embedder.embed_dim}, index expects {index.dim}. "
            "Rebuild the index with a matching --model."
        )

    if args.image is not None:
        logger.info("Querying with image %s", args.image)
        query_vec = embedder.embed_images([load_image(args.image)])
    else:
        logger.info("Querying with text '%s'", args.text)
        query_vec = embedder.embed_text([args.text])

    hits = index.search(query_vec, k=args.k)[0]

    print(f"\nTop {len(hits)} matches:")
    print(f"{'rank':>4} {'score':>8}  {'label':<28} source/split/idx")
    print("-" * 72)
    for h in hits:
        m = h.metadata
        print(
            f"{h.rank:>4} {h.score:>8.4f}  {m.get('label_name', '?'):<28} "
            f"{m.get('source', '?')}/{m.get('split', '?')}/{m.get('idx', '?')}"
        )


if __name__ == "__main__":
    main()
