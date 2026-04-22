"""Evaluate the quality of the embedding space via k-NN classification.

For each held-out query we retrieve k neighbors from the index and vote on
the majority label. Accuracy + macro-F1 are reported. High scores with no
training of the backbone = the pretrained embeddings are clinically useful.

Example:
    python scripts/evaluate.py \\
        --index artifacts/pathmnist_train \\
        --eval-dataset pathmnist --eval-split test --max-samples 1000 \\
        --model biomedclip --k 5
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from medembed.data import load_medmnist
from medembed.embedder import Embedder, EmbedderConfig
from medembed.index import VectorIndex
from medembed.utils import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--eval-dataset", default="pathmnist")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--model", default="biomedclip")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    return p.parse_args()


def knn_predict(neighbor_labels: list[int]) -> int:
    """Majority vote with ties broken by first occurrence."""
    return Counter(neighbor_labels).most_common(1)[0][0]


def main() -> None:
    args = parse_args()

    index = VectorIndex.load(args.index)
    embedder = Embedder(EmbedderConfig(model=args.model))

    samples = list(load_medmnist(
        name=args.eval_dataset,
        split=args.eval_split,
        max_samples=args.max_samples,
        download_root=args.data_root,
    ))
    logger.info("Evaluating on %d samples", len(samples))

    vecs = embedder.embed_images([s.image for s in samples])
    results = index.search(vecs, k=args.k)

    y_true: list[int] = []
    y_pred: list[int] = []
    for sample, hits in zip(samples, results, strict=True):
        neighbor_labels = [int(h.metadata["label"]) for h in hits]
        y_pred.append(knn_predict(neighbor_labels))
        y_true.append(sample.label)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean())
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))

    print("\n=== k-NN embedding quality ===")
    print(f"dataset:      {args.eval_dataset} [{args.eval_split}]")
    print(f"samples:      {len(samples)}")
    print(f"k:            {args.k}")
    print(f"accuracy:     {acc:.4f}")
    print(f"macro-F1:     {f1_macro:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true_arr, y_pred_arr, zero_division=0))


if __name__ == "__main__":
    main()
