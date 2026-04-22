# Architecture notes

## Design goals

1. **Small** — the library is ~400 lines of Python. You can read it in one sitting.
2. **Replaceable** — every external dependency is behind a boundary. Change the
   backbone, the index type, or the dataset loader independently.
3. **Honest about trade-offs** — no training, no claims of clinical validity;
   this is an infrastructure sketch, not a diagnostic product.

## Components

### `medembed.embedder.Embedder`

Thin wrapper over an `open_clip` model. Handles:
- Lazy torch import (unit tests don't need torch installed)
- Device auto-selection (`cuda` if available, else `cpu`)
- Image + text encoding, both L2-normalized
- A model registry so the same CLI flag can swap backbones

Model choice matters more than any other knob here. BiomedCLIP was
pretrained on 15M biomedical image-text pairs from PubMed Central, which
gives it priors that generic CLIP lacks: it knows what "H&E staining" or
"pulmonary effusion" look like. For non-medical imagery, drop back to
`clip-vit-b16`.

### `medembed.index.VectorIndex`

Wraps `faiss.IndexFlatIP` — exact inner-product search on normalized
vectors, which is cosine similarity. Exact search is fine up to ~500k
vectors on a laptop; past that, switch to `IndexHNSWFlat` (roadmap v0.3).

Metadata is stored as a plain Python list aligned with the index. On
save, we write three files:

```
artifacts/index/
├── vectors.faiss   # FAISS binary
├── metadata.pkl    # List[dict] aligned with the index
└── info.json       # {"dim": 512, "size": 2000}
```

### `medembed.data.load_medmnist`

Yields `Sample` objects (PIL image + label + metadata) from MedMNIST
v2's 224x224 variants. Everything downstream only needs a PIL image
and a metadata dict, so swapping in a different dataset is mechanical.

## What this isn't

- **Not a diagnostic tool.** The embeddings are a useful *retrieval prior*,
  not a classifier. k-NN accuracy on MedMNIST is the right sanity-check
  metric; clinical validation would require an entirely different study
  design.
- **Not fine-tuned.** v0.1 is pure inference. v0.2 adds a projection head.
- **Not approximate.** Exact `IndexFlatIP` is fine at the scales we target
  for a demo. Approximate indexes are a scale-up concern, not a quality
  concern.

## Performance notes

Rough numbers on a 2020-era laptop CPU (no GPU), BiomedCLIP ViT-B/16:

- Embedding: ~8 images/sec per CPU core
- Index build for 2000 images: ~4 minutes
- Query (k=5, 10k vectors): ~2 ms
- Streamlit reload of cached backbone: instant after first load

On a CUDA GPU the embedding step is ~50x faster; query time is unchanged
(FAISS CPU is already sub-millisecond at this scale).
