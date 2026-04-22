# Roadmap

## v0.1 — "Works end-to-end" (current)

Scope: frozen BiomedCLIP + FAISS + CLI + Streamlit, on a single MedMNIST dataset.

- [x] Embedder wrapping BiomedCLIP via `open_clip`
- [x] FAISS `IndexFlatIP` with save/load + aligned metadata
- [x] `build_index.py`, `query.py`, `evaluate.py` CLIs
- [x] Streamlit demo with image + text queries
- [x] Unit tests for utils + index (no model download required)
- [x] CI: lint + unit tests on push

## v0.2 — Quality tuning

- [ ] Lightweight projection head on top of frozen backbone, trained with
      supervised contrastive loss on MedMNIST labels
- [ ] Comparison table: frozen vs. projection-head vs. full fine-tune
      across 3+ MedMNIST variants (accuracy, macro-F1, retrieval P@5)
- [ ] Hard-negative mining notebook
- [ ] UMAP/t-SNE embedding atlas notebook, committed as a static HTML export

## v0.3 — Scale

- [ ] Swap `IndexFlatIP` for `IndexHNSWFlat` with tuned `efSearch`
- [ ] Chunked index building with on-disk checkpoints
- [ ] Benchmark: build time, query latency p50/p95, memory, recall@5
      vs. exact baseline at 10k, 100k, 1M vectors
- [ ] NIH ChestX-ray14 loader (224x224 PNGs, 112k images)

## v0.4 — Multimodal retrieval

- [ ] Image → radiology-report retrieval (MIMIC-CXR-style pairs)
- [ ] Report → image retrieval with highlighted attended regions
- [ ] Evaluation: recall@1, recall@5, recall@10 on held-out pairs

## v0.5 — Serve it

- [ ] FastAPI service with `/embed` and `/search` endpoints
- [ ] Dockerfile + Compose stack (API + Streamlit frontend)
- [ ] Observability: latency histogram, index-size gauge, cache hit rate
- [ ] Load test: 100 QPS target on a laptop-class CPU

## Stretch ideas

- [ ] Active learning: flag query images that fall in low-density regions
      of the index (likely novel cases worth human review)
- [ ] Fault-pattern retrieval for medical-device sensor snapshots
      (ties in my Dr. Pirouz Hospital biomedical-engineering experience —
      out-of-domain but same pattern holds for non-medical imagery)
- [ ] Federated index building across hospitals without sharing raw images
