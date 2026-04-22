"""Streamlit demo: upload a medical image or type a query, see nearest cases.

Run with:
    streamlit run app/streamlit_app.py -- --index artifacts/pathmnist_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make `medembed` importable when run via `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medembed.data import load_medmnist  # noqa: E402
from medembed.embedder import Embedder, EmbedderConfig  # noqa: E402
from medembed.index import VectorIndex  # noqa: E402


def parse_app_args() -> argparse.Namespace:
    """Parse CLI args that come after `--` when Streamlit passes them through."""
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=str, default="artifacts/index")
    p.add_argument("--model", type=str, default="biomedclip")
    p.add_argument("--dataset", type=str, default="pathmnist",
                   help="Used to re-load reference images for visualization")
    p.add_argument("--split", type=str, default="test")
    return p.parse_args()


# Streamlit doesn't forward argv cleanly; keep it simple and best-effort.
try:
    APP_ARGS = parse_app_args()
except SystemExit:
    APP_ARGS = argparse.Namespace(
        index="artifacts/index", model="biomedclip",
        dataset="pathmnist", split="test",
    )


@st.cache_resource(show_spinner="Loading backbone...")
def get_embedder(model: str) -> Embedder:
    return Embedder(EmbedderConfig(model=model))


@st.cache_resource(show_spinner="Loading index...")
def get_index(path: str) -> VectorIndex:
    return VectorIndex.load(path)


@st.cache_resource(show_spinner="Loading reference images...")
def get_reference_images(dataset: str, split: str) -> dict[int, Image.Image]:
    """Reload source dataset once so we can render neighbor thumbnails."""
    samples = list(load_medmnist(name=dataset, split=split, max_samples=None))
    return {s.idx: s.image for s in samples}


def main() -> None:
    st.set_page_config(page_title="MedEmbed", page_icon=":mag:", layout="wide")
    st.title("MedEmbed — medical image similarity search")
    st.caption(
        "Frozen BiomedCLIP embeddings + FAISS. No fine-tuning, no GPU at query time."
    )

    with st.sidebar:
        st.header("Settings")
        k = st.slider("Neighbors (k)", 1, 20, 5)
        mode = st.radio("Query mode", ["Image", "Text"], horizontal=True)
        st.caption(f"Index: `{APP_ARGS.index}`  |  Model: `{APP_ARGS.model}`")

    embedder = get_embedder(APP_ARGS.model)
    index = get_index(APP_ARGS.index)
    ref_images = get_reference_images(APP_ARGS.dataset, APP_ARGS.split)

    if embedder.embed_dim != index.dim:
        st.error(
            f"Backbone dim {embedder.embed_dim} != index dim {index.dim}. "
            "Rebuild the index with a matching model."
        )
        return

    # ---------- Query input ----------
    col_q, col_r = st.columns([1, 2], gap="large")

    with col_q:
        st.subheader("Query")
        if mode == "Image":
            uploaded = st.file_uploader(
                "Upload a medical image (PNG/JPG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded is None:
                st.info("Upload an image to run a search.")
                return
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Query image", use_column_width=True)
            query_vec = embedder.embed_images([img])
        else:
            text = st.text_input(
                "Describe the finding",
                value="colorectal adenocarcinoma with desmoplastic stroma",
            )
            if not text.strip():
                st.info("Type a description to run a search.")
                return
            query_vec = embedder.embed_text([text])

    # ---------- Results ----------
    with col_r:
        st.subheader(f"Top {k} matches")
        hits = index.search(query_vec, k=k)[0]
        cols_per_row = 5
        rows = [hits[i:i + cols_per_row] for i in range(0, len(hits), cols_per_row)]
        for row in rows:
            cols = st.columns(cols_per_row)
            for col, hit in zip(cols, row, strict=False):
                m = hit.metadata
                with col:
                    img = ref_images.get(int(m.get("idx", -1)))
                    if img is not None:
                        st.image(img, use_column_width=True)
                    st.caption(
                        f"**{m.get('label_name', '?')}**  \n"
                        f"score: `{hit.score:.3f}`  \n"
                        f"`{m.get('source','?')}/{m.get('split','?')}/{m.get('idx','?')}`"
                    )


if __name__ == "__main__":
    main()
