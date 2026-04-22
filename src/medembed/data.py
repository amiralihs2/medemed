"""Loaders for public medical image datasets used in MedEmbed.

v1 uses MedMNIST v2 — a set of lightweight, standardized medical imaging
datasets (pathology, dermatology, chest X-ray, retinal OCT, ...). They are
small enough to embed end-to-end on a laptop in minutes, which makes them
great for iteration. Swapping in a larger dataset (NIH ChestX-ray14, ISIC,
CheXpert) is a drop-in change: just supply PIL images and labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from medembed.utils import get_logger

logger = get_logger(__name__)


# Class-label lookup for a few MedMNIST datasets. Full list at
# https://medmnist.com/ — extend here as new datasets are added.
LABEL_NAMES: dict[str, list[str]] = {
    "pathmnist": [
        "adipose", "background", "debris", "lymphocytes", "mucus",
        "smooth muscle", "normal colon mucosa",
        "cancer-associated stroma", "colorectal adenocarcinoma",
    ],
    "dermamnist": [
        "actinic keratoses", "basal cell carcinoma", "benign keratosis",
        "dermatofibroma", "melanoma", "melanocytic nevi",
        "vascular lesions",
    ],
    "pneumoniamnist": ["normal", "pneumonia"],
    "chestmnist": [f"finding_{i}" for i in range(14)],  # multi-label, 14 findings
    "octmnist": ["CNV", "DME", "drusen", "normal"],
    "bloodmnist": [
        "basophil", "eosinophil", "erythroblast", "ig",
        "lymphocyte", "monocyte", "neutrophil", "platelet",
    ],
}


@dataclass
class Sample:
    image: Image.Image
    label: int
    label_name: str
    source: str        # dataset name e.g. "pathmnist"
    split: str         # "train" / "val" / "test"
    idx: int           # position within the split

    def as_metadata(self) -> dict:
        """Serializable metadata dict for storage alongside a vector."""
        return {
            "label": int(self.label),
            "label_name": self.label_name,
            "source": self.source,
            "split": self.split,
            "idx": int(self.idx),
        }


def load_medmnist(
    name: str = "pathmnist",
    split: str = "test",
    max_samples: int | None = 1000,
    download_root: str | Path = "data",
) -> Iterator[Sample]:
    """Yield `Sample` objects from a MedMNIST dataset.

    Args:
        name: MedMNIST flag (e.g. 'pathmnist', 'dermamnist', 'pneumoniamnist').
        split: 'train' | 'val' | 'test'.
        max_samples: cap number of yielded samples (None = all).
        download_root: directory to cache downloaded archives.
    """
    import medmnist
    from medmnist import INFO

    if name not in INFO:
        raise ValueError(f"Unknown MedMNIST flag '{name}'. Available: {sorted(INFO)}")

    info = INFO[name]
    dataset_cls = getattr(medmnist, info["python_class"])
    logger.info("Loading MedMNIST[%s] split=%s ...", name, split)

    ds = dataset_cls(
        split=split,
        download=True,
        root=str(download_root),
        size=224,             # MedMNIST+ offers 224x224 variants — preferred for ViTs
    )

    label_names = LABEL_NAMES.get(name) or [f"class_{i}" for i in range(len(info["label"]))]

    for idx, (img, label) in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            break
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        lbl_int = int(np.asarray(label).flatten()[0])
        lbl_name = label_names[lbl_int] if lbl_int < len(label_names) else str(lbl_int)
        yield Sample(
            image=img,
            label=lbl_int,
            label_name=lbl_name,
            source=name,
            split=split,
            idx=idx,
        )
