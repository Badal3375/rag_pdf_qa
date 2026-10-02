"""STEP 3 - EMBED: turn text into dense vectors (runs locally, free)."""
from __future__ import annotations

from typing import List

import numpy as np

_MODEL = None
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_model(name: str = DEFAULT_MODEL):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(name)
    return _MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )
    return np.asarray(vecs, dtype="float32")
