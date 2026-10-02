"""STEP 4 - INDEX / STORE: a tiny vector store using cosine similarity.

Vectors are L2-normalised, so cosine similarity == dot product.
Swap for FAISS / Chroma / pgvector when the corpus grows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self):
        self.vectors = np.zeros((0, 0), dtype="float32")
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        self.chunks = list(chunks)
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> List[Tuple[Chunk, float]]:
        if not len(self):
            return []
        scores = self.vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence (manage: save / load / delete) ----
    def save(self, folder: str) -> None:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vectors)
        (p / "chunks.json").write_text(
            json.dumps([c.__dict__ for c in self.chunks]), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        p = Path(folder)
        vs = cls()
        vs.vectors = np.load(p / "vectors.npy")
        data = json.loads((p / "chunks.json").read_text(encoding="utf-8"))
        vs.chunks = [Chunk(**d) for d in data]
        return vs
