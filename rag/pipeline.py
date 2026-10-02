"""STEP 5 - RETRIEVE + orchestration of the full RAG pipeline."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import generator
from .chunker import Chunk, chunk_pages
from .embedder import embed_texts
from .loader import load_pdf
from .vectorstore import VectorStore

INDEX_ROOT = Path("indexes")


@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[dict]


class RAGPipeline:
    def __init__(self, chunk_size: int = 800, overlap: int = 150, top_k: int = 4):
        self.chunk_size, self.overlap, self.top_k = chunk_size, overlap, top_k
        self.store = VectorStore()
        self.num_pages = 0

    # ---- build ----
    def ingest(self, pdf_bytes: bytes, name: Optional[str] = None) -> int:
        pages = load_pdf(pdf_bytes)                                  # 1 load
        self.num_pages = len(pages)
        chunks = chunk_pages(pages, self.chunk_size, self.overlap)   # 2 chunk
        vectors = embed_texts([c.text for c in chunks])              # 3 embed
        self.store.add(chunks, vectors)                              # 4 index
        if name:
            self.store.save(str(INDEX_ROOT / name))
        return len(chunks)

    # ---- query ----
    def retrieve(self, question: str):
        qv = embed_texts([question])[0]
        return self.store.search(qv, self.top_k)                     # 5 retrieve

    def ask(self, question: str, api_key=None, model=None, history=None) -> QAResult:
        hits = self.retrieve(question)
        ans = generator.answer(question, hits, api_key, model, history)  # 6 generate
        sources = [
            {"page": c.page, "score": round(s, 3), "text": c.text} for c, s in hits
        ]
        return QAResult(question, ans, sources)

    def generate_questions(self, n=10, api_key=None, model=None) -> List[str]:
        return generator.generate_questions(self.store.chunks, n, api_key, model)

    def answer_many(self, questions, api_key=None, model=None) -> List[QAResult]:
        return [self.ask(q, api_key, model) for q in questions]

    # ---- manage saved indexes ----
    @staticmethod
    def list_indexes() -> List[str]:
        if not INDEX_ROOT.exists():
            return []
        return sorted(p.name for p in INDEX_ROOT.iterdir() if p.is_dir())

    def load_index(self, name: str) -> None:
        self.store = VectorStore.load(str(INDEX_ROOT / name))

    @staticmethod
    def delete_index(name: str) -> None:
        shutil.rmtree(INDEX_ROOT / name, ignore_errors=True)
