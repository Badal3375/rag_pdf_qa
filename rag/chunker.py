"""STEP 2 - CHUNK: split page text into overlapping, sentence-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    id: int
    page: int
    text: str


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(
    pages: List[Tuple[int, str]], chunk_size: int = 800, overlap: int = 150
) -> List[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[Chunk] = []
    for page_no, text in pages:
        current = ""
        for sent in _sentences(text):
            # very long sentence: hard-split it
            while len(sent) > chunk_size:
                if current:
                    chunks.append(Chunk(len(chunks), page_no, current.strip()))
                    current = ""
                chunks.append(Chunk(len(chunks), page_no, sent[:chunk_size]))
                sent = sent[chunk_size - overlap:]
            if len(current) + len(sent) + 1 <= chunk_size:
                current = f"{current} {sent}".strip()
            else:
                chunks.append(Chunk(len(chunks), page_no, current.strip()))
                tail = current[-overlap:] if overlap else ""
                current = f"{tail} {sent}".strip()
        if current.strip():
            chunks.append(Chunk(len(chunks), page_no, current.strip()))
    return chunks
