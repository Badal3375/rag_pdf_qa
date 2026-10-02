"""STEP 6 - GENERATE: build a grounded prompt and produce the answer.

Two modes:
  * LLM mode  - Anthropic API (set ANTHROPIC_API_KEY)
  * Extractive mode - no API key; returns the most relevant sentences
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from .chunker import Chunk

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context from a PDF. "
    "If the answer is not in the context, say: 'I could not find this in the document.' "
    "Be concise and cite page numbers like (p. 3)."
)

STOPWORDS = set(
    "the a an and or of to in is are was were be been for on with as by at from that this "
    "it its into than then which who whom their there these those also can may such not "
    "but if we you they he she our your has have had will would should could more most "
    "other about over under between using used use based".split()
)


def build_context(hits: List[Tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[Page {c.page}]\n{c.text}" for c, _ in hits)


def _client(api_key: Optional[str]):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key)


def _model(model: Optional[str]) -> str:
    return model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


def llm_available(api_key: Optional[str] = None) -> bool:
    return _client(api_key) is not None


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower()) if w not in STOPWORDS]


def extractive_answer(question: str, hits: List[Tuple[Chunk, float]], n: int = 3) -> str:
    q = set(_tokens(question))
    scored = []
    for chunk, _ in hits:
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(q & set(_tokens(sent)))
            if overlap and len(sent) > 25:
                scored.append((overlap, sent.strip(), chunk.page))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "I could not find this in the document."
    seen, out = set(), []
    for _, sent, page in scored:
        if sent not in seen:
            seen.add(sent)
            out.append(f"{sent} (p. {page})")
        if len(out) == n:
            break
    return " ".join(out)


def answer(question: str, hits, api_key=None, model=None, history=None) -> str:
    client = _client(api_key)
    if client is None:
        return extractive_answer(question, hits)
    messages = list(history or [])
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}",
        }
    )
    resp = client.messages.create(
        model=_model(model),
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def top_keywords(chunks: List[Chunk], n: int) -> List[str]:
    counts = Counter(w for c in chunks for w in _tokens(c.text))
    return [w for w, _ in counts.most_common(n)]


def generate_questions(chunks: List[Chunk], n: int = 10, api_key=None, model=None) -> List[str]:
    """Create n study questions about the document."""
    client = _client(api_key)
    if client is not None:
        step = max(1, len(chunks) // 12)
        sample = "\n\n".join(c.text for c in chunks[::step][:12])[:9000]
        resp = client.messages.create(
            model=_model(model),
            max_tokens=900,
            system="You write clear study questions. Output one question per line, no numbering.",
            messages=[
                {
                    "role": "user",
                    "content": f"Write {n} diverse questions (factual, conceptual, "
                    f"summary) answerable from this text:\n\n{sample}",
                }
            ],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        qs = [re.sub(r"^\s*[\d\-\.\)\*]+\s*", "", l).strip() for l in text.splitlines()]
        qs = [q for q in qs if q.endswith("?")]
        if len(qs) >= n:
            return qs[:n]
    # fallback: keyword-based questions
    templates = [
        "What does the document say about {k}?",
        "How is {k} explained in the document?",
        "Why is {k} important in this document?",
    ]
    kws = top_keywords(chunks, n)
    return [templates[i % len(templates)].format(k=k) for i, k in enumerate(kws)][:n]
