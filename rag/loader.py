"""STEP 1 - LOAD: extract text from every page of a PDF."""
from __future__ import annotations

import io
import re
from typing import List, Tuple

from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)      # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(data: bytes) -> List[Tuple[int, str]]:
    """Return [(page_number, text), ...] (1-indexed). Skips empty pages."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned images (needs OCR)."
        )
    return pages
