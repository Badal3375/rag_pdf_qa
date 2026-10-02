# RAG PDF Question & Answer (Streamlit)

A complete, end-to-end Retrieval-Augmented Generation (RAG) project. Upload any PDF,
ask questions, and auto-generate **10+ questions with answers** grounded in the document.

## Pipeline (end to end)
| # | Step | File | Tech |
|---|------|------|------|
| 1 | Load PDF text | `rag/loader.py` | pypdf |
| 2 | Chunk with overlap | `rag/chunker.py` | sentence-aware splitter |
| 3 | Embed | `rag/embedder.py` | sentence-transformers (MiniLM) |
| 4 | Index / store | `rag/vectorstore.py` | NumPy cosine store + disk save |
| 5 | Retrieve top-k | `rag/pipeline.py` | cosine similarity |
| 6 | Generate answer | `rag/generator.py` | Anthropic API or extractive fallback |
| 7 | UI + manage | `app.py` | Streamlit |

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # optional: add ANTHROPIC_API_KEY
streamlit run app.py
```
First run downloads the embedding model (~90 MB).

## How to use
1. Upload a PDF, click **Build index**.
2. **Ask questions** tab: chat with the document; expand *Sources* to see cited chunks/pages.
3. **Auto Q&A** tab: generates 10–25 questions and answers; download as CSV.
4. Sidebar: tune chunk size / overlap / top-k; load or delete saved indexes.

## Modes
- **LLM mode** (API key set): fluent, grounded answers with page citations.
- **Extractive mode** (no key): returns the most relevant sentences. Fully free/offline after model download.

## Tests
```bash
pytest -q
```

## Limitations and next steps
- Scanned PDFs need OCR (e.g. `pytesseract`).
- Replace NumPy store with FAISS/Chroma for large corpora.
- Add a reranker (cross-encoder) and hybrid BM25 search.
- Evaluate with RAGAS (faithfulness, answer relevancy).
- Deploy on Streamlit Community Cloud / Hugging Face Spaces.
