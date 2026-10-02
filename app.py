"""Streamlit UI: RAG PDF Question & Answer app.  Run: streamlit run app.py"""
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag import RAGPipeline
from rag.generator import llm_available

load_dotenv()
st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄", layout="wide")

# ---------------- sidebar: settings + manage ----------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input(
    "Anthropic API key (optional)", type="password",
    value=os.getenv("ANTHROPIC_API_KEY", ""),
    help="Leave empty to use free extractive mode.",
)
model = st.sidebar.text_input("Model", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 1500, 800, 50)
overlap = st.sidebar.slider("Chunk overlap", 0, 400, 150, 25)
top_k = st.sidebar.slider("Top-k retrieved chunks", 1, 10, 4)
mode = "LLM (Anthropic)" if llm_available(api_key) else "Extractive (no API key)"
st.sidebar.info(f"Answer mode: **{mode}**")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
    st.session_state.chat = []
    st.session_state.auto = []
    st.session_state.doc_name = None
rag: RAGPipeline = st.session_state.rag
rag.top_k = top_k

st.sidebar.markdown("---")
st.sidebar.subheader("🗂 Saved indexes")
saved = RAGPipeline.list_indexes()
if saved:
    pick = st.sidebar.selectbox("Choose", saved)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Load"):
        rag.load_index(pick)
        st.session_state.doc_name = pick
        st.session_state.chat, st.session_state.auto = [], []
        st.rerun()
    if c2.button("Delete"):
        RAGPipeline.delete_index(pick)
        st.rerun()
else:
    st.sidebar.caption("No saved indexes yet.")

# ---------------- main ----------------
st.title("📄 RAG PDF Question & Answer")
st.caption("Upload a PDF → chunk → embed → retrieve → generate grounded answers.")

up = st.file_uploader("Upload a PDF", type=["pdf"])
if up and st.button("🔨 Build index", type="primary"):
    with st.spinner("Reading, chunking and embedding the PDF..."):
        try:
            rag.chunk_size, rag.overlap = chunk_size, overlap
            name = re.sub(r"[^\w\-]+", "_", up.name.rsplit(".", 1)[0])
            n = rag.ingest(up.getvalue(), name=name)
            st.session_state.doc_name = name
            st.session_state.chat, st.session_state.auto = [], []
            st.success(f"Indexed {rag.num_pages} pages into {n} chunks.")
        except Exception as e:
            st.error(f"Could not process PDF: {e}")

ready = len(rag.store) > 0
if ready:
    st.success(f"Active document: **{st.session_state.doc_name}** — {len(rag.store)} chunks")

tab_chat, tab_auto, tab_how = st.tabs(["💬 Ask questions", "📝 Auto Q&A (10+)", "🧠 How it works"])

with tab_chat:
    if not ready:
        st.info("Upload a PDF and click **Build index** to begin.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    with st.expander("Sources"):
                        for s in m["sources"]:
                            st.markdown(f"**Page {s['page']}** (score {s['score']})")
                            st.caption(s["text"][:500])
        q = st.chat_input("Ask something about the PDF...")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                res = rag.ask(q, api_key or None, model)
            st.session_state.chat.append(
                {"role": "assistant", "content": res.answer, "sources": res.sources}
            )
            st.rerun()
        if st.session_state.chat and st.button("Clear chat"):
            st.session_state.chat = []
            st.rerun()

with tab_auto:
    if not ready:
        st.info("Build an index first.")
    else:
        n_q = st.slider("Number of questions", 10, 25, 10)
        if st.button("✨ Generate questions & answers"):
            with st.spinner(f"Generating {n_q} questions and answering them..."):
                qs = rag.generate_questions(n_q, api_key or None, model)
                st.session_state.auto = rag.answer_many(qs, api_key or None, model)
        if st.session_state.auto:
            for i, r in enumerate(st.session_state.auto, 1):
                with st.expander(f"Q{i}. {r.question}"):
                    st.markdown(r.answer)
                    st.caption("Pages: " + ", ".join(sorted({str(s['page']) for s in r.sources})))
            df = pd.DataFrame(
                [{"Question": r.question, "Answer": r.answer} for r in st.session_state.auto]
            )
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "qa.csv", "text/csv")

with tab_how:
    st.markdown(
        """
### End-to-end RAG pipeline
1. **Load** – `pypdf` extracts and cleans text page by page.
2. **Chunk** – sentence-aware chunks with overlap keep context across boundaries.
3. **Embed** – `all-MiniLM-L6-v2` converts each chunk into a 384-d vector.
4. **Index** – normalised vectors stored in a NumPy vector store (saved to `indexes/`).
5. **Retrieve** – the question is embedded; top-k chunks by cosine similarity are returned.
6. **Generate** – retrieved context + question go to the LLM with a grounding prompt
   (or extractive fallback). Page citations are included.
7. **Manage** – save, reload and delete indexes from the sidebar; export Q&A as CSV.
"""
    )
