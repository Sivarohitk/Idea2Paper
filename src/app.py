# src/app.py
from __future__ import annotations

# --- ensure imports work when running "streamlit run src/app.py"
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------------------

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import (
    DEVICE, SUM_MODEL,
    TOP_K, MAX_ARXIV_RESULTS,
    USE_FULL_PDFS, PDF_TIMEOUT, PDF_MAX_BYTES, PDF_CHAR_CAP,
    LONG_SUM_CHUNK_TOKENS, LONG_SUM_OVERLAP, LONG_SUM_REDUCE_MAX_LEN,
    DRAFT_DIR,
)
from src.ingest import extract_many
from src.keywords import extract_keywords
from src.retrieval import make_query_from_keywords, fetch_arxiv
from src.ranker import SpecterRanker
from src.summarizer import Summarizer
from src.generator import build_markdown, save_markdown

# Optional: full-PDF extraction helper (we'll import lazily when toggled)
try:
    from src.fulltext import fetch_pdf_text
except Exception:
    fetch_pdf_text = None  # type: ignore


# ===================== UI CONFIG =====================
st.set_page_config(page_title="Idea2Paper — Simple", page_icon="📄", layout="wide")
st.title("📄 Idea2Paper — Upload → find related arXiv papers → draft")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K papers to use", 3, 20, TOP_K, 1)
    max_arxiv = st.slider("Max arXiv results to fetch", 10, 120, MAX_ARXIV_RESULTS, 10)
    use_full_pdfs = st.checkbox("Use full PDFs for background (slower, better)", value=USE_FULL_PDFS)
    st.caption(f"Summarizer: `{SUM_MODEL}` • Device: `{DEVICE}`")

    if use_full_pdfs:
        with st.expander("Advanced PDF limits"):
            st.write(f"Timeout per PDF: {PDF_TIMEOUT}s")
            st.write(f"Max PDF bytes: {PDF_MAX_BYTES:,}")
            st.write(f"Char cap per PDF text: {PDF_CHAR_CAP:,}")
            st.write(f"Long-sum chunk: {LONG_SUM_CHUNK_TOKENS} tokens, overlap: {LONG_SUM_OVERLAP}, reduce to {LONG_SUM_REDUCE_MAX_LEN} tokens")


# ===================== SESSION HELPERS =====================
def get_ranker() -> SpecterRanker:
    if "ranker" not in st.session_state:
        st.session_state.ranker = SpecterRanker(device=DEVICE, show_progress_bar=False)
    return st.session_state.ranker

def get_summarizer() -> Summarizer:
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = Summarizer(SUM_MODEL, device=DEVICE)
    return st.session_state.summarizer


# ===================== STEP 1: UPLOAD =====================
st.subheader("1) Upload your draft / notes / experiment log")
files = st.file_uploader(
    "PDF, DOCX, TXT, MD, CSV accepted",
    type=["pdf", "docx", "txt", "md", "csv"],
    accept_multiple_files=True,
)

c1, c2 = st.columns(2)
with c1:
    extract_clicked = st.button("🧾 Extract text & suggest keywords", use_container_width=True, disabled=not files)
with c2:
    if st.button("🧹 Clear", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in {"ranker", "summarizer"}:
                del st.session_state[k]
        st.experimental_rerun()

if extract_clicked:
    extracted = extract_many(files, min_chars=200)
    st.session_state["uploaded_texts"] = extracted
    merged = "\n\n".join(t for _, t in extracted) if extracted else ""
    st.session_state["merged_text"] = merged
    st.session_state["keywords"] = extract_keywords(merged, top_k=12)

if "uploaded_texts" in st.session_state:
    st.success(f"Parsed {len(st.session_state['uploaded_texts'])} file(s).")
    with st.expander("Preview extracted text (first 1000 chars)"):
        st.code((st.session_state["merged_text"][:1000] + "…") if len(st.session_state["merged_text"]) > 1000 else st.session_state["merged_text"])

if "keywords" in st.session_state:
    st.subheader("Suggested keywords")
    kws = st.session_state["keywords"]
    st.write(", ".join(kws))
    # Let user tweak keywords
    kw_edit = st.text_input("Edit keywords (comma-separated)", value=", ".join(kws))
    if kw_edit:
        st.session_state["keywords"] = [k.strip() for k in kw_edit.split(",") if k.strip()]


# ===================== STEP 2: RETRIEVE & RANK =====================
st.subheader("2) Retrieve related arXiv papers and rank with SPECTER")

colA, colB = st.columns(2)
with colA:
    retrieve_clicked = st.button("🔎 Retrieve from arXiv", use_container_width=True, disabled="keywords" not in st.session_state)
with colB:
    rank_clicked = st.button("🏷️ Rank with SPECTER", use_container_width=True,
                             disabled=("merged_text" not in st.session_state) or ("arxiv_df" not in st.session_state))

if retrieve_clicked:
    q = make_query_from_keywords(st.session_state["keywords"])
    with st.spinner("Querying arXiv…"):
        df = fetch_arxiv(q, max_results=max_arxiv)
    st.session_state["arxiv_df"] = df
    st.success(f"Retrieved {len(df)} result(s).")

if "arxiv_df" in st.session_state:
    with st.expander("arXiv results (raw)"):
        st.dataframe(st.session_state["arxiv_df"][["title", "published", "summary", "url", "pdf_url"]], use_container_width=True, height=260)

if rank_clicked:
    df = st.session_state["arxiv_df"].copy()
    user_text = st.session_state["merged_text"]
    with st.spinner("Embedding & ranking (SPECTER)…"):
        ranker = get_ranker()
        ranked = ranker.rank(user_text, df, prefer_fulltext=False, top_k=None)  # rank by title+abstract first
    st.session_state["ranked_df"] = ranked
    st.success("Ranked papers are ready.")

if "ranked_df" in st.session_state:
    st.markdown("**Top ranked papers:**")
    st.dataframe(st.session_state["ranked_df"][["title", "published", "similarity", "url"]].head(top_k), use_container_width=True, height=300)


# ===================== STEP 3: SUMMARIZE & DRAFT =====================
st.subheader("3) Summarize & generate research-style draft (Markdown)")

def _one_sentence_synopsis(summarizer: Summarizer, text: str) -> str:
    try:
        return summarizer.one_sentence(text, max_len=70)
    except Exception:
        return summarizer.summarize(text, max_len=70)

generate_clicked = st.button("🧪 Summarize & Generate", use_container_width=True,
                             disabled=("ranked_df" not in st.session_state) or ("merged_text" not in st.session_state))

if generate_clicked:
    ranked = st.session_state["ranked_df"].head(top_k).copy()
    summarizer = get_summarizer()

    # 1) Summarize the user's own upload
    with st.spinner("Summarizing your document…"):
        user_summary = summarizer.summarize(st.session_state["merged_text"], max_len=220)

    # 2) Build background from papers (PDFs or abstracts)
    papers_background = ""
    per_paper_bullets = []
    if use_full_pdfs:
        if fetch_pdf_text is None:
            st.warning("Full-PDF helper not found; falling back to abstracts.")
        else:
            st.info("Downloading + extracting full PDFs (this may take a while)…")
            pdf_texts = []
            successes = 0
            for i, (_, row) in enumerate(ranked.iterrows(), start=1):
                purl = row.get("pdf_url") or row.get("url", "")
                try:
                    text = fetch_pdf_text(purl, timeout=PDF_TIMEOUT, max_bytes=PDF_MAX_BYTES)
                    if text and len(text) > 0:
                        if len(text) > PDF_CHAR_CAP:
                            text = text[:PDF_CHAR_CAP]
                        pdf_texts.append(text)
                        ranked.loc[row.name, "full_text"] = text
                        successes += 1
                except Exception as e:
                    st.info(f"PDF failed ({row['title'][:50]}…): {e}")

            if successes == 0:
                st.warning("Could not extract any PDFs; falling back to abstracts.")
            else:
                with st.spinner("Summarizing full-paper background (hierarchical)…"):
                    mega = " ".join(pdf_texts)
                    papers_background = summarizer.summarize_long(
                        mega,
                        chunk_tokens=LONG_SUM_CHUNK_TOKENS,
                        overlap=LONG_SUM_OVERLAP,
                        reduce_max_len=LONG_SUM_REDUCE_MAX_LEN,
                    )

    if not papers_background:
        # fallback to abstracts
        with st.spinner("Summarizing abstracts background…"):
            abstracts = " ".join((ranked["title"] + ". " + ranked["summary"]).tolist())
            papers_background = summarizer.summarize(abstracts, max_len=900)

    # Per-paper one-liners
    with st.spinner("Creating per-paper synopses…"):
        for _, row in ranked.iterrows():
            src_txt = str(row.get("summary") or "") if not use_full_pdfs else str(row.get("full_text") or row.get("summary") or "")
            syn = _one_sentence_synopsis(summarizer, src_txt[:4000] if src_txt else "")
            per_paper_bullets.append((row["title"], syn, row["url"]))

    background = f"{user_summary}\n\n{papers_background}"

    # 3) Compose Markdown and save
    idea_hint = ", ".join(st.session_state.get("keywords", [])[:5]) or "auto-draft"
    md = build_markdown(
        idea=idea_hint,
        ranked_df=ranked,
        background_summary=background,
        feasibility_text="Related work retrieved and ranked; background synthesized from selected papers.",
        per_paper_bullets=per_paper_bullets,
    )
    path = save_markdown(md, DRAFT_DIR, prefix="draft", idea_hint=idea_hint)

    st.success(f"Draft generated and saved to **{path.name}**")
    st.markdown("### Preview")
    st.markdown(md)
    st.download_button("⬇️ Download Markdown", md, file_name=path.name, mime="text/markdown")

st.caption("Tip: start with fewer Top K papers (e.g., 5) in full-PDF mode, then increase once it works smoothly.")
