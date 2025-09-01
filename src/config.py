"""
src/config.py
Central configuration for the simplified Idea2Paper app:

Upload doc → extract keywords → fetch arXiv → rank with SPECTER →
summarize (PEGASUS) using abstracts OR full PDFs → compose Markdown draft.

All values can be overridden via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


# ------------------------
# Project paths
# ------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data_samples"
DRAFT_DIR = ROOT_DIR / "drafts"

# Ensure directories exist
for d in (DATA_DIR, DRAFT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ------------------------
# Helpers
# ------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _detect_device() -> str:
    # Allow override
    want = os.getenv("IDEA2PAPER_DEVICE")
    if want in {"cpu", "cuda"}:
        return want
    try:
        import torch  # lazy import
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ------------------------
# Device
# ------------------------
# CHANGED: default to "cuda" (GPU). You can still override with env var IDEA2PAPER_DEVICE.
DEVICE = os.getenv("IDEA2PAPER_DEVICE", "cuda")


# ------------------------
# Retrieval / ranking (arXiv + SPECTER)
# ------------------------
# How many raw arXiv results to fetch before ranking
MAX_ARXIV_RESULTS = int(os.getenv("MAX_ARXIV_RESULTS", "40"))

# How many top-ranked papers to use in the draft (also controls PDF downloads)
TOP_K = int(os.getenv("TOP_K", "10"))

# SPECTER model (paper embeddings)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/allenai-specter")

# Optional: restrict to categories (comma-separated, e.g., "cs.LG,cs.AI")
ARXIV_CATEGORIES = [c.strip() for c in os.getenv("ARXIV_CATEGORIES", "").split(",") if c.strip()]


# ------------------------
# Summarization (PEGASUS or any HF seq2seq)
# ------------------------
# Default academic summarizer
SUM_MODEL = os.getenv("SUM_MODEL", "google/pegasus-arxiv")

# Long background length budget when summarizing abstracts (tokens-ish)
MAX_SUMMARY_LEN = int(os.getenv("MAX_SUMMARY_LEN", "900"))

# One-sentence per-paper bullet length (tokens-ish)
PER_PAPER_SENT_LEN = int(os.getenv("PER_PAPER_SENT_LEN", "70"))

# When composing background from multiple papers,
# how many papers to consider at most
BACKGROUND_PAPERS = int(os.getenv("BACKGROUND_PAPERS", str(TOP_K)))


# ------------------------
# Full-PDF mode (for higher-quality background)
# ------------------------
# Use full PDFs (download & extract) instead of only abstracts
USE_FULL_PDFS = _env_bool("USE_FULL_PDFS", True)

# PDF download guardrails
PDF_TIMEOUT = int(os.getenv("PDF_TIMEOUT", "30"))        # seconds per PDF
PDF_MAX_BYTES = int(os.getenv("PDF_MAX_BYTES", str(50_000_000)))  # 50 MB cap
PDF_CHAR_CAP = int(os.getenv("PDF_CHAR_CAP", str(150_000)))       # keep runtime sane

# Hierarchical (map-reduce) summarization for very long text
LONG_SUM_CHUNK_TOKENS = int(os.getenv("LONG_SUM_CHUNK_TOKENS", "900"))
LONG_SUM_OVERLAP = int(os.getenv("LONG_SUM_OVERLAP", "120"))
LONG_SUM_REDUCE_MAX_LEN = int(os.getenv("LONG_SUM_REDUCE_MAX_LEN", "1000"))


# ------------------------
# Misc
# ------------------------
SEED = int(os.getenv("SEED", "42"))


__all__ = [
    # Paths
    "ROOT_DIR", "DATA_DIR", "DRAFT_DIR",
    # Device
    "DEVICE",
    # Retrieval / ranking
    "MAX_ARXIV_RESULTS", "TOP_K", "EMBED_MODEL", "ARXIV_CATEGORIES",
    # Summarization
    "SUM_MODEL", "MAX_SUMMARY_LEN", "PER_PAPER_SENT_LEN", "BACKGROUND_PAPERS",
    # Full-PDF mode
    "USE_FULL_PDFS", "PDF_TIMEOUT", "PDF_MAX_BYTES", "PDF_CHAR_CAP",
    "LONG_SUM_CHUNK_TOKENS", "LONG_SUM_OVERLAP", "LONG_SUM_REDUCE_MAX_LEN",
    # Misc
    "SEED",
]
