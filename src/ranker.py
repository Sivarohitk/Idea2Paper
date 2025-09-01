"""
src/ranker.py
SPECTER-based relevance ranking for arXiv results vs. the user's uploaded document.

Usage:
    ranker = SpecterRanker()  # or SpecterRanker(device="cuda")
    ranked_df = ranker.rank(user_text, arxiv_df, prefer_fulltext=True, top_k=10)

- Embedding model: sentence-transformers/allenai-specter (configurable)
- Similarity: cosine (via normalized embeddings)
- Input DataFrame columns expected:
    title (str), summary (str), [optional] full_text (str), url (str), pdf_url (str)
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

# Pull defaults from central config with safe fallbacks
try:
    from src.config import EMBED_MODEL as _CFG_MODEL, DEVICE as _CFG_DEVICE
except Exception:
    _CFG_MODEL, _CFG_DEVICE = "sentence-transformers/allenai-specter", "cpu"

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required for SPECTER ranking.\n"
        "Install with: pip install sentence-transformers"
    ) from e

# NEW: prefer CUDA if available (no other behavior changes)
try:
    import torch  # type: ignore
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False


def _combine_fields(row: pd.Series, prefer_fulltext: bool = True) -> str:
    """
    Build the text to embed for a row.
    - If 'full_text' is present and prefer_fulltext=True, use it.
    - Else use:  title + '. ' + summary
    - Fall back to whichever exists.
    """
    if prefer_fulltext and isinstance(row.get("full_text", None), str) and row["full_text"].strip():
        return row["full_text"].strip()
    title = str(row.get("title", "") or "").strip()
    summ = str(row.get("summary", "") or "").strip()
    if title and summ:
        return f"{title}. {summ}"
    return title or summ


class SpecterRanker:
    """
    Minimal, fast SPECTER ranker with batching and cosine similarity.
    """

    def __init__(
        self,
        model_name: str = _CFG_MODEL,
        device: Optional[str] = _CFG_DEVICE,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        self.model_name = model_name
        self.normalize = bool(normalize_embeddings)
        self.batch_size = int(batch_size)
        self.show_progress_bar = bool(show_progress_bar)

        # Prefer GPU if available; fall back to CPU (no other changes)
        if device is None:
            device = "cuda" if _HAS_CUDA else "cpu"
        elif device == "cuda" and not _HAS_CUDA:
            device = "cpu"

        # Load model; SentenceTransformer handles device internally
        self.model = SentenceTransformer(model_name, device=device)

    # ---------- embedding helpers ----------

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a list of strings → float32 numpy matrix [n, d].
        """
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        X = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        return np.asarray(X, dtype="float32")

    # ---------- public API ----------

    def rank(
        self,
        user_text: str,
        df: pd.DataFrame,
        *,
        prefer_fulltext: bool = True,
        top_k: Optional[int] = None,
        add_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Rank arXiv rows by cosine similarity to `user_text`.

        Args:
            user_text: merged text from the user's uploads (≥ few sentences recommended)
            df: DataFrame with at least `title` and `summary` columns
                (optionally `full_text` for PDF-based pipelines)
            prefer_fulltext: if True and `full_text` exists, use it for embedding
            top_k: if set, return only the top_k rows
            add_columns: if True, adds/overwrites a `similarity` column

        Returns:
            DataFrame sorted by `similarity` desc (and truncated to top_k if provided).
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["title", "summary", "authors", "published", "url", "pdf_url", "similarity"])

        # Build corpus for arXiv entries
        corpus: List[str] = [_combine_fields(df.iloc[i], prefer_fulltext=prefer_fulltext) for i in range(len(df))]
        # Guard against all-empty corpus
        if not any(isinstance(t, str) and t.strip() for t in corpus):
            out = df.copy()
            if add_columns:
                out["similarity"] = 0.0
            return out.sort_values("similarity", ascending=False) if "similarity" in out.columns else out

        # Embed query and corpus
        q = (user_text or "").strip() or (df.iloc[0].get("summary", "") or "")
        q_emb = self._embed([q])  # [1, d]
        X = self._embed(corpus)   # [n, d]

        if q_emb.size == 0 or X.size == 0:
            out = df.copy()
            if add_columns:
                out["similarity"] = 0.0
            return out

        # Cosine similarity (dot if normalized)
        sims = (q_emb @ X.T).astype("float32")[0]  # [n]
        # Numerical safety: clip range
        sims = np.clip(sims, -1.0, 1.0)

        out = df.copy()
        if add_columns:
            out["similarity"] = sims

        out = out.sort_values("similarity", ascending=False, kind="mergesort").reset_index(drop=True)
        if top_k is not None:
            out = out.head(int(top_k))
        return out
