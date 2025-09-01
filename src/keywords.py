"""
src/keywords.py
Keyword/keyphrase extraction for the simplified Idea2Paper pipeline.

- Preferred: KeyBERT with SentenceTransformer ("all-MiniLM-L6-v2")
- Fallback: lightweight n-gram frequency scorer with stopword filtering

Public API:
    extract_keywords(text: str, top_k: int = 12, ngram_range=(1,3)) -> List[str]
    normalize_keywords(kws: List[Tuple[str, float]], top_k: int = 12) -> List[str]
"""

from __future__ import annotations

import re
from typing import List, Tuple, Iterable, Dict, Optional

# ------------------------------
# Optional deps (graceful import)
# ------------------------------
_SB = None
_KB = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SB = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        from keybert import KeyBERT  # type: ignore
        _KB = KeyBERT(model=_SB)
    except Exception:
        _KB = None
except Exception:
    _SB = None
    _KB = None


# ------------------------------
# Basic text utilities
# ------------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-]")

# conservative English stopwords (trimmed)
_STOP = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","by","with","from","into","as","at",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","their","our","your","his","her",
    "we","they","you","i","me","my","mine","ours","yours","theirs","he","she","them","us",
    "can","could","may","might","shall","should","will","would","must","do","does","did","done","doing",
    "not","no","yes","than","such","via","using","used","use","based","approach","method","results","paper","study",
}

def _clean(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# ------------------------------
# Fallback n-gram scorer
# ------------------------------
def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{1,}", text.lower()) if t not in _STOP]

def _ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0 or not tokens:
        return []
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

def _score_candidates(text: str, ngram_range: Tuple[int, int]) -> Dict[str, float]:
    toks = _tokens(text)
    if not toks:
        return {}
    nmin, nmax = ngram_range
    scores: Dict[str, float] = {}
    for n in range(nmin, nmax + 1):
        for ng in _ngrams(toks, n):
            if all(t in _STOP for t in ng):
                continue
            phrase = " ".join(ng)
            # simple frequency + slight boost for longer n-grams
            scores[phrase] = scores.get(phrase, 0.0) + 1.0 * (1.0 + 0.15 * (n - 1))
    # normalize by length to avoid very long phrases dominating
    for k in list(scores.keys()):
        L = max(1, len(k))
        scores[k] = scores[k] / (1.0 + 0.002 * L)
    return scores


# ------------------------------
# Normalization / final selection
# ------------------------------
def normalize_keywords(kws: List[Tuple[str, float]], top_k: int = 12) -> List[str]:
    """
    Sort by score desc, deduplicate, cleanup punctuation/whitespace.
    """
    seen = set()
    out: List[str] = []
    for k, _ in sorted(kws, key=lambda x: -x[1]):
        k = _clean(k)
        if len(k) < 3:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= top_k:
            break
    return out


# ------------------------------
# Public API
# ------------------------------
def extract_keywords(text: str, top_k: int = 12, ngram_range: Tuple[int, int] = (1, 3)) -> List[str]:
    """
    Return deduplicated keyphrases for the given text.
    Prefers KeyBERT; falls back to an n-gram frequency scorer.

    Args:
        text: source document text (≥ ~200 chars recommended)
        top_k: number of keyphrases to return
        ngram_range: (min_n, max_n) n-gram lengths to consider
    """
    text = (text or "").strip()
    if len(text) < 80:
        return []

    # ---- Preferred: KeyBERT ----
    if _KB is not None:
        try:
            raw = _KB.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                use_maxsum=True,
                nr_candidates=max(40, top_k * 4),
                top_n=max(30, top_k * 3),
            )  # -> List[Tuple[str, score]]
            return normalize_keywords(raw, top_k=top_k)
        except Exception:
            pass  # fall through to fallback

    # ---- Fallback: simple n-gram frequency ----
    scores = _score_candidates(text, ngram_range=ngram_range)
    raw = list(scores.items())
    return normalize_keywords(raw, top_k=top_k)


__all__ = ["extract_keywords", "normalize_keywords"]
