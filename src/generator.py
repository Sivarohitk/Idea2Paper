"""
src/generator.py
Compose and save a research-style Markdown draft.

Inputs (typical):
- idea: short title/descriptor of the project or source document
- ranked_df: DataFrame with columns: title, summary, authors[List[str]|str], published, url, pdf_url, similarity
- background_summary: long background text (e.g., from user's doc + (abstracts or full PDFs))
- feasibility_text: short sentence(s) about plausibility / evidence
- per_paper_bullets: optional list of (title, one_sentence, url)

Public API:
    build_markdown(...)
    save_markdown(text, out_dir=DRAFT_DIR) -> Path
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple, Optional, Any
import re

try:
    # central config (paths etc.)
    from src.config import DRAFT_DIR
except Exception:
    DRAFT_DIR = Path("./drafts")


# ---------------------------
# Small helpers
# ---------------------------

def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def _clean_ws(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s+\n\s*", "\n", s)
    return s.strip()

def _first_paragraph(text: str, max_chars: int = 1200) -> str:
    """
    Use the first non-empty paragraph for Abstract, clipped to a budget.
    """
    if not text:
        return ""
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        return _clean_ws(text)[:max_chars]
    abstract = parts[0]
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rsplit(" ", 1)[0] + "…"
    return _clean_ws(abstract)

def _safe_slug(text: str, max_len: int = 64) -> str:
    text = (text or "draft").lower()
    text = re.sub(r"[^a-z0-9\-_ ]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)[:max_len]
    return text or "draft"

def _fmt_date(iso: str) -> str:
    """
    arXiv publishes ISO timestamps. Keep only YYYY-MM-DD if possible.
    """
    if not iso:
        return ""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", iso)
    return m.group(1) if m else iso

def _fmt_authors(a: Any) -> str:
    if isinstance(a, list):
        return ", ".join(a)
    return str(a or "")

def _mk_refs(ranked_df) -> List[str]:
    refs: List[str] = []
    if ranked_df is None or len(ranked_df) == 0:
        return refs
    for _, row in ranked_df.iterrows():
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip() or str(row.get("pdf_url", "")).strip()
        pub = _fmt_date(str(row.get("published", "")))
        authors = _fmt_authors(row.get("authors", ""))
        sim = row.get("similarity", None)
        sim_txt = f" (sim={sim:.3f})" if isinstance(sim, (int, float)) else ""
        item = f"- {authors} ({pub}). **{title}**. [arXiv]({url}){sim_txt}"
        refs.append(item)
    return refs


# ---------------------------
# Markdown composer
# ---------------------------

def build_markdown(
    *,
    idea: str,
    ranked_df,
    background_summary: str,
    feasibility_text: str = "",
    per_paper_bullets: Optional[List[Tuple[str, str, str]]] = None,
    include_placeholders: bool = True,
) -> str:
    """
    Compose a full Markdown draft from inputs.

    per_paper_bullets: list of (title, one_sentence_synopsis, url)
    """
    idea = idea.strip() or "Auto-generated research-style draft"
    generated = _now_str()

    # Abstract from first paragraph of the background
    abstract = _first_paragraph(background_summary, max_chars=1200)

    # Related Work (bullets)
    related = ""
    if per_paper_bullets:
        lines = [f"- **{t}** — {s} ([arXiv]({u}))" for (t, s, u) in per_paper_bullets if (t and s and u)]
        if lines:
            related = "## Related Work (Top-K synopses)\n" + "\n".join(lines) + "\n\n"

    refs = _mk_refs(ranked_df)
    refs_block = "\n".join(refs) if refs else "_No references retrieved._"

    # Optional placeholders help students structure their own sections.
    placeholders = ""
    if include_placeholders:
        placeholders = """## Proposed Method
Describe key components, assumptions, and operating regimes. Provide diagrams/equations as needed.

## Experimental Setup / Data
Summarize datasets or experiments (inputs, apparatus, conditions), train/validation splits, baselines.

## Results
Report key metrics with brief analysis; include error bars/confidence where applicable.

## Discussion
Interpret results, compare to prior work, and note limitations and future directions.

"""

    md = f"""# {idea}

**Generated:** {generated}

## Abstract
{abstract}

## Background
{_clean_ws(background_summary)}

{related}{placeholders}## Feasibility / Evidence Signal
{_clean_ws(feasibility_text or "Related work retrieved and ranked; see references.")}

## References
{refs_block}
"""
    return md


# ---------------------------
# Save helper
# ---------------------------

def save_markdown(text: str, out_dir: Path = DRAFT_DIR, prefix: str = "draft", idea_hint: str = "") -> Path:
    """
    Save Markdown text to the drafts directory with a timestamped filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(idea_hint or "draft")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_{slug}_{ts}.md"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["build_markdown", "save_markdown"]
