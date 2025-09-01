"""
src/retrieval.py
arXiv retrieval for: upload → keywords → arXiv → rank (SPECTER)

- Build fielded search queries from keywords (title/abstract + optional categories)
- Fetch from arXiv API (Atom) via requests
- Parse with xml.etree (no feedparser dependency)
- Return DataFrame with: title, summary, authors[List[str]], published, url (abs), pdf_url

Public API:
    make_query_from_keywords(keywords: List[str], categories: List[str] | None = None) -> str
    fetch_arxiv(query: str, max_results: int = 40, sort_by: str = "relevance", sort_order: str = "descending") -> pd.DataFrame
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd

try:
    # Optional central config
    from src.config import ARXIV_CATEGORIES
except Exception:
    ARXIV_CATEGORIES = []  # type: ignore


_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_UA = (
    "Idea2Paper/1.0 (+https://example.local/; contact: none) "
    "Python-requests arXiv-client"
)


# ---------------------------
# Helpers
# ---------------------------

def _clean_ws(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()


def _abs_to_pdf(url: str) -> str:
    """Convert https://arxiv.org/abs/XXXX to a direct PDF link."""
    try:
        m = re.search(r"arxiv\.org/abs/([^?#]+)", url)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    except Exception:
        pass
    return url


def _get_text(elem: ET.Element, path: str) -> str:
    x = elem.find(path, _ATOM_NS)
    return _clean_ws(x.text if x is not None else "")


def _get_all(elem: ET.Element, path: str) -> List[str]:
    return [_clean_ws(n.text or "") for n in elem.findall(path, _ATOM_NS) if (n.text or "").strip()]


def _parse_authors(entry: ET.Element) -> List[str]:
    names = []
    for a in entry.findall("a:author", _ATOM_NS):
        nm = _get_text(a, "a:name")
        if nm:
            names.append(nm)
    return names


def _parse_links(entry: ET.Element) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for l in entry.findall("a:link", _ATOM_NS):
        href = (l.attrib.get("href") or "").strip()
        rel = (l.attrib.get("rel") or "").strip()
        typ = (l.attrib.get("type") or "").strip()
        if not href:
            continue
        if rel == "alternate":
            out["abs"] = href
        if typ == "application/pdf":
            out["pdf"] = href
    return out


# ---------------------------
# Query builder
# ---------------------------

def make_query_from_keywords(keywords: List[str], categories: Optional[List[str]] = None) -> str:
    """
    Build a fielded arXiv query:
      (ti:"kw1" OR abs:"kw1" OR ti:"kw2" OR abs:"kw2" ...) AND (cat:cs.LG OR cat:cs.AI ...)
    """
    kws = [k.strip() for k in (keywords or []) if k and k.strip()]
    if not kws:
        # safe default
        kws = ["machine learning"]

    # Title/abstract ORs
    pieces: List[str] = []
    for k in kws[:12]:  # keep it reasonable
        kq = k.replace('"', "")  # avoid breaking the query
        pieces.append(f'ti:"{kq}"')
        pieces.append(f'abs:"{kq}"')
    field_block = "(" + " OR ".join(pieces) + ")"

    # Optional categories (cat:cs.LG OR cat:cs.AI)
    cats = [c.strip() for c in (categories if categories is not None else ARXIV_CATEGORIES) if c.strip()]
    if cats:
        cat_block = "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"
        return f"{field_block} AND {cat_block}"
    return field_block


# ---------------------------
# Fetch + parse
# ---------------------------

def fetch_arxiv(
    query: str,
    max_results: int = 40,
    sort_by: str = "relevance",      # relevance | lastUpdatedDate | submittedDate
    sort_order: str = "descending",  # ascending | descending
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Call arXiv API and return a DataFrame with columns:
      ['title','summary','authors','published','url','pdf_url']

    Notes:
      - authors is a List[str]
      - url is the abstract page
      - pdf_url is a direct link to the PDF (constructed if not provided)
    """
    max_results = max(1, min(int(max_results), 200))  # arXiv recommends pagination; we keep ≤200 per call
    params = {
        "search_query": query or "all:machine learning",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    qstr = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
    url = f"{_ARXIV_API}?{qstr}"

    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Return empty DF but make it obvious in logs
        print(f"[arXiv] HTTP error: {e}")
        return pd.DataFrame(columns=["title", "summary", "authors", "published", "url", "pdf_url"])
    except Exception as e:
        print(f"[arXiv] Request error: {e}")
        return pd.DataFrame(columns=["title", "summary", "authors", "published", "url", "pdf_url"])

    # Parse Atom XML
    try:
        root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"[arXiv] XML parse error: {e}")
        return pd.DataFrame(columns=["title", "summary", "authors", "published", "url", "pdf_url"])

    rows: List[Dict[str, Any]] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        title = _get_text(entry, "a:title")
        summary = _get_text(entry, "a:summary")
        published = _get_text(entry, "a:published") or _get_text(entry, "a:updated")
        authors = _parse_authors(entry)
        links = _parse_links(entry)
        abs_url = links.get("abs") or _get_text(entry, "a:id") or ""
        pdf_url = links.get("pdf") or _abs_to_pdf(abs_url)

        if not title and not summary:
            continue

        rows.append(
            {
                "title": title,
                "summary": summary,
                "authors": authors,            # List[str]
                "published": published,        # ISO datetime string from arXiv
                "url": abs_url,                # abstract page
                "pdf_url": pdf_url,            # direct PDF link (best effort)
            }
        )

    if not rows:
        return pd.DataFrame(columns=["title", "summary", "authors", "published", "url", "pdf_url"])

    df = pd.DataFrame(rows)
    # Light cleanup: ensure types
    if "authors" in df.columns:
        df["authors"] = df["authors"].apply(lambda a: a if isinstance(a, list) else ([] if pd.isna(a) else [str(a)]))
    return df
