# src/fulltext.py
from __future__ import annotations

import io
import re
from typing import Optional

import requests

# Optional PDF extractors (prefer PyMuPDF, fallback to pypdf)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore


_UA = "Idea2Paper-PDF/1.0 (+https://local) Python-requests"


def _collapse_ws(s: str) -> str:
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def _extract_with_pymupdf(data: bytes) -> str:
    if not fitz:
        return ""
    try:
        parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                parts.append(page.get_text("text") or "")
        return _collapse_ws("\n".join(parts))
    except Exception:
        return ""


def _extract_with_pypdf(data: bytes) -> str:
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for p in getattr(reader, "pages", []):
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                parts.append("")
        return _collapse_ws("\n".join(parts))
    except Exception:
        return ""


def fetch_pdf_text(
    url: str,
    *,
    timeout: int = 25,
    max_bytes: int = 15_000_000,
    char_cap: Optional[int] = None,
) -> str:
    """
    Download a PDF (streaming) and return extracted text.
    - `max_bytes` limits the download size.
    - `char_cap` (optional) trims the returned text.

    Raises:
        requests.HTTPError, ValueError, or generic Exception on fatal issues.
    """
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": _UA}, allow_redirects=True) as r:
        r.raise_for_status()

        total = 0
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=32768):
            if not chunk:
                continue
            buf.extend(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"PDF too large (> {max_bytes} bytes)")

    data = bytes(buf)

    text = _extract_with_pymupdf(data) or _extract_with_pypdf(data)
    if not text:
        raise ValueError("Could not extract text from PDF (image-only or encrypted?)")

    if char_cap and len(text) > char_cap:
        text = text[:char_cap]
    return _collapse_ws(text)
