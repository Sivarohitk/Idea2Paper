"""
src/utils.py
Lightweight helpers for the simplified Idea2Paper app.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

# -------------------------
# Text helpers
# -------------------------

_WS = re.compile(r"[ \t\r\f\v]+")
_NONPRINT = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")

def normalize_ws(text: str) -> str:
    """
    Remove non-printables, collapse whitespace, trim.
    """
    if not text:
        return ""
    t = _NONPRINT.sub(" ", text)
    t = _WS.sub(" ", t)
    return t.strip()

def safe_truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    """
    Truncate to at most max_chars, cutting on a word boundary when possible.
    """
    text = (text or "")
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars)]
    # try to cut on last space to avoid mid-word chops
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + (suffix if suffix else "")

def combine_with_limit(chunks: Sequence[str], max_chars: int, sep: str = "\n\n") -> str:
    """
    Join string chunks until hitting a character budget.
    """
    out: List[str] = []
    total = 0
    for c in chunks:
        if not c:
            continue
        c = str(c)
        if total + len(c) + (len(sep) if out else 0) > max_chars:
            # append truncated tail if there is some budget left
            budget = max_chars - total - (len(sep) if out else 0)
            if budget > 10:
                out.append(safe_truncate(c, budget))
            break
        out.append(c)
        total += len(c) + (len(sep) if out else 0)
    return sep.join(out)

def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    seen = set()
    out: List[T] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def join_nonempty(parts: Sequence[str], sep: str = "\n\n") -> str:
    return sep.join(p for p in parts if p and str(p).strip())


# -------------------------
# Batching & filesystem
# -------------------------

def batched(iterable: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """
    Yield successive n-sized chunks from a sequence.
    """
    if n <= 0:
        n = 1
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# Timing & seeding
# -------------------------

@contextmanager
def timer(label: str = "step") -> Iterator[None]:
    """
    Usage:
        with timer("summarize"):
            ...
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timer] {label}: {dt:.2f}s")

def set_seed(seed: int = 42) -> None:
    """
    Seed common RNGs if available (numpy/torch).
    """
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -------------------------
# Tiny import helper
# -------------------------

def try_import(modname: str):
    """
    Attempt to import a module by name; return module or None.
    """
    try:
        return __import__(modname)
    except Exception:
        return None


__all__ = [
    "normalize_ws",
    "safe_truncate",
    "combine_with_limit",
    "dedupe_preserve_order",
    "join_nonempty",
    "batched",
    "ensure_dir",
    "timer",
    "set_seed",
    "try_import",
]
