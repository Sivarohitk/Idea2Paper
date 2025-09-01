"""
src/summarizer.py
Abstractive summarization utilities for Idea2Paper (simplified app).

- Default model: google/pegasus-arxiv
- Works on CPU or CUDA
- Provides:
    * Summarizer.summarize(text, max_len=...)
    * Summarizer.one_sentence(text, max_len=...)
    * Summarizer.summarize_long(text, chunk_tokens=..., overlap=..., reduce_max_len=...)

Notes:
- PEGASUS sometimes emits artifacts like "_<n>_"; _clean() removes these.
- summarize_long() splits by tokens → per-chunk summaries → reduces to a final summary.

Security note:
- We prefer loading models via *safetensors* (use_safetensors=True). If the
  repository doesn't provide safetensors and your torch < 2.6, loading a
  legacy .bin will be blocked by Transformers. In that case, either upgrade
  torch to >= 2.6 or pick a summarizer that ships safetensors
  (e.g., 'google/pegasus-xsum').
"""

from __future__ import annotations

import inspect
import re
from typing import List, Optional

import torch
from packaging.version import parse as parse_version
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---- Load knobs from central config (with safe fallbacks) ----
try:
    from src.config import (
        DEVICE as _CFG_DEVICE,
        SUM_MODEL as _CFG_MODEL,
        LONG_SUM_CHUNK_TOKENS as _CFG_CHUNK,
        LONG_SUM_OVERLAP as _CFG_OVERLAP,
        LONG_SUM_REDUCE_MAX_LEN as _CFG_REDUCE_LEN,
        MAX_SUMMARY_LEN as _CFG_MAX_SUMLEN,
        PER_PAPER_SENT_LEN as _CFG_ONE_SENT_LEN,
    )
except Exception:
    _CFG_DEVICE = "cpu"
    _CFG_MODEL = "google/pegasus-arxiv"
    _CFG_CHUNK = 900
    _CFG_OVERLAP = 120
    _CFG_REDUCE_LEN = 1000
    _CFG_MAX_SUMLEN = 900
    _CFG_ONE_SENT_LEN = 70


# ---------------------------
# Text cleaning for PEGASUS
# ---------------------------

_ARTIFACT = re.compile(r"\s*_<[^>]+>_?\s*")


def _clean(text: str) -> str:
    """Remove common seq2seq artifacts and normalize spacing/punctuation."""
    if not text:
        return ""
    t = _ARTIFACT.sub(" ", text)
    t = t.replace(" _ ", " ")
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = " ".join(t.split())
    return t


# ---------------------------
# Summarizer
# ---------------------------


class Summarizer:
    def __init__(
        self,
        model_name: str = _CFG_MODEL,
        device: str = _CFG_DEVICE,
        max_input_tokens: Optional[int] = None,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 4,
        length_penalty: float = 1.1,
    ):
        """
        Args:
            model_name: HF seq2seq model id (e.g., google/pegasus-arxiv)
            device: "cpu" or "cuda"
            max_input_tokens: truncate inputs to this token count (defaults to model max)
        """
        self.model_name = model_name
        self.device = device if device in {"cpu", "cuda"} else "cpu"
        # Prefer GPU if requested; gracefully fall back if not available
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ----- Model load with SAFETENSORS preference -----
        # transformers ≥ 4.44 prefers `dtype`; older uses `torch_dtype`
        fp16_on_gpu = (self.device == "cuda")
        supports_dtype_kw = "dtype" in inspect.signature(
            AutoModelForSeq2SeqLM.from_pretrained
        ).parameters

        dtype_kwargs = {}
        if fp16_on_gpu:
            if supports_dtype_kw:
                dtype_kwargs["dtype"] = torch.float16
            else:
                dtype_kwargs["torch_dtype"] = torch.float16

        # Try safetensors first to avoid torch.load on .bin files
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                **dtype_kwargs,
            )
        except Exception as e_sft:
            # If safetensors is unavailable, .bin load requires torch >= 2.6
            torch_ver = parse_version(torch.__version__.split("+")[0])
            if torch_ver < parse_version("2.6"):
                raise RuntimeError(
                    "Failed to load model via safetensors and your PyTorch version "
                    f"is {torch.__version__} (< 2.6). Transformers now blocks loading "
                    "legacy .bin checkpoints on older torch due to a security advisory.\n\n"
                    "Fix options:\n"
                    "  • Upgrade PyTorch to >= 2.6, OR\n"
                    "  • Switch to a summarizer that provides safetensors "
                    "(e.g., 'google/pegasus-xsum' or 'sshleifer/distill-pegasus-cnn-16-4').\n\n"
                    f"Original safetensors error:\n{e_sft}"
                ) from e_sft
            # Torch is new enough — try legacy weights as a fallback.
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                **dtype_kwargs,
            )

        self.model.to(self.device)

        # Generation knobs
        self.num_beams = int(num_beams)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.length_penalty = float(length_penalty)

        # Determine max input tokens (respect model limit if not given)
        self.max_input_tokens = (
            int(max_input_tokens)
            if max_input_tokens is not None
            else min(getattr(self.tokenizer, "model_max_length", 1024) or 1024, 2048)
        )

    # ------------- internal helpers -------------

    def _encode(self, text: str):
        return self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

    def _generate(self, inputs, max_new_tokens: int, min_length: int) -> str:
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                min_length=int(min_length),
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                length_penalty=self.length_penalty,
                early_stopping=True,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return _clean(text.strip())

    # ------------- public API -------------

    def summarize(self, text: str, max_len: int = _CFG_MAX_SUMLEN) -> str:
        """
        Summarize a single text blob (truncated to model input limit).
        """
        text = (text or "").strip()
        if not text:
            return ""
        toks = self._encode(text)
        # Heuristic: ensure a reasonable minimum
        min_len = max(60, min(220, max_len // 2))
        return self._generate(toks, max_new_tokens=int(max_len), min_length=int(min_len))

    def one_sentence(self, text: str, max_len: int = _CFG_ONE_SENT_LEN) -> str:
        """
        Force a terse 1–2 sentence gloss (good for per-paper bullets).
        """
        text = (text or "").strip()
        if not text:
            return ""
        toks = self._encode(text)
        min_len = 25  # small but avoids ultra-short fragments
        return self._generate(toks, max_new_tokens=int(max_len), min_length=min_len)

    def summarize_long(
        self,
        text: str,
        *,
        chunk_tokens: int = _CFG_CHUNK,
        overlap: int = _CFG_OVERLAP,
        reduce_max_len: int = _CFG_REDUCE_LEN,
    ) -> str:
        """
        Hierarchical (map-reduce) summarization for very long inputs:
          1) split by tokens (with overlap),
          2) summarize each chunk,
          3) summarize the concatenation of chunk-summaries.
        """
        text = (text or "").strip()
        if not text:
            return ""

        # Tokenize once
        ids = self.tokenizer.encode(text, truncation=False)
        if not ids:
            return ""

        # Build overlapping windows
        chunks: List[str] = []
        i = 0
        step = max(1, int(chunk_tokens) - max(0, int(overlap)))
        while i < len(ids):
            window = ids[i : i + int(chunk_tokens)]
            chunk_txt = self.tokenizer.decode(window, skip_special_tokens=True)
            if chunk_txt.strip():
                chunks.append(chunk_txt)
            i += step

        # Map step: per-chunk summaries
        parts: List[str] = []
        for c in chunks:
            parts.append(self.summarize(c, max_len=240))

        # Reduce step: summarize concatenated parts
        combined = " ".join(parts)
        return self.summarize(combined, max_len=int(reduce_max_len))


__all__ = ["Summarizer"]
