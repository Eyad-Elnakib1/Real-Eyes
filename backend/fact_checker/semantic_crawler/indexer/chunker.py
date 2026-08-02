"""
indexer/chunker.py
-------------------
Split cleaned document text into overlapping semantic chunks suitable for
embedding and vector-store indexing.

Three strategies
----------------
"fixed"     — Split on a fixed character/token count with overlap.
              Fast, deterministic, ignores sentence boundaries.
"sentence"  — Split on sentence boundaries using NLTK, then group sentences
              into chunks of ≤ chunk_size tokens.  Best balance of speed
              and quality.  This is the recommended default.
"semantic"  — Sentence-level grouping, but merge sentences as long as the
              cosine similarity of consecutive sentences remains high.
              Most expensive (loads a second model) but produces the most
              coherent chunks.

All strategies share the same output format: a list of TextChunk objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = len(self.text.split())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rough_token_count(text: str) -> int:
    return len(text.split())


def _split_sentences_regex(text: str) -> list[str]:
    """Regex sentence splitter — fast fallback, no NLTK needed."""
    pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _split_sentences_nltk(text: str) -> list[str]:
    """
    NLTK Punkt sentence splitter.

    NLTK 3.8 renamed the punkt data from
        tokenizers/punkt/english.pickle   (old)
    to
        tokenizers/punkt_tab/english/     (new)

    We try both paths and download whichever is needed.
    """
    try:
        import nltk  # type: ignore

        # --- Try new path first (NLTK >= 3.8) ---
        tokenizer = None
        try:
            tokenizer = nltk.data.load("tokenizers/punkt_tab/english/")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
                tokenizer = nltk.data.load("tokenizers/punkt_tab/english/")
            except Exception:
                pass

        # --- Fall back to old path (NLTK < 3.8) ---
        if tokenizer is None:
            try:
                tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            except LookupError:
                nltk.download("punkt", quiet=True)
                tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

        return tokenizer.tokenize(text)

    except Exception as exc:
        logger.debug("NLTK sentence splitting failed (%s), using regex.", exc)
        return _split_sentences_regex(text)


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 100,
) -> list[TextChunk]:
    """Split text into fixed-size character chunks with overlap."""
    chunks: list[TextChunk] = []
    stride = max(chunk_size - overlap, 1)
    idx = 0
    chunk_n = 0

    while idx < len(text):
        end = min(idx + chunk_size, len(text))
        chunk_text = text[idx:end].strip()
        if len(chunk_text) >= min_length:
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_n,
                    start_char=idx,
                    end_char=end,
                )
            )
            chunk_n += 1
        idx += stride

    return chunks


def chunk_sentences(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 100,
) -> list[TextChunk]:
    """
    Group sentences into chunks of ≤ chunk_size tokens, with sentence-level
    overlap.  Uses NLTK punkt (with automatic fallback to regex).
    """
    sentences = _split_sentences_nltk(text)
    if not sentences:
        return chunk_fixed(text, chunk_size, overlap, min_length)

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_tokens = 0
    chunk_n = 0
    char_ptr = 0
    overlap_sents = max(1, overlap // 20)

    for sent in sentences:
        t = _rough_token_count(sent)
        if current_tokens + t > chunk_size and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= min_length:
                chunks.append(
                    TextChunk(text=chunk_text, chunk_index=chunk_n,
                               start_char=char_ptr)
                )
                chunk_n += 1
            current = current[-overlap_sents:]
            current_tokens = sum(_rough_token_count(s) for s in current)
        current.append(sent)
        current_tokens += t

    if current:
        chunk_text = " ".join(current).strip()
        if len(chunk_text) >= min_length:
            chunks.append(
                TextChunk(text=chunk_text, chunk_index=chunk_n,
                           start_char=char_ptr)
            )

    return chunks


def chunk_semantic(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 100,
) -> list[TextChunk]:
    """
    Semantic chunking: merge consecutive sentences while their embedding
    similarity remains above a threshold, then start a new chunk.

    Falls back to sentence chunking if embeddings are unavailable.
    """
    try:
        import numpy as np  # type: ignore

        model = _get_cached_model(
            settings.embedder.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        sentences = _split_sentences_nltk(text)
        if len(sentences) < 2:
            return chunk_sentences(text, chunk_size, overlap, min_length)

        embs = model.encode(sentences, batch_size=64, show_progress_bar=False)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

        chunks: list[TextChunk] = []
        current_sents: list[str] = [sentences[0]]
        current_tokens = _rough_token_count(sentences[0])
        chunk_n = 0
        SIM_THRESHOLD = 0.75

        for i in range(1, len(sentences)):
            sim = float(np.dot(embs[i - 1], embs[i]))
            t = _rough_token_count(sentences[i])
            if sim < SIM_THRESHOLD or current_tokens + t > chunk_size:
                chunk_text = " ".join(current_sents).strip()
                if len(chunk_text) >= min_length:
                    chunks.append(TextChunk(text=chunk_text, chunk_index=chunk_n))
                    chunk_n += 1
                overlap_sents = max(1, overlap // 20)
                current_sents = current_sents[-overlap_sents:] + [sentences[i]]
                current_tokens = sum(_rough_token_count(s) for s in current_sents)
            else:
                current_sents.append(sentences[i])
                current_tokens += t

        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) >= min_length:
                chunks.append(TextChunk(text=chunk_text, chunk_index=chunk_n))

        return chunks or chunk_sentences(text, chunk_size, overlap, min_length)

    except Exception as exc:
        logger.debug("Semantic chunking failed (%s), falling back to sentence.", exc)
        return chunk_sentences(text, chunk_size, overlap, min_length)


_model_cache: dict[str, object] = {}


def _get_cached_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ── Public API ────────────────────────────────────────────────────────────────

class Chunker:
    """Strategy-aware text chunker. Reads settings from config."""

    def __init__(self) -> None:
        cfg = settings.chunker
        self._strategy: str = cfg.get("strategy", "sentence")
        self._chunk_size: int = cfg.get("chunk_size", 512)
        self._overlap: int = cfg.get("chunk_overlap", 64)
        self._min_length: int = cfg.get("min_chunk_length", 100)

    def chunk(self, text: str) -> list[TextChunk]:
        """Chunk *text* using the configured strategy."""
        if not text or len(text) < self._min_length:
            return []

        if self._strategy == "fixed":
            chunks = chunk_fixed(text, self._chunk_size, self._overlap, self._min_length)
        elif self._strategy == "semantic":
            chunks = chunk_semantic(text, self._chunk_size, self._overlap, self._min_length)
        else:  # "sentence" — default
            chunks = chunk_sentences(text, self._chunk_size, self._overlap, self._min_length)

        logger.debug("Chunked text into %d chunks (strategy=%s)", len(chunks), self._strategy)
        return chunks
