"""
indexer/embedder.py
--------------------
Generates dense vector embeddings for text chunks.

Wraps sentence-transformers with:
* Configurable model (default: all-MiniLM-L6-v2, 384 dims, fast + good).
* Batch processing for throughput.
* L2-normalisation (required for cosine similarity via dot-product).
* Result caching (embeddings are expensive; avoid re-computing the same text).
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np

from config.settings import settings
from storage.cache import cache
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thread-safe sentence-transformer wrapper.

    Embeddings are L2-normalised so cosine similarity = dot product.
    """

    def __init__(self) -> None:
        cfg = settings.embedder
        self._model_name: str = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self._batch_size: int = cfg.get("batch_size", 64)
        self._device: str = cfg.get("device", "cpu")
        self._normalise: bool = cfg.get("normalize", True)

        self._model = None
        self._lock = threading.Lock()
        self._dim: Optional[int] = None

    def _load(self) -> None:
        """Lazily load the model on first use."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self._model_name, device=self._device)
                # Warm up with dummy input to determine embedding dimension
                dummy = self._model.encode(["hello"], show_progress_bar=False)
                self._dim = dummy.shape[1]
                logger.info(
                    "EmbeddingModel loaded: %s  dim=%d  device=%s",
                    self._model_name, self._dim, self._device,
                )
            except Exception as exc:
                logger.error("Failed to load embedding model: %s", exc)
                raise

    @property
    def dimension(self) -> int:
        self._load()
        return self._dim or 384

    def _cache_key(self, text: str) -> str:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"emb:{h}"

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text string.  Returns (dim,) numpy array."""
        self._load()
        key = self._cache_key(text)
        cached = cache.get(key)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

        vec = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._normalise,
        )[0]
        cache.set(key, vec.tolist(), ttl=86400)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.  Returns (N, dim) numpy array.
        Checks cache for each text individually; only calls model for misses.
        """
        self._load()

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = cache.get(key)
            if cached is not None:
                results[i] = np.array(cached, dtype=np.float32)
            else:
                uncached_indices.append(i)

        if uncached_indices:
            batch_texts = [texts[i] for i in uncached_indices]
            batch_vecs = self._model.encode(
                batch_texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=self._normalise,
            )
            for j, idx in enumerate(uncached_indices):
                vec = batch_vecs[j].astype(np.float32)
                results[idx] = vec
                key = self._cache_key(texts[idx])
                cache.set(key, vec.tolist(), ttl=86400)

        # Stack all results
        return np.stack([r for r in results], axis=0)  # type: ignore


# Module-level singleton
embedding_model = EmbeddingModel()
