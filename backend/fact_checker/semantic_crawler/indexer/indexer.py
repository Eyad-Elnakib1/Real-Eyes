"""
indexer/indexer.py
-------------------
Consumes CrawledPage objects from the crawler's content queue, chunks the
text, generates embeddings, and writes everything to the vector store.

Runs in its own thread so it operates in parallel with the crawler workers.

Design
------
* Processes pages from a thread-safe Queue in a tight loop.
* Batches embeddings (up to EMBED_BATCH chunks) for efficiency.
* Deduplicates by content hash before inserting to avoid re-indexing the
  same text (content drift detection).
* Marks pages as indexed in the metadata store after success.
* Gracefully drains the queue on shutdown (sentinel None value).
"""

from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional

import numpy as np

from crawler.crawler_manager import CrawledPage
from indexer.chunker import Chunker, TextChunk
from indexer.embedder import EmbeddingModel, embedding_model
from indexer.vector_store import BaseVectorStore, build_vector_store, vector_store
from storage.metadata_store import metadata_store
from utils.logger import get_logger
from utils.url_utils import extract_domain

logger = get_logger(__name__)

EMBED_BATCH = 64    # max chunks to embed in one call
INDEX_SENTINEL = None   # poison pill to stop the indexer thread


class Indexer:
    """
    Background indexer that processes pages from a queue.

    Usage::

        indexer = Indexer(content_queue)
        indexer.start()
        # ... crawler runs ...
        indexer.stop()
    """

    def __init__(
        self,
        content_queue: Queue,
        store: Optional[BaseVectorStore] = None,
        embedder: Optional[EmbeddingModel] = None,
    ) -> None:
        self._queue = content_queue
        self._store = store or vector_store
        self._embedder = embedder or embedding_model
        self._chunker = Chunker()

        self._stop_event = Event()
        self._thread: Optional[Thread] = None

        self._pages_indexed: int = 0
        self._chunks_indexed: int = 0
        self._pages_skipped: int = 0
        self._errors: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the indexer background thread."""
        self._thread = Thread(target=self._run, name="IndexerThread", daemon=True)
        self._thread.start()
        logger.info("Indexer thread started")

    def stop(self, timeout: float = 60.0) -> None:
        """
        Signal the indexer to stop after draining remaining queue items.
        Waits up to *timeout* seconds for the thread to finish.
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(
            "Indexer stopped: %d pages indexed, %d chunks, %d skipped, %d errors",
            self._pages_indexed, self._chunks_indexed,
            self._pages_skipped, self._errors,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Consumer loop running in the background thread."""
        logger.debug("Indexer loop started")
        while True:
            try:
                page: Optional[CrawledPage] = self._queue.get(timeout=2.0)
            except Empty:
                if self._stop_event.is_set():
                    # Drain complete
                    break
                continue

            if page is INDEX_SENTINEL:
                break

            try:
                self._index_page(page)
            except Exception as exc:
                self._errors += 1
                logger.error("Indexer error for %s: %s", page.url, exc, exc_info=True)
            finally:
                self._queue.task_done()

    # ── Page processing ───────────────────────────────────────────────────────

    def _index_page(self, page: CrawledPage) -> None:
        """Chunk, embed, and store a single CrawledPage."""
        # Incremental: skip if content hash unchanged
        existing = metadata_store.get(page.url)
        content_hash = _hash_text(page.text)
        if existing and existing.indexed and existing.content_hash == content_hash:
            self._pages_skipped += 1
            logger.debug("Skipping unchanged: %s", page.url)
            return

        # Chunk
        chunks: list[TextChunk] = self._chunker.chunk(page.text)
        if not chunks:
            logger.debug("No chunks for %s", page.url)
            return

        # Embed in batches
        chunk_texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)  # (N, dim)

        # Build metadata for each chunk
        metadatas = [
            {
                "url": page.url,
                "title": page.title[:200] if page.title else "",
                "domain": extract_domain(page.url),
                "chunk_index": c.chunk_index,
                "depth": page.depth,
                "relevance_score": round(page.relevance_score, 4),
                "crawl_timestamp": page.crawl_timestamp,
                "published_date": getattr(page, "published_date", "") or "",
                "content_hash": content_hash,
            }
            for c in chunks
        ]

        # Persist to vector store
        ids = self._store.add(
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Mark as indexed in metadata store
        metadata_store.mark_indexed(page.url)

        self._pages_indexed += 1
        self._chunks_indexed += len(chunks)

        logger.info(
            "Indexed: %s → %d chunks (total: %d pages / %d chunks)",
            page.url, len(chunks),
            self._pages_indexed, self._chunks_indexed,
        )

    # ── Index-one helper (synchronous, for testing / API triggers) ────────────

    def index_page_sync(self, page: CrawledPage) -> int:
        """
        Index a single page synchronously (no queue).
        Returns number of chunks indexed.
        """
        before = self._chunks_indexed
        self._index_page(page)
        return self._chunks_indexed - before

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "pages_indexed": self._pages_indexed,
            "chunks_indexed": self._chunks_indexed,
            "pages_skipped": self._pages_skipped,
            "errors": self._errors,
            "vector_store_count": self._store.count(),
        }


# ── Helper ────────────────────────────────────────────────────────────────────

def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:32]
