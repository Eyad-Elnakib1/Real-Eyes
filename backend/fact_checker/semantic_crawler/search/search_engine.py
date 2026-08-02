"""
search/search_engine.py
------------------------
The public façade for the Semantic Search Engine (SSE).

Responsibilities
----------------
1. Accept a natural-language user query.
2. Decide whether to trigger a fresh crawl (cold-start or staleness).
3. Process the query (normalise + embed + expand).
4. Retrieve and rank results from the vector store.
5. Emit a typed SearchResponse that serialises to the required JSON format:

   {
     "user_query": "...",
     "evidence": [
       ["url1", similarity_score],
       ["url2", similarity_score]
     ]
   }
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from config.settings import settings
from indexer.vector_store import vector_store
from search.query_processor import QueryProcessor, ProcessedQuery
from search.retriever import Retriever, RankedResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass
class SearchResponse:
    user_query: str
    evidence: list[list]            # [[url, score], ...]
    results: list[RankedResult] = field(default_factory=list, repr=False)
    query_time_ms: float = 0.0
    total_results: int = 0
    expanded_terms: list[str] = field(default_factory=list)
    crawl_triggered: bool = False

    def to_json(self, indent: int = 2) -> str:
        """Serialise to the required output JSON format."""
        return json.dumps(
            {
                "user_query": self.user_query,
                "evidence": self.evidence,
            },
            indent=indent,
            ensure_ascii=False,
        )

    def to_full_json(self, indent: int = 2) -> str:
        """Extended JSON with snippets and metadata."""
        return json.dumps(
            {
                "user_query": self.user_query,
                "query_time_ms": round(self.query_time_ms, 1),
                "total_results": self.total_results,
                "expanded_terms": self.expanded_terms,
                "crawl_triggered": self.crawl_triggered,
                "evidence": self.evidence,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "snippet": r.snippet,
                        "score": r.score,
                        "domain": r.domain,
                    }
                    for r in self.results
                ],
            },
            indent=indent,
            ensure_ascii=False,
        )


# ── Search Engine ─────────────────────────────────────────────────────────────

class SearchEngine:
    """
    High-level semantic search engine.

    Parameters
    ----------
    auto_crawl:
        If True and the vector store is empty (or has too few results for
        the query), automatically trigger a crawl before searching.
    """

    # Minimum result count before triggering an auto-crawl
    _AUTO_CRAWL_THRESHOLD = 3

    def __init__(self, auto_crawl: bool = False) -> None:
        self._processor = QueryProcessor()
        self._retriever = Retriever()
        self._auto_crawl = auto_crawl

    # ── Primary search method ─────────────────────────────────────────────────

    def search(self, query: str, auto_crawl: Optional[bool] = None) -> SearchResponse:
        """
        Execute a semantic search for *query*.

        Parameters
        ----------
        query      : Natural language search query.
        auto_crawl : Override instance-level auto_crawl setting.
        """
        t0 = time.monotonic()
        do_crawl = auto_crawl if auto_crawl is not None else self._auto_crawl

        # 1. Process query
        try:
            pq: ProcessedQuery = self._processor.process(query)
        except ValueError as exc:
            logger.warning("Query processing failed: %s", exc)
            return self._empty_response(query, elapsed_ms=0.0)

        crawl_triggered = False

        # 2. Check if we need to crawl first (count-only check — no retrieve yet)
        if do_crawl and vector_store.count() < self._AUTO_CRAWL_THRESHOLD:
            logger.info(
                "Index too small (%d chunks) — triggering auto-crawl for '%s'",
                vector_store.count(), query,
            )
            self._trigger_crawl(pq)
            crawl_triggered = True

        # 3. Retrieve
        results: list[RankedResult] = self._retriever.retrieve(pq)

        elapsed_ms = (time.monotonic() - t0) * 1000

        # 4. Build evidence list: [[url, score], ...]
        evidence = [[r.url, r.score] for r in results]

        response = SearchResponse(
            user_query=query,
            evidence=evidence,
            results=results,
            query_time_ms=elapsed_ms,
            total_results=len(results),
            expanded_terms=pq.expanded_terms,
            crawl_triggered=crawl_triggered,
        )

        logger.info(
            "Search '%s' → %d results in %.0fms%s",
            query, len(results), elapsed_ms,
            " (crawl triggered)" if crawl_triggered else "",
        )
        return response

    # ── Trigger crawl (sync wrapper around async crawler) ────────────────────

    def _trigger_crawl(self, pq: ProcessedQuery) -> None:
        """
        Synchronously run a small focused crawl for *pq* to seed the
        vector store.  Runs in a dedicated event loop.
        """
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._async_crawl(pq))
            loop.close()
        except Exception as exc:
            logger.error("Auto-crawl failed: %s", exc, exc_info=True)

    async def _async_crawl(self, pq: ProcessedQuery) -> None:
        from queue import Queue
        from crawler.crawler_manager import CrawlerManager
        from indexer.indexer import Indexer

        content_queue: Queue = Queue(maxsize=200)
        crawler = CrawlerManager(content_queue=content_queue)
        indexer = Indexer(content_queue=content_queue)

        indexer.start()
        await crawler.crawl(
            query=pq.raw,
            # Use a small page cap for auto-crawl to keep latency acceptable
        )
        indexer.stop(timeout=120.0)

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def _empty_response(query: str, elapsed_ms: float) -> SearchResponse:
        return SearchResponse(
            user_query=query,
            evidence=[],
            query_time_ms=elapsed_ms,
        )
