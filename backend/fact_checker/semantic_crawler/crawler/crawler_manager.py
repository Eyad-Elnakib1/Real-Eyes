"""
crawler/crawler_manager.py
---------------------------
Orchestrates the full crawl pipeline:

    Frontier → Downloader → Extractor → LinkExtractor → Frontier (cycle)
                                      ↓
                              ContentQueue → Indexer

Design
------
* N async workers pull items from the frontier concurrently.
* Each worker uses its own aiohttp session (but shares a common
  Downloader instance for config).
* After extraction, content is placed on a thread-safe queue that the
  indexer consumes in a separate thread.
* The manager exposes ``crawl(query, seed_urls)`` which runs until the
  frontier is exhausted or the page cap is reached.
* Incremental mode skips URLs already indexed in the metadata store.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Optional

import aiohttp

from config.settings import settings
from crawler.downloader import Downloader, DownloadResult
from crawler.extractor import extract_content, ExtractedContent
from crawler.frontier import BaseFrontier, FrontierItem, build_frontier
from crawler.link_extractor import LinkExtractor
from crawler.query_expander import QueryExpander
from storage.cache import cache
from storage.metadata_store import metadata_store, PageMeta
from utils.logger import get_logger
from utils.url_utils import canonicalise_url, extract_domain, url_fingerprint

logger = get_logger(__name__)


# ── Content item sent to the indexer ─────────────────────────────────────────

@dataclass
class CrawledPage:
    url: str
    final_url: str
    title: str
    text: str
    depth: int
    relevance_score: float
    crawl_timestamp: float
    published_date: str = ""


# ── CrawlerManager ────────────────────────────────────────────────────────────

class CrawlerManager:
    """
    High-level orchestrator for the semantic web crawler.

    Parameters
    ----------
    content_queue:
        Thread-safe queue into which CrawledPage objects are placed.
        The indexer thread reads from this queue.
    frontier:
        Optional pre-built frontier (for testing / re-use).
    """

    def __init__(
        self,
        content_queue: Optional[Queue] = None,
        frontier: Optional[BaseFrontier] = None,
    ) -> None:
        cfg = settings.crawler

        self._max_workers: int = cfg.get("max_workers", 4)
        self._max_pages: int = cfg.get("max_pages", 5000)
        self._max_depth: int = cfg.get("max_depth", 5)
        self._default_delay: float = cfg.get("default_delay", 1.0)
        self._incremental: bool = cfg.get("incremental", True)

        self._frontier: BaseFrontier = frontier or build_frontier()
        self._content_queue: Queue = content_queue or Queue(maxsize=500)
        self._link_extractor = LinkExtractor()
        self._query_expander = QueryExpander()

        self._pages_crawled: int = 0
        self._pages_failed: int = 0
        self._start_time: float = 0.0
        self._query_terms: list[str] = []

        # Semaphore shared across workers to limit concurrent domain fetches
        self._sem: Optional[asyncio.Semaphore] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl(self, query: str, seed_urls: Optional[list[str]] = None) -> int:
        """
        Start crawling.  Blocks until frontier is exhausted or max_pages is
        reached.  Returns number of pages successfully crawled.

        Parameters
        ----------
        query     : natural language crawl goal (used for relevance scoring)
        seed_urls : initial URLs; if None, derived from query via search
        """
        self._start_time = time.monotonic()
        self._query_terms = self._query_expander.get_query_terms(query)
        self._link_extractor.set_query_terms(self._query_terms)

        # Obtain seed URLs from search if none provided
        if not seed_urls:
            logger.info("No seed URLs provided — searching for '%s'", query)
            expanded = self._query_expander.expand(query)
            seed_urls = self._query_expander.get_seed_urls(query, expanded)

        if not seed_urls:
            logger.error("Could not find any seed URLs for query: '%s'", query)
            return 0

        # Push seeds into frontier
        for url in seed_urls:
            canonical = canonicalise_url(url)
            if canonical:
                self._frontier.push(FrontierItem.from_seed(canonical))

        logger.info(
            "Starting crawl: query='%s', seeds=%d, max_pages=%d, workers=%d",
            query, len(seed_urls), self._max_pages, self._max_workers,
        )

        self._sem = asyncio.Semaphore(self._max_workers * 2)

        # Launch N worker coroutines
        workers = [
            asyncio.create_task(self._worker(worker_id=i))
            for i in range(self._max_workers)
        ]

        await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.monotonic() - self._start_time
        logger.info(
            "Crawl complete: %d pages in %.1fs (%.1f pages/s), %d failed",
            self._pages_crawled, elapsed,
            self._pages_crawled / max(elapsed, 1),
            self._pages_failed,
        )
        return self._pages_crawled

    # ── Worker coroutine ──────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """Single async crawl worker."""
        logger.debug("Worker %d started", worker_id)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False, limit_per_host=2),
            timeout=aiohttp.ClientTimeout(total=settings.crawler.get("request_timeout", 15)),
        ) as session:
            downloader = _SessionDownloader(session)

            while True:
                # Global page cap
                if self._pages_crawled >= self._max_pages:
                    break

                item = self._frontier.pop()
                if item is None:
                    # Nothing available right now — wait and retry
                    await asyncio.sleep(0.5)
                    # If frontier truly empty for a while, stop
                    if self._frontier.size() == 0:
                        await asyncio.sleep(2)
                        if self._frontier.size() == 0:
                            break
                    continue

                if item.depth > self._max_depth:
                    self._frontier.mark_done(item.url)
                    continue

                # Incremental: skip already-indexed URLs
                if self._incremental and metadata_store.is_indexed(item.url):
                    logger.debug("Skipping already-indexed: %s", item.url)
                    self._frontier.mark_done(item.url)
                    continue

                async with self._sem:
                    await self._process_url(downloader, item)

        logger.debug("Worker %d finished", worker_id)

    async def _process_url(
        self, downloader: "_SessionDownloader", item: FrontierItem
    ) -> None:
        """Download, extract, score, and enqueue links for one URL."""
        url = item.url
        domain = extract_domain(url)

        result: DownloadResult = await downloader.fetch(url)

        # Update domain politeness
        delay = max(result.crawl_delay, self._default_delay)
        self._frontier.update_politeness(domain, delay)

        if not result.ok:
            self._pages_failed += 1
            self._frontier.mark_failed(url)
            # Store failed attempt in metadata
            meta = PageMeta(
                url=url, http_status=result.status,
                depth=item.depth, relevance_score=item.relevance_score,
            )
            metadata_store.upsert(meta)
            return

        # Extract content
        content: Optional[ExtractedContent] = extract_content(
            result.html, result.final_url
        )

        if content is None or content.is_empty:
            self._frontier.mark_done(url)
            return

        # Persist metadata
        meta = PageMeta(
            url=result.final_url,
            title=content.title,
            http_status=200,
            depth=item.depth,
            relevance_score=item.relevance_score,
            content_hash=PageMeta.compute_content_hash(content.text),
        )
        metadata_store.upsert(meta)

        # Cache raw HTML for potential re-processing
        cache.set(f"html:{url_fingerprint(url)}", result.html[:50_000], ttl=3600)

        # Emit to indexer queue (non-blocking; drop if full)
        page = CrawledPage(
            url=result.final_url,
            final_url=result.final_url,
            title=content.title,
            text=content.text,
            depth=item.depth,
            relevance_score=item.relevance_score,
            crawl_timestamp=time.time(),
            published_date=getattr(content, "published_date", "") or "",
        )
        try:
            self._content_queue.put_nowait(page)
        except Exception:
            pass  # queue full; indexer will catch up

        self._pages_crawled += 1
        self._frontier.mark_done(url)

        # Extract and enqueue links
        if item.depth < self._max_depth:
            links = self._link_extractor.extract(
                result.html, result.final_url, self._query_terms
            )
            for link in links:
                canonical = canonicalise_url(link.url)
                if canonical and not self._frontier.seen(canonical):
                    fi = FrontierItem.from_link(
                        url=canonical,
                        depth=item.depth + 1,
                        relevance_score=link.relevance_score,
                        parent_url=result.final_url,
                    )
                    self._frontier.push(fi)

        logger.info(
            "[%d/%d] Crawled: %s (%d words)",
            self._pages_crawled, self._max_pages, url, content.word_count,
        )

    @property
    def content_queue(self) -> Queue:
        return self._content_queue

    def stats(self) -> dict:
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        return {
            "pages_crawled": self._pages_crawled,
            "pages_failed": self._pages_failed,
            "frontier_size": self._frontier.size(),
            "elapsed_seconds": round(elapsed, 1),
            "pages_per_second": round(
                self._pages_crawled / max(elapsed, 1), 2
            ),
        }


# ── Thin session wrapper ──────────────────────────────────────────────────────

class _SessionDownloader:
    """Wraps an aiohttp.ClientSession with the Downloader's fetch logic."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session
        cfg = settings.crawler
        self._user_agents: list[str] = cfg.get("user_agents") or [
            "Mozilla/5.0 (compatible; SemanticBot/1.0)"
        ]
        self._max_content = settings.downloader.get("max_content_length", 5_242_880)
        self._allowed_ct = settings.downloader.get(
            "allowed_content_types", ["text/html", "application/xhtml+xml"]
        )

    async def fetch(self, url: str) -> DownloadResult:
        import random
        ua = random.choice(self._user_agents)
        try:
            async with self._session.get(
                url, headers={"User-Agent": ua}, allow_redirects=True
            ) as resp:
                ct = resp.content_type or ""
                if not any(ct.startswith(a) for a in self._allowed_ct):
                    return DownloadResult(url=url, final_url=str(resp.url),
                                          status=resp.status, html=None,
                                          error=f"bad_ct:{ct}")
                body = await resp.content.read(self._max_content)
                html = body.decode("utf-8", errors="replace")
                return DownloadResult(url=url, final_url=str(resp.url),
                                      status=resp.status, html=html if resp.status == 200 else None)
        except asyncio.TimeoutError:
            return DownloadResult(url=url, final_url=url, status=-1, html=None, error="timeout")
        except Exception as exc:
            return DownloadResult(url=url, final_url=url, status=-1, html=None, error=str(exc))

    def update_politeness(self, domain: str, delay: float) -> None:
        pass  # Politeness is managed by the frontier
