"""
crawler/downloader.py
---------------------
Async HTTP downloader.

Features
--------
* aiohttp-based async fetching for high concurrency.
* Exponential-backoff retry with jitter.
* User-agent rotation across a pool.
* robots.txt compliance check before every fetch.
* Politeness delay enforcement (delegated to frontier).
* Content-type and size guards to avoid downloading binary files.
* Returns a typed DownloadResult so callers can inspect status without
  catching exceptions.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from config.settings import settings
from utils.logger import get_logger
from utils.robots import fetch_robots, is_allowed, get_crawl_delay
from utils.url_utils import extract_domain

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DownloadResult:
    url: str
    final_url: str          # after redirects
    status: int             # HTTP status code; -1 on network error
    html: Optional[str]     # raw HTML body (None on failure)
    content_type: str = ""
    error: Optional[str] = None
    crawl_delay: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.html is not None


# ── Downloader ────────────────────────────────────────────────────────────────

class Downloader:
    """
    Manages a shared aiohttp session and exposes ``fetch`` for single URLs.

    Usage (as async context manager)::

        async with Downloader() as dl:
            result = await dl.fetch("https://example.com")
    """

    def __init__(self) -> None:
        cfg = settings.crawler
        dl_cfg = settings.downloader

        self._user_agents: list[str] = cfg.get("user_agents") or [
            "Mozilla/5.0 (compatible; SemanticBot/1.0)"
        ]
        self._timeout_s: int = cfg.get("request_timeout", 15)
        self._max_retries: int = cfg.get("max_retries", 3)
        self._max_content: int = dl_cfg.get("max_content_length", 5_242_880)
        self._allowed_ct: list[str] = dl_cfg.get(
            "allowed_content_types", ["text/html", "application/xhtml+xml"]
        )
        self._follow_redirects: bool = dl_cfg.get("follow_redirects", True)
        self._max_redirects: int = dl_cfg.get("max_redirects", 5)
        self._respect_robots: bool = cfg.get("respect_robots_txt", True)

        self._session: Optional[ClientSession] = None
        # robots cache: domain → RobotFileParser
        self._robots_cache: dict[str, object] = {}

    async def __aenter__(self) -> "Downloader":
        await self._start_session()
        return self

    async def __aexit__(self, *_) -> None:
        await self._close_session()

    async def _start_session(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=4,
            ssl=False,  # skip SSL verification for broad crawling
        )
        timeout = ClientTimeout(total=self._timeout_s)
        self._session = ClientSession(
            connector=connector,
            timeout=timeout,
            max_redirects=self._max_redirects if self._follow_redirects else 0,
        )

    async def _close_session(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _pick_user_agent(self) -> str:
        return random.choice(self._user_agents)

    async def _check_robots(self, url: str) -> tuple[bool, float]:
        """
        Returns (is_allowed, crawl_delay).
        Fetches robots.txt lazily and caches the result.
        """
        if not self._respect_robots:
            return True, 0.0

        domain = extract_domain(url)
        if domain not in self._robots_cache:
            parser = await fetch_robots(url, self._session)
            self._robots_cache[domain] = parser
        else:
            parser = self._robots_cache[domain]

        agent = self._pick_user_agent().split("/")[0]
        allowed = is_allowed(parser, url, agent)
        delay = get_crawl_delay(parser, agent)
        return allowed, delay

    async def fetch(self, url: str) -> DownloadResult:
        """
        Download *url* and return a DownloadResult.
        Handles retries internally with exponential backoff.
        """
        if self._session is None:
            await self._start_session()

        # robots.txt gate
        try:
            allowed, crawl_delay = await self._check_robots(url)
        except Exception:
            allowed, crawl_delay = True, 0.0

        if not allowed:
            logger.debug("robots.txt disallows: %s", url)
            return DownloadResult(
                url=url, final_url=url, status=-1, html=None,
                error="disallowed_by_robots", crawl_delay=0.0
            )

        last_error = ""
        for attempt in range(self._max_retries):
            try:
                t0 = time.monotonic()
                headers = {"User-Agent": self._pick_user_agent()}
                async with self._session.get(
                    url,
                    headers=headers,
                    allow_redirects=self._follow_redirects,
                ) as resp:
                    elapsed = (time.monotonic() - t0) * 1000

                    # Check content-type
                    ct = resp.content_type or ""
                    if not any(ct.startswith(a) for a in self._allowed_ct):
                        return DownloadResult(
                            url=url,
                            final_url=str(resp.url),
                            status=resp.status,
                            html=None,
                            content_type=ct,
                            error=f"unsupported_content_type:{ct}",
                            crawl_delay=crawl_delay,
                            elapsed_ms=elapsed,
                        )

                    # Enforce size limit
                    body = await resp.content.read(self._max_content + 1)
                    if len(body) > self._max_content:
                        return DownloadResult(
                            url=url,
                            final_url=str(resp.url),
                            status=resp.status,
                            html=None,
                            content_type=ct,
                            error="content_too_large",
                            crawl_delay=crawl_delay,
                            elapsed_ms=elapsed,
                        )

                    html = body.decode("utf-8", errors="replace")
                    logger.debug(
                        "Fetched [%d] %s in %.0fms", resp.status, url, elapsed
                    )
                    return DownloadResult(
                        url=url,
                        final_url=str(resp.url),
                        status=resp.status,
                        html=html if resp.status == 200 else None,
                        content_type=ct,
                        crawl_delay=crawl_delay,
                        elapsed_ms=elapsed,
                    )

            except asyncio.TimeoutError:
                last_error = "timeout"
            except aiohttp.ClientError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)

            # Exponential backoff with jitter before retry
            backoff = (2 ** attempt) + random.uniform(0, 1)
            logger.debug(
                "Retry %d/%d for %s after %.1fs (error: %s)",
                attempt + 1, self._max_retries, url, backoff, last_error,
            )
            await asyncio.sleep(backoff)

        logger.warning("Failed to fetch %s after %d retries: %s", url, self._max_retries, last_error)
        return DownloadResult(
            url=url, final_url=url, status=-1, html=None, error=last_error
        )
