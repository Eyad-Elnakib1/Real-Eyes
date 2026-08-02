"""
utils/robots.py
---------------
Async-friendly robots.txt fetcher and rule checker.

We cache parsed robots.txt objects in memory (per domain) with a TTL so
we don't re-fetch on every request while still respecting updates.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from utils.logger import get_logger

logger = get_logger(__name__)

# ── in-memory cache entry ────────────────────────────────────────────────────

_CACHE_TTL = 3600  # 1 hour


class _CacheEntry:
    def __init__(self, parser: Optional[RobotFileParser], ts: float) -> None:
        self.parser = parser
        self.ts = ts

    def is_stale(self) -> bool:
        return (time.monotonic() - self.ts) > _CACHE_TTL


# module-level cache: domain -> _CacheEntry
_cache: dict[str, _CacheEntry] = {}
_lock = asyncio.Lock()


# ── public helpers ────────────────────────────────────────────────────────────


async def fetch_robots(domain_url: str, session) -> Optional[RobotFileParser]:
    """
    Fetch and parse robots.txt for the given domain URL.
    Returns a RobotFileParser or None if unavailable.
    *session* is an aiohttp.ClientSession.
    """
    parsed = urlparse(domain_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    domain = parsed.netloc

    async with _lock:
        entry = _cache.get(domain)
        if entry and not entry.is_stale():
            return entry.parser

    # Fetch outside the lock to avoid blocking other coroutines
    parser: Optional[RobotFileParser] = None
    try:
        async with session.get(robots_url, timeout=10) as resp:
            if resp.status == 200:
                content = await resp.text(errors="replace")
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.read()
                # Manually feed the content since read() does a network call
                # We override the internal feed
                parser.parse(content.splitlines())
    except Exception as exc:
        logger.debug("Could not fetch robots.txt for %s: %s", domain, exc)

    async with _lock:
        _cache[domain] = _CacheEntry(parser, time.monotonic())

    return parser


def is_allowed(robots_parser: Optional[RobotFileParser], url: str, agent: str = "*") -> bool:
    """
    Check whether *agent* is allowed to fetch *url* according to *robots_parser*.
    Returns True (allowed) when parser is None (no robots.txt available).
    """
    if robots_parser is None:
        return True
    try:
        return robots_parser.can_fetch(agent, url)
    except Exception:
        return True


def get_crawl_delay(robots_parser: Optional[RobotFileParser], agent: str = "*") -> float:
    """
    Return the Crawl-delay directive value for *agent*, or 0.0 if not set.
    """
    if robots_parser is None:
        return 0.0
    try:
        delay = robots_parser.crawl_delay(agent)
        return float(delay) if delay is not None else 0.0
    except Exception:
        return 0.0
