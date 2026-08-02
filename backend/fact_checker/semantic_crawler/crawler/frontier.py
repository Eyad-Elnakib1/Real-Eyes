"""
crawler/frontier.py
--------------------
URL Frontier — the heart of the crawl scheduler.

Design
------
* Priority queue ordered by (priority DESC, depth ASC, discovered_at ASC).
  Higher priority → crawled first.
* Deduplication via an in-memory set of URL fingerprints backed by SQLite.
* Domain-level politeness: a per-domain "next_allowed_at" timestamp ensures
  we respect Crawl-delay and the configured default_delay.
* Two backends:
    "disk"  — SQLite (default, zero extra dependencies)
    "redis" — Redis sorted-set (for multi-worker distributed crawling)

Priority scoring
----------------
score = base_relevance * depth_decay * freshness_bonus
  base_relevance  = relevance score from classifier (0–1)
  depth_decay     = 1 / (1 + depth)  (shallower pages rank higher)
  freshness_bonus = 1.0 if URL never seen, 0.5 if re-crawl candidate
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from config.settings import settings
from utils.logger import get_logger
from utils.trust import crawl_trust_boost
from utils.url_utils import url_fingerprint, extract_domain

logger = get_logger(__name__)


# ── Frontier item ─────────────────────────────────────────────────────────────

@dataclass(order=True)
class FrontierItem:
    priority: float          # higher = more important (we negate for heap)
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    parent_url: Optional[str] = field(compare=False, default=None)
    discovered_at: float = field(compare=False, default_factory=time.time)
    relevance_score: float = field(compare=False, default=0.5)

    @classmethod
    def from_seed(cls, url: str) -> "FrontierItem":
        return cls(priority=1.0, url=url, depth=0, relevance_score=1.0)

    @classmethod
    def from_link(
        cls,
        url: str,
        depth: int,
        relevance_score: float,
        parent_url: Optional[str] = None,
    ) -> "FrontierItem":
        # depth decay: shallower links get higher priority
        depth_decay = 1.0 / (1.0 + depth)
        # trust boost: trusted domains (.gov/.edu/fact-checkers) crawled first
        trust_boost = crawl_trust_boost(url)
        priority = relevance_score * depth_decay * trust_boost
        return cls(
            priority=priority,
            url=url,
            depth=depth,
            parent_url=parent_url,
            relevance_score=relevance_score,
        )


# ── Base frontier ─────────────────────────────────────────────────────────────

class BaseFrontier:
    def push(self, item: FrontierItem) -> bool:
        """Add item. Returns True if accepted (not duplicate)."""
        raise NotImplementedError

    def pop(self) -> Optional[FrontierItem]:
        """Return highest-priority item that respects politeness, or None."""
        raise NotImplementedError

    def seen(self, url: str) -> bool:
        raise NotImplementedError

    def mark_done(self, url: str) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # Politeness helpers
    def get_next_allowed(self, domain: str) -> float:
        raise NotImplementedError

    def set_next_allowed(self, domain: str, ts: float) -> None:
        raise NotImplementedError


# ── SQLite-backed frontier ────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier_queue (
    fingerprint     TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    priority        REAL NOT NULL,
    depth           INTEGER DEFAULT 0,
    parent_url      TEXT,
    discovered_at   REAL,
    relevance_score REAL DEFAULT 0.5,
    status          TEXT DEFAULT 'pending'  -- pending | done | failed
);
CREATE TABLE IF NOT EXISTS seen_urls (
    fingerprint TEXT PRIMARY KEY,
    added_at    REAL
);
CREATE TABLE IF NOT EXISTS domain_politeness (
    domain          TEXT PRIMARY KEY,
    next_allowed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_priority ON frontier_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_status   ON frontier_queue(status);
"""


class SQLiteFrontier(BaseFrontier):
    """
    Persistent frontier backed by SQLite.
    Thread-safe via a single threading.Lock.
    """

    def __init__(self, db_path: str, max_size: int = 100_000) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._max_size = max_size
        self._lock = Lock()
        self._default_delay: float = settings.crawler.get("default_delay", 1.0)
        self._init_db()
        logger.info("SQLiteFrontier ready at %s (max_size=%d)", db_path, max_size)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def push(self, item: FrontierItem) -> bool:
        fp = url_fingerprint(item.url)
        with self._lock:
            with self._conn() as conn:
                # Duplicate check
                row = conn.execute(
                    "SELECT 1 FROM seen_urls WHERE fingerprint=?", (fp,)
                ).fetchone()
                if row:
                    return False

                # Enforce max queue size
                q_size = conn.execute(
                    "SELECT COUNT(*) FROM frontier_queue WHERE status='pending'"
                ).fetchone()[0]
                if q_size >= self._max_size:
                    logger.debug("Frontier full (%d), dropping %s", q_size, item.url)
                    return False

                conn.execute(
                    """INSERT OR IGNORE INTO seen_urls (fingerprint, added_at)
                       VALUES (?, ?)""",
                    (fp, time.time()),
                )
                conn.execute(
                    """INSERT OR REPLACE INTO frontier_queue
                       (fingerprint, url, priority, depth, parent_url,
                        discovered_at, relevance_score, status)
                       VALUES (?,?,?,?,?,?,?,'pending')""",
                    (
                        fp, item.url, item.priority, item.depth,
                        item.parent_url, item.discovered_at, item.relevance_score,
                    ),
                )
        return True

    def pop(self) -> Optional[FrontierItem]:
        """
        Return the highest-priority pending item whose domain politeness
        allows crawling right now, or None if none available.
        """
        now = time.time()
        with self._lock:
            with self._conn() as conn:
                # Fetch candidates ordered by priority desc
                rows = conn.execute(
                    """SELECT * FROM frontier_queue
                       WHERE status='pending'
                       ORDER BY priority DESC
                       LIMIT 50"""
                ).fetchall()

                for row in rows:
                    domain = extract_domain(row["url"])
                    next_allowed = self._get_next_allowed_unsafe(conn, domain)
                    if now >= next_allowed:
                        # Reserve the item
                        conn.execute(
                            "UPDATE frontier_queue SET status='processing' WHERE fingerprint=?",
                            (row["fingerprint"],),
                        )
                        return FrontierItem(
                            priority=row["priority"],
                            url=row["url"],
                            depth=row["depth"],
                            parent_url=row["parent_url"],
                            discovered_at=row["discovered_at"],
                            relevance_score=row["relevance_score"],
                        )
        return None

    def mark_done(self, url: str) -> None:
        fp = url_fingerprint(url)
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE frontier_queue SET status='done' WHERE fingerprint=?",
                    (fp,),
                )

    def mark_failed(self, url: str) -> None:
        fp = url_fingerprint(url)
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE frontier_queue SET status='failed' WHERE fingerprint=?",
                    (fp,),
                )

    def seen(self, url: str) -> bool:
        fp = url_fingerprint(url)
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM seen_urls WHERE fingerprint=?", (fp,)
                ).fetchone()
            return row is not None

    def size(self) -> int:
        with self._lock:
            with self._conn() as conn:
                return conn.execute(
                    "SELECT COUNT(*) FROM frontier_queue WHERE status='pending'"
                ).fetchone()[0]

    def _get_next_allowed_unsafe(
        self, conn: sqlite3.Connection, domain: str
    ) -> float:
        row = conn.execute(
            "SELECT next_allowed_at FROM domain_politeness WHERE domain=?",
            (domain,),
        ).fetchone()
        return row["next_allowed_at"] if row else 0.0

    def get_next_allowed(self, domain: str) -> float:
        with self._lock:
            with self._conn() as conn:
                return self._get_next_allowed_unsafe(conn, domain)

    def set_next_allowed(self, domain: str, ts: float) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO domain_politeness
                       (domain, next_allowed_at) VALUES (?,?)""",
                    (domain, ts),
                )

    def update_politeness(self, domain: str, delay: float) -> None:
        """Call after each successful fetch to advance the domain clock."""
        effective_delay = max(delay, self._default_delay)
        next_ts = time.time() + effective_delay
        self.set_next_allowed(domain, next_ts)

    def close(self) -> None:
        logger.info("SQLiteFrontier closed")


# ── Factory ───────────────────────────────────────────────────────────────────

def build_frontier() -> BaseFrontier:
    cfg = settings.frontier
    backend = cfg.get("backend", "disk")
    max_size = cfg.get("max_queue_size", 100_000)

    if backend == "disk":
        db_path = cfg.get("disk_path", "data/frontier.db")
        return SQLiteFrontier(db_path=db_path, max_size=max_size)

    # Future: Redis implementation for distributed mode
    raise ValueError(f"Unsupported frontier backend: {backend}")
