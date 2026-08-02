"""
storage/metadata_store.py
--------------------------
Persistent metadata store for crawled documents.

Stores per-URL metadata:
    url, fingerprint, title, domain, crawl_timestamp, http_status,
    content_hash (for change detection), indexed (bool)

Two backends:
    "sqlite"  — local SQLite via the standard library (default)
    "mongodb" — pymongo (for distributed setups)
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import settings
from utils.logger import get_logger
from utils.url_utils import url_fingerprint, extract_domain

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PageMeta:
    url: str
    fingerprint: str = field(default_factory=str)
    title: str = ""
    domain: str = ""
    crawl_timestamp: float = field(default_factory=time.time)
    http_status: int = 0
    content_hash: str = ""
    indexed: bool = False
    depth: int = 0
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = url_fingerprint(self.url)
        if not self.domain:
            self.domain = extract_domain(self.url)

    @staticmethod
    def compute_content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Base class ────────────────────────────────────────────────────────────────

class BaseMetadataStore:
    def upsert(self, meta: PageMeta) -> None:
        raise NotImplementedError

    def get(self, url: str) -> Optional[PageMeta]:
        raise NotImplementedError

    def mark_indexed(self, url: str) -> None:
        raise NotImplementedError

    def is_known(self, url: str) -> bool:
        raise NotImplementedError

    def is_indexed(self, url: str) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def recent(self, limit: int = 50) -> list[PageMeta]:
        raise NotImplementedError


# ── SQLite backend ────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pages (
    fingerprint     TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    title           TEXT,
    domain          TEXT,
    crawl_timestamp REAL,
    http_status     INTEGER,
    content_hash    TEXT,
    indexed         INTEGER DEFAULT 0,
    depth           INTEGER DEFAULT 0,
    relevance_score REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_url        ON pages(url);
CREATE INDEX IF NOT EXISTS idx_domain     ON pages(domain);
CREATE INDEX IF NOT EXISTS idx_indexed    ON pages(indexed);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON pages(crawl_timestamp);
"""


class SQLiteMetadataStore(BaseMetadataStore):
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TABLE)
        logger.info("SQLiteMetadataStore ready at %s", self._path)

    def upsert(self, meta: PageMeta) -> None:
        sql = """
        INSERT INTO pages
            (fingerprint, url, title, domain, crawl_timestamp, http_status,
             content_hash, indexed, depth, relevance_score)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            title=excluded.title,
            crawl_timestamp=excluded.crawl_timestamp,
            http_status=excluded.http_status,
            content_hash=excluded.content_hash,
            indexed=excluded.indexed,
            depth=excluded.depth,
            relevance_score=excluded.relevance_score
        """
        with self._conn() as conn:
            conn.execute(
                sql,
                (
                    meta.fingerprint, meta.url, meta.title, meta.domain,
                    meta.crawl_timestamp, meta.http_status, meta.content_hash,
                    int(meta.indexed), meta.depth, meta.relevance_score,
                ),
            )

    def get(self, url: str) -> Optional[PageMeta]:
        fp = url_fingerprint(url)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE fingerprint=?", (fp,)
            ).fetchone()
        if row is None:
            return None
        return PageMeta(
            url=row["url"],
            fingerprint=row["fingerprint"],
            title=row["title"] or "",
            domain=row["domain"] or "",
            crawl_timestamp=row["crawl_timestamp"],
            http_status=row["http_status"],
            content_hash=row["content_hash"] or "",
            indexed=bool(row["indexed"]),
            depth=row["depth"],
            relevance_score=row["relevance_score"],
        )

    def mark_indexed(self, url: str) -> None:
        fp = url_fingerprint(url)
        with self._conn() as conn:
            conn.execute(
                "UPDATE pages SET indexed=1 WHERE fingerprint=?", (fp,)
            )

    def is_known(self, url: str) -> bool:
        fp = url_fingerprint(url)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM pages WHERE fingerprint=?", (fp,)
            ).fetchone()
        return row is not None

    def is_indexed(self, url: str) -> bool:
        fp = url_fingerprint(url)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT indexed FROM pages WHERE fingerprint=?", (fp,)
            ).fetchone()
        return bool(row and row["indexed"])

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    def recent(self, limit: int = 50) -> list[PageMeta]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pages ORDER BY crawl_timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            PageMeta(
                url=r["url"], fingerprint=r["fingerprint"], title=r["title"] or "",
                domain=r["domain"] or "", crawl_timestamp=r["crawl_timestamp"],
                http_status=r["http_status"], content_hash=r["content_hash"] or "",
                indexed=bool(r["indexed"]), depth=r["depth"],
                relevance_score=r["relevance_score"],
            )
            for r in rows
        ]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_metadata_store() -> BaseMetadataStore:
    cfg = settings.metadata_store
    backend = cfg.get("backend", "sqlite")
    if backend == "sqlite":
        return SQLiteMetadataStore(db_path=cfg.get("sqlite_path", "data/metadata/metadata.db"))
    raise ValueError(f"Unsupported metadata backend: {backend}")


metadata_store: BaseMetadataStore = build_metadata_store()
