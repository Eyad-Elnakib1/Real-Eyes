"""
storage/cache.py
----------------
Pluggable cache layer with two backends:

* "disk"  — shelf-backed (shelve + fcntl locking), zero dependencies.
* "redis" — Redis backend for distributed / multi-worker deployments.

Both backends expose the same interface:
    get(key)  → value | None
    set(key, value, ttl?)
    delete(key)
    exists(key) → bool
    flush()
"""

from __future__ import annotations

import json
import shelve
import time
from pathlib import Path
from typing import Any, Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseCache:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def flush(self) -> None:
        raise NotImplementedError


# ── Disk (shelve) backend ─────────────────────────────────────────────────────

class DiskCache(BaseCache):
    """
    Simple persistent cache backed by Python's shelve module.
    Values are stored alongside an expiry timestamp.
    Thread-safe via a threading.Lock; not process-safe (use Redis for that).
    """

    def __init__(self, path: str, default_ttl: int = 86400) -> None:
        import threading
        self._path = path
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _shelf(self):
        return shelve.open(self._path, writeback=False)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                with self._shelf() as db:
                    entry = db.get(key)
                    if entry is None:
                        return None
                    value, expires_at = entry
                    if expires_at and time.time() > expires_at:
                        # Expired — lazy delete
                        return None
                    return value
            except Exception as exc:
                logger.debug("DiskCache.get error for key=%s: %s", key, exc)
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            try:
                with self._shelf() as db:
                    db[key] = (value, expires_at)
            except Exception as exc:
                logger.warning("DiskCache.set error for key=%s: %s", key, exc)

    def delete(self, key: str) -> None:
        with self._lock:
            try:
                with self._shelf() as db:
                    db.pop(key, None)
            except Exception as exc:
                logger.debug("DiskCache.delete error for key=%s: %s", key, exc)

    def flush(self) -> None:
        with self._lock:
            try:
                with self._shelf() as db:
                    db.clear()
            except Exception as exc:
                logger.warning("DiskCache.flush error: %s", exc)


# ── Redis backend ─────────────────────────────────────────────────────────────

class RedisCache(BaseCache):
    """
    Redis-backed cache — suitable for multi-process / distributed crawling.
    Values are JSON-serialised before storage.
    """

    def __init__(self, redis_url: str, default_ttl: int = 86400) -> None:
        import redis  # optional dependency
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._default_ttl = default_ttl
        logger.info("RedisCache connected to %s", redis_url)

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("RedisCache.get error: %s", exc)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            self._client.setex(
                key,
                ttl if ttl is not None else self._default_ttl,
                json.dumps(value, default=str),
            )
        except Exception as exc:
            logger.warning("RedisCache.set error: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:
            logger.debug("RedisCache.delete error: %s", exc)

    def flush(self) -> None:
        try:
            self._client.flushdb()
        except Exception as exc:
            logger.warning("RedisCache.flush error: %s", exc)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_cache() -> BaseCache:
    """Return the configured cache backend."""
    cache_cfg = settings.cache
    backend = cache_cfg.get("backend", "disk")
    ttl = cache_cfg.get("ttl_seconds", 86400)

    if backend == "redis":
        redis_url = cache_cfg.get("redis_url", "redis://localhost:6379/1")
        return RedisCache(redis_url=redis_url, default_ttl=ttl)
    else:
        disk_path = cache_cfg.get("disk_path", "data/cache/cache")
        return DiskCache(path=disk_path, default_ttl=ttl)


# Module-level singleton
cache: BaseCache = build_cache()
