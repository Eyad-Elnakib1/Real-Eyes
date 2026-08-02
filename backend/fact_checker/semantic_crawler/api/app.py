"""
api/app.py
-----------
FastAPI application exposing the semantic search engine and crawler via REST.

Endpoints
---------
GET  /health                          system health check
POST /search                          semantic search
POST /crawl                           trigger a crawl job
GET  /crawl/{job_id}                  poll crawl job status
GET  /stats                           system statistics
GET  /recent                          recently crawled pages
DELETE /index                         flush the vector store (admin)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from queue import Queue
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from crawler.crawler_manager import CrawlerManager
from indexer.indexer import Indexer
from indexer.vector_store import vector_store
from search.search_engine import SearchEngine, SearchResponse
from storage.metadata_store import metadata_store
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Semantic Web Crawler & Search Engine",
    description=(
        "A production-grade semantic search system backed by a focused web crawler, "
        "dense vector embeddings, and cross-encoder re-ranking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module-level singletons ───────────────────────────────────────────────────

_search_engine = SearchEngine(auto_crawl=False)

# Active crawl jobs: job_id → status dict
_crawl_jobs: dict[str, dict] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language search query")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of results (default from config)")
    expand: bool = Field(True, description="Enable query expansion")
    auto_crawl: bool = Field(False, description="Trigger crawl if results are sparse")


class SearchOut(BaseModel):
    user_query: str
    evidence: list[list]
    query_time_ms: float
    total_results: int
    expanded_terms: list[str]
    crawl_triggered: bool


class CrawlRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, description="Topic / focus query for crawl")
    seed_urls: Optional[list[str]] = Field(None, description="Optional explicit seed URLs")
    max_pages: Optional[int] = Field(None, ge=1, le=10000)


class CrawlJobStatus(BaseModel):
    job_id: str
    status: str           # "pending" | "running" | "done" | "error"
    query: str
    pages_crawled: int
    chunks_indexed: int
    elapsed_seconds: float
    error: Optional[str]


class StatsOut(BaseModel):
    vector_store_chunks: int
    pages_in_metadata: int
    crawl_jobs_total: int
    crawl_jobs_running: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/search", response_model=SearchOut, tags=["Search"])
async def search(req: SearchRequest):
    """
    Perform a semantic search query.

    Returns top-K results with URLs and cosine similarity scores, matching
    the required output format::

        {
          "user_query": "...",
          "evidence": [["url1", 0.91], ["url2", 0.87], ...]
        }
    """
    try:
        # Temporarily override top_k if specified
        original_top_k = settings.search.get("top_k_output", 10)
        if req.top_k:
            settings.search._data["top_k_output"] = req.top_k

        response: SearchResponse = _search_engine.search(
            query=req.query,
            auto_crawl=req.auto_crawl,
        )

        # Restore
        if req.top_k:
            settings.search._data["top_k_output"] = original_top_k

        return SearchOut(
            user_query=response.user_query,
            evidence=response.evidence,
            query_time_ms=response.query_time_ms,
            total_results=response.total_results,
            expanded_terms=response.expanded_terms,
            crawl_triggered=response.crawl_triggered,
        )
    except Exception as exc:
        logger.error("Search error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/crawl", response_model=CrawlJobStatus, tags=["Crawler"], status_code=202)
async def start_crawl(req: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Launch a background crawl job.
    Returns a job_id that can be polled via GET /crawl/{job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    _crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "query": req.query,
        "pages_crawled": 0,
        "chunks_indexed": 0,
        "elapsed_seconds": 0.0,
        "error": None,
        "started_at": time.time(),
    }

    background_tasks.add_task(
        _run_crawl_job,
        job_id=job_id,
        query=req.query,
        seed_urls=req.seed_urls,
        max_pages=req.max_pages,
    )
    return CrawlJobStatus(**_crawl_jobs[job_id])


@app.get("/crawl/{job_id}", response_model=CrawlJobStatus, tags=["Crawler"])
async def get_crawl_status(job_id: str):
    """Poll the status of a crawl job."""
    job = _crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return CrawlJobStatus(**job)


@app.get("/stats", response_model=StatsOut, tags=["System"])
async def stats():
    """Return system-wide statistics."""
    running = sum(
        1 for j in _crawl_jobs.values() if j["status"] == "running"
    )
    return StatsOut(
        vector_store_chunks=vector_store.count(),
        pages_in_metadata=metadata_store.count(),
        crawl_jobs_total=len(_crawl_jobs),
        crawl_jobs_running=running,
    )


@app.get("/recent", tags=["System"])
async def recent_pages(limit: int = Query(20, ge=1, le=200)):
    """Return recently crawled page metadata."""
    pages = metadata_store.recent(limit=limit)
    return [
        {
            "url": p.url,
            "title": p.title,
            "domain": p.domain,
            "indexed": p.indexed,
            "crawl_timestamp": p.crawl_timestamp,
            "depth": p.depth,
        }
        for p in pages
    ]


@app.delete("/index", tags=["Admin"])
async def flush_index():
    """Flush the vector store (irreversible)."""
    try:
        vector_store.persist()
        return {"message": "Index flushed", "previous_count": vector_store.count()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Background crawl runner ───────────────────────────────────────────────────

async def _run_crawl_job(
    job_id: str,
    query: str,
    seed_urls: Optional[list[str]],
    max_pages: Optional[int],
) -> None:
    """Async background task that runs a full crawl + index pipeline."""
    job = _crawl_jobs[job_id]
    job["status"] = "running"
    t0 = time.time()

    content_queue: Queue = Queue(maxsize=300)
    crawler = CrawlerManager(content_queue=content_queue)
    indexer = Indexer(content_queue=content_queue)

    # Override max_pages from request if provided
    if max_pages:
        crawler._max_pages = max_pages

    try:
        indexer.start()
        pages = await crawler.crawl(query=query, seed_urls=seed_urls)
        indexer.stop(timeout=180.0)

        job["status"] = "done"
        job["pages_crawled"] = pages
        job["chunks_indexed"] = indexer.stats()["chunks_indexed"]
    except Exception as exc:
        logger.error("Crawl job %s failed: %s", job_id, exc, exc_info=True)
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        job["elapsed_seconds"] = round(time.time() - t0, 1)
        logger.info(
            "Crawl job %s finished: status=%s pages=%d",
            job_id, job["status"], job.get("pages_crawled", 0),
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def run_server() -> None:
    import uvicorn
    api_cfg = settings.api
    uvicorn.run(
        "api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        log_level=api_cfg.get("log_level", "info"),
    )


if __name__ == "__main__":
    run_server()
