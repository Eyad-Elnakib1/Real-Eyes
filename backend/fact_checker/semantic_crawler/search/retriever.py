"""
search/retriever.py
--------------------
Retrieves, re-ranks, deduplicates, and diversifies results from the vector
store for a given ProcessedQuery.

Pipeline
--------
1. Vector similarity search (cosine) via vector store → top-K raw results.
2. Cross-encoder re-ranking (optional) — improves precision significantly.
3. Deduplication — collapse near-identical chunks from the same URL.
4. Domain diversity filter — suppress over-represented domains.
5. Score threshold — drop results below min_score.

Output: ordered list of RankedResult up to top_k_output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import settings
from indexer.vector_store import SearchResult, vector_store, BaseVectorStore
from search.query_processor import ProcessedQuery
from utils.logger import get_logger
from utils.trust import clickbait_penalty, freshness_score, normalized_trust
from utils.url_utils import extract_domain

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RankedResult:
    url: str
    title: str
    snippet: str          # best-matching text chunk (truncated)
    score: float          # final ranking score in [0, 1]
    domain: str = ""
    chunk_count: int = 1  # how many chunks from this URL contributed
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain:
            self.domain = extract_domain(self.url)


# ── Cross-encoder re-ranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for passage re-ranking.
    Loaded lazily and cached as a module-level singleton.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("CrossEncoder re-ranker loaded: %s", self._model_name)
            return True
        except Exception as exc:
            logger.warning("CrossEncoder not available (%s), skipping re-rank", exc)
            return False

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[tuple[SearchResult, float]]:
        """
        Returns (result, rerank_score) pairs sorted by score descending.
        Falls back to original similarity scores if model unavailable.
        """
        if not self._load():
            return [(r, r.score) for r in results]

        try:
            pairs = [(query, r.text[:512]) for r in results]
            logits = self._model.predict(pairs)
            # Sigmoid normalisation
            import math
            scores = [1.0 / (1.0 + math.exp(-float(l))) for l in logits]
            combined = sorted(
                zip(results, scores), key=lambda x: x[1], reverse=True
            )
            return combined
        except Exception as exc:
            logger.warning("Re-ranking failed: %s", exc)
            return [(r, r.score) for r in results]


_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        model = settings.search.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        _reranker = CrossEncoderReranker(model)
    return _reranker


# ── Main retriever ────────────────────────────────────────────────────────────

class Retriever:
    """
    Full retrieval pipeline: search → rerank → deduplicate → diversify.
    """

    def __init__(self, store: Optional[BaseVectorStore] = None) -> None:
        self._store = store or vector_store
        cfg = settings.search
        self._top_k_retrieval: int = cfg.get("top_k_retrieval", 20)
        self._top_k_output: int = cfg.get("top_k_output", 10)
        self._enable_reranking: bool = cfg.get("enable_reranking", True)
        self._diversity_threshold: float = cfg.get("diversity_threshold", 0.85)
        self._min_score: float = cfg.get("min_score", 0.2)
        # Final-ranking blend weights (relevance + trust + freshness)
        self._w_relevance: float = cfg.get("rank_w_relevance", 0.50)
        self._w_trust: float = cfg.get("rank_w_trust", 0.30)
        self._w_freshness: float = cfg.get("rank_w_freshness", 0.20)

    def retrieve(self, pq: ProcessedQuery) -> list[RankedResult]:
        """
        Full retrieval pipeline for a ProcessedQuery.
        Returns ordered list of unique, diverse RankedResult objects.
        """
        # Step 1: vector similarity search
        raw_results = self._store.search(
            query_embedding=pq.embedding,
            top_k=self._top_k_retrieval,
        )
        if not raw_results:
            logger.info("Vector store returned no results for: '%s'", pq.raw)
            return []

        logger.debug("Retrieved %d raw results for '%s'", len(raw_results), pq.raw)

        # Step 2: Re-ranking
        if self._enable_reranking and len(raw_results) > 1:
            reranked = get_reranker().rerank(pq.normalised, raw_results)
        else:
            reranked = [(r, r.score) for r in raw_results]

        # Step 3: Aggregate chunks by URL (keep best score per URL)
        url_map: dict[str, tuple[SearchResult, float]] = {}
        for result, score in reranked:
            if result.url not in url_map or score > url_map[result.url][1]:
                url_map[result.url] = (result, score)

        # Step 4: Deduplicate (near-identical snippets from same domain)
        deduped = self._deduplicate(list(url_map.values()))

        # Step 5: Domain diversity
        diverse = self._diversify(deduped)

        # Step 6: Blend relevance + trust + freshness, demote clickbait
        blended = self._blend_scores(diverse)

        # Step 7: Score threshold
        filtered = [(r, s) for r, s in blended if s >= self._min_score]

        # Step 8: Build output objects
        output: list[RankedResult] = []
        for result, score in filtered[: self._top_k_output]:
            output.append(
                RankedResult(
                    url=result.url,
                    title=result.metadata.get("title", ""),
                    snippet=_truncate_snippet(result.text),
                    score=round(score, 4),
                    metadata=result.metadata,
                )
            )

        logger.info(
            "Retrieved %d results for '%s' (from %d raw)",
            len(output), pq.raw, len(raw_results),
        )
        return output

    def _deduplicate(
        self,
        items: list[tuple[SearchResult, float]],
    ) -> list[tuple[SearchResult, float]]:
        """
        Remove near-duplicate results (same domain + very similar text).
        Uses simple shingling overlap as a proxy for near-duplicate detection.
        """
        kept: list[tuple[SearchResult, float]] = []
        seen_shingles: dict[str, set[str]] = {}  # domain → shingle sets

        for result, score in sorted(items, key=lambda x: x[1], reverse=True):
            domain = result.domain
            shingles = _shingles(result.text)

            if domain in seen_shingles:
                overlap = len(shingles & seen_shingles[domain]) / max(len(shingles), 1)
                if overlap > self._diversity_threshold:
                    continue
                seen_shingles[domain] |= shingles
            else:
                seen_shingles[domain] = shingles

            kept.append((result, score))

        return kept

    def _diversify(
        self,
        items: list[tuple[SearchResult, float]],
    ) -> list[tuple[SearchResult, float]]:
        """
        Enforce domain diversity: max 3 results per domain.
        """
        domain_count: dict[str, int] = {}
        output: list[tuple[SearchResult, float]] = []
        MAX_PER_DOMAIN = 3

        for result, score in items:
            domain = result.domain
            if domain_count.get(domain, 0) >= MAX_PER_DOMAIN:
                continue
            domain_count[domain] = domain_count.get(domain, 0) + 1
            output.append((result, score))

        return output

    def _blend_scores(
        self,
        items: list[tuple[SearchResult, float]],
    ) -> list[tuple[SearchResult, float]]:
        """Combine relevance + source trust + freshness into a final score,
        then demote clickbait. Re-sorts descending by the blended score.

            final = (w_rel*relevance + w_trust*trust + w_fresh*freshness)
                    * clickbait_penalty(title)
        """
        blended: list[tuple[SearchResult, float]] = []
        for result, relevance in items:
            meta = result.metadata or {}
            trust = normalized_trust(result.url)
            fresh = freshness_score(meta.get("published_date")
                                    or meta.get("crawl_timestamp"))
            score = (
                self._w_relevance * relevance
                + self._w_trust * trust
                + self._w_freshness * fresh
            )
            score *= clickbait_penalty(meta.get("title", ""))
            blended.append((result, score))

        blended.sort(key=lambda x: x[1], reverse=True)
        return blended


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shingles(text: str, k: int = 4) -> set[str]:
    """Return set of k-character shingles for near-duplicate detection."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _truncate_snippet(text: str, max_chars: int = 800) -> str:
    """Return a clean, truncated snippet.

    800 chars (~150 words) gives the NLI model substantially more context
    while staying well within the 512-token limit when combined with a
    typical claim hypothesis (~30 tokens).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
