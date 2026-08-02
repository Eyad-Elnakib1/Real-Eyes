"""
search/evaluation.py
---------------------
Evaluation utilities for the semantic search engine.

Metrics implemented
-------------------
* precision@k      — fraction of top-k results that are relevant
* recall@k         — fraction of relevant docs found in top-k
* average_precision — area under the P/R curve for a single query
* mean_average_precision (MAP)
* nDCG@k           — normalised discounted cumulative gain
* mean_reciprocal_rank (MRR)

Usage
-----
    from search.evaluation import evaluate_search

    results = evaluate_search(
        engine=search_engine,
        queries=["machine learning", "neural networks"],
        relevant_urls={
            "machine learning": {"https://en.wikipedia.org/wiki/ML", ...},
            "neural networks":  {"https://en.wikipedia.org/wiki/ANN", ...},
        },
        k=10,
    )
    print(results)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from search.search_engine import SearchEngine
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Per-query metrics ─────────────────────────────────────────────────────────

@dataclass
class QueryMetrics:
    query: str
    precision_at_k: float
    recall_at_k: float
    average_precision: float
    ndcg_at_k: float
    reciprocal_rank: float
    k: int
    num_retrieved: int
    num_relevant: int


# ── Aggregate metrics ─────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    map_score: float          # Mean Average Precision
    mrr_score: float          # Mean Reciprocal Rank
    mean_ndcg: float          # Mean nDCG@k
    mean_precision: float
    mean_recall: float
    k: int
    num_queries: int
    per_query: list[QueryMetrics] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"  Evaluation Results (k={self.k}, n={self.num_queries} queries)",
            f"{'='*50}",
            f"  MAP                 : {self.map_score:.4f}",
            f"  MRR                 : {self.mrr_score:.4f}",
            f"  Mean nDCG@{self.k}        : {self.mean_ndcg:.4f}",
            f"  Mean Precision@{self.k}   : {self.mean_precision:.4f}",
            f"  Mean Recall@{self.k}      : {self.mean_recall:.4f}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ── Metric functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved[:k]
    hits = sum(1 for url in top_k if url in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for url in top_k if url in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Area under the precision-recall curve."""
    if not relevant:
        return 0.0
    hits = 0
    cumulative_precision = 0.0
    for i, url in enumerate(retrieved, 1):
        if url in relevant:
            hits += 1
            cumulative_precision += hits / i
    return cumulative_precision / len(relevant) if relevant else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance (relevant=1, not relevant=0).
    """
    def dcg(ranked: list[str]) -> float:
        score = 0.0
        for i, url in enumerate(ranked[:k], 1):
            if url in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    actual = dcg(retrieved)
    # Ideal DCG: put all relevant docs at the top
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return actual / ideal if ideal > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, url in enumerate(retrieved, 1):
        if url in relevant:
            return 1.0 / i
    return 0.0


# ── High-level evaluator ──────────────────────────────────────────────────────

def evaluate_search(
    engine: SearchEngine,
    queries: list[str],
    relevant_urls: dict[str, set[str]],
    k: int = 10,
) -> EvalResults:
    """
    Run *queries* through *engine* and compute retrieval metrics against
    the provided relevance judgements.

    Parameters
    ----------
    engine        : configured SearchEngine instance
    queries       : list of query strings
    relevant_urls : mapping from query string → set of relevant URLs
    k             : evaluation depth

    Returns
    -------
    EvalResults with per-query and aggregate metrics
    """
    per_query: list[QueryMetrics] = []

    for query in queries:
        relevant = relevant_urls.get(query, set())
        response = engine.search(query=query)
        retrieved = [r.url for r in response.results]

        qm = QueryMetrics(
            query=query,
            precision_at_k=precision_at_k(retrieved, relevant, k),
            recall_at_k=recall_at_k(retrieved, relevant, k),
            average_precision=average_precision(retrieved, relevant),
            ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
            k=k,
            num_retrieved=len(retrieved),
            num_relevant=len(relevant),
        )
        per_query.append(qm)
        logger.info(
            "Eval '%s': P@%d=%.3f R@%d=%.3f AP=%.3f nDCG=%.3f RR=%.3f",
            query, k, qm.precision_at_k, k, qm.recall_at_k,
            qm.average_precision, qm.ndcg_at_k, qm.reciprocal_rank,
        )

    n = len(per_query) or 1
    return EvalResults(
        map_score=sum(m.average_precision for m in per_query) / n,
        mrr_score=sum(m.reciprocal_rank for m in per_query) / n,
        mean_ndcg=sum(m.ndcg_at_k for m in per_query) / n,
        mean_precision=sum(m.precision_at_k for m in per_query) / n,
        mean_recall=sum(m.recall_at_k for m in per_query) / n,
        k=k,
        num_queries=len(per_query),
        per_query=per_query,
    )
