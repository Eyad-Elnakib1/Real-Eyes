"""
search/query_processor.py
--------------------------
Processes a raw natural-language user query into a form suitable for
vector-store retrieval.

Steps
-----
1. Text normalisation (lowercase, strip punctuation noise).
2. Dense embedding generation via the shared EmbeddingModel.
3. Optional query expansion (WordNet / LLM) to boost recall.
4. Returns a ProcessedQuery bundle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import settings
from crawler.query_expander import QueryExpander
from indexer.embedder import embedding_model
from utils.logger import get_logger

logger = get_logger(__name__)

_NOISE_RE = re.compile(r"[^\w\s\-']")   # keep word chars, spaces, hyphens, apostrophes


@dataclass
class ProcessedQuery:
    raw: str
    normalised: str
    embedding: np.ndarray
    expanded_terms: list[str] = field(default_factory=list)
    all_terms: list[str] = field(default_factory=list)


class QueryProcessor:
    """
    Transforms a raw user query string into a ProcessedQuery.

    The expanded embedding is a weighted sum of:
        0.8 * embed(original_query) + 0.2 * mean(embed(expansions))
    which slightly broadens the search without losing focus.
    """

    def __init__(self) -> None:
        self._expander = QueryExpander()

    def process(self, query: str, expand: bool = False) -> ProcessedQuery:
        """
        Parameters
        ----------
        query   : Raw user query string.
        expand  : Whether to perform query expansion.  Disabled by default
                  because WordNet synonyms (e.g. "tower" → "turret", "column")
                  shift the query embedding away from the factual topic and
                  hurt retrieval precision for fact-checking queries.
                  Set True for short keyword queries where recall matters more.
        """
        # 1. Normalise
        normalised = self._normalise(query)
        if not normalised:
            raise ValueError(f"Empty query after normalisation: {query!r}")

        # 2. Base embedding
        base_emb = embedding_model.embed_one(normalised)

        # 3. Query expansion
        expanded_terms: list[str] = []
        final_emb = base_emb

        if expand:
            try:
                expanded_terms = self._expander.expand(normalised)
            except Exception as exc:
                logger.warning("Query expansion failed: %s", exc)
                expanded_terms = []

            if expanded_terms:
                # Embed the top expansion terms and blend
                exp_texts = expanded_terms[:3]
                exp_embs = embedding_model.embed_batch(exp_texts)  # (N, dim)
                mean_exp_emb = exp_embs.mean(axis=0)
                # Weighted blend
                blended = 0.8 * base_emb + 0.2 * mean_exp_emb
                norm = np.linalg.norm(blended)
                final_emb = blended / (norm + 1e-9)

        # 4. Aggregate terms for keyword scoring
        base_tokens = [t for t in normalised.lower().split() if len(t) > 2]
        extra_tokens: list[str] = []
        for term in expanded_terms:
            extra_tokens.extend(t for t in term.lower().split() if len(t) > 2)
        all_terms = list(dict.fromkeys(base_tokens + extra_tokens))

        logger.debug(
            "Processed query '%s' → %d expansion terms, emb_dim=%d",
            query, len(expanded_terms), len(final_emb),
        )

        return ProcessedQuery(
            raw=query,
            normalised=normalised,
            embedding=final_emb.astype(np.float32),
            expanded_terms=expanded_terms,
            all_terms=all_terms,
        )

    @staticmethod
    def _normalise(text: str) -> str:
        """Basic text normalisation: strip noise but preserve meaning."""
        text = text.strip()
        text = _NOISE_RE.sub(" ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
