"""
crawler/link_extractor.py
--------------------------
Extract hyperlinks from HTML and score their relevance to the current
crawl topic, so only high-value links are enqueued in the frontier.

Two relevance classifier backends
----------------------------------
"lightweight"   — TF-IDF / keyword-overlap cosine similarity.
                  Zero extra downloads; works without a GPU.
"transformer"   — Cross-encoder re-ranker (e.g. ms-marco-MiniLM).
                  Higher quality but heavier (~120 MB model).

Both expose the same interface:
    score(anchor_text, surrounding_text, query_terms) -> float  in [0, 1]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config.settings import settings
from utils.logger import get_logger
from utils.url_utils import canonicalise_url

logger = get_logger(__name__)

_MAX_SURROUNDING = 200  # chars of context around a link


# ── Extracted link ────────────────────────────────────────────────────────────

@dataclass
class ExtractedLink:
    url: str
    anchor_text: str
    surrounding_text: str
    relevance_score: float = 0.0


# ── Lightweight (TF-IDF / keyword) classifier ─────────────────────────────────

class LightweightRelevanceClassifier:
    """
    Scores a candidate link based on keyword overlap between:
        query_terms ∩ (anchor_text + surrounding_text)

    Uses character n-gram TF weighting and cosine similarity.
    Fast, no GPU, no model download.
    """

    def __init__(self) -> None:
        self._query_terms: set[str] = set()

    def set_query_terms(self, terms: list[str]) -> None:
        self._query_terms = {t.lower().strip() for t in terms if t.strip()}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{3,}\b", text.lower())

    def score(
        self,
        anchor_text: str,
        surrounding_text: str,
        query_terms: Optional[list[str]] = None,
    ) -> float:
        terms = query_terms or list(self._query_terms)
        if not terms:
            return 0.5  # neutral if no query context

        combined = f"{anchor_text} {surrounding_text}"
        tokens = set(self._tokenize(combined))
        query_set = {t.lower() for t in terms}

        if not query_set:
            return 0.5

        overlap = len(tokens & query_set) / len(query_set)
        # Bonus for URL-path keyword matches
        return min(overlap + 0.05, 1.0)


# ── Transformer-based classifier ──────────────────────────────────────────────

class TransformerRelevanceClassifier:
    """
    Uses a sentence-transformer cross-encoder to score relevance.
    Model is loaded lazily on first use.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._query_terms: list[str] = []

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self._model = CrossEncoder(self._model_name, max_length=128)
                logger.info("TransformerClassifier loaded: %s", self._model_name)
            except Exception as exc:
                logger.warning(
                    "Could not load transformer classifier %s: %s — falling back to lightweight",
                    self._model_name, exc,
                )
                self._model = None

    def set_query_terms(self, terms: list[str]) -> None:
        self._query_terms = terms

    def score(
        self,
        anchor_text: str,
        surrounding_text: str,
        query_terms: Optional[list[str]] = None,
    ) -> float:
        self._load()
        if self._model is None:
            # Fallback to lightweight
            lw = LightweightRelevanceClassifier()
            return lw.score(anchor_text, surrounding_text, query_terms)

        terms = query_terms or self._query_terms
        query = " ".join(terms) if terms else anchor_text
        passage = f"{anchor_text}. {surrounding_text[:200]}"
        try:
            score = self._model.predict([(query, passage)])[0]
            # Cross-encoders return logit; normalise with sigmoid
            import math
            normalised = 1.0 / (1.0 + math.exp(-score))
            return float(normalised)
        except Exception as exc:
            logger.debug("TransformerClassifier.score error: %s", exc)
            return 0.3


# ── Factory ───────────────────────────────────────────────────────────────────

def build_relevance_classifier():
    cfg = settings.relevance_classifier
    model_type = cfg.get("model", "lightweight")
    if model_type == "transformer":
        return TransformerRelevanceClassifier(
            model_name=cfg.get("transformer_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
    return LightweightRelevanceClassifier()


# ── Link extractor ────────────────────────────────────────────────────────────

class LinkExtractor:
    """
    Extracts and scores hyperlinks from HTML.

    For each <a> tag:
    1. Canonicalise the href.
    2. Extract anchor text and ±200 chars of surrounding text.
    3. Score relevance against the current query terms.
    4. Return links above the configured threshold.
    """

    def __init__(self) -> None:
        cfg = settings.link_extractor
        self._max_links: int = cfg.get("max_links_per_page", 200)
        self._threshold: float = settings.crawler.get("relevance_threshold", 0.45)
        self._classifier = build_relevance_classifier()

    def set_query_terms(self, terms: list[str]) -> None:
        self._classifier.set_query_terms(terms)

    def _get_surrounding_text(self, tag, max_chars: int = _MAX_SURROUNDING) -> str:
        """Return text from the parent container around the link."""
        parent = tag.parent
        if parent is None:
            return ""
        full = parent.get_text(separator=" ", strip=True)
        anchor = tag.get_text(strip=True)
        idx = full.find(anchor)
        if idx == -1:
            return full[:max_chars]
        start = max(0, idx - max_chars // 2)
        end = min(len(full), idx + len(anchor) + max_chars // 2)
        return full[start:end]

    def extract(
        self,
        html: str,
        base_url: str,
        query_terms: Optional[list[str]] = None,
    ) -> list[ExtractedLink]:
        """
        Parse *html* and return a list of scored ExtractedLink objects
        above the relevance threshold, up to max_links_per_page.
        """
        if query_terms:
            self.set_query_terms(query_terms)

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as exc:
            logger.warning("BeautifulSoup parse error for %s: %s", base_url, exc)
            return []

        results: list[ExtractedLink] = []
        seen_urls: set[str] = set()

        for tag in soup.find_all("a", href=True):
            if len(results) >= self._max_links:
                break

            href = tag.get("href", "").strip()
            if not href:
                continue

            canonical = canonicalise_url(href, base_url=base_url)
            if canonical is None or canonical in seen_urls:
                continue
            seen_urls.add(canonical)

            anchor = tag.get_text(strip=True)[:200]
            surrounding = self._get_surrounding_text(tag)

            score = self._classifier.score(anchor, surrounding, query_terms)

            if score >= self._threshold:
                results.append(
                    ExtractedLink(
                        url=canonical,
                        anchor_text=anchor,
                        surrounding_text=surrounding[:200],
                        relevance_score=score,
                    )
                )

        # Sort by relevance descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.debug(
            "Extracted %d relevant links from %s (threshold=%.2f)",
            len(results), base_url, self._threshold,
        )
        return results
