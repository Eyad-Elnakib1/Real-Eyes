"""
crawler/query_expander.py
--------------------------
Expand a seed query into a richer set of related terms and seed URLs.

Three backends
--------------
"wordnet"    — NLTK WordNet synonyms + hypernyms.  Offline, fast.
"embeddings" — Nearest-neighbour lookup in a small pre-built word
               embedding vocabulary.  Offline, moderate quality.
"llm"        — Claude API for high-quality, context-aware expansion.
               Requires ANTHROPIC_API_KEY.

Seed-URL discovery order (most-reliable first)
-----------------------------------------------
1. ``duckduckgo-search`` library (pip install duckduckgo-search) — no key.
2. Wikipedia OpenSearch API — always available, authoritative for facts.
3. DuckDuckGo HTML scraping — last resort, may be rate-limited.
4. Google Custom Search API — if configured.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import httpx

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── WordNet expander ──────────────────────────────────────────────────────────

def _expand_wordnet(query: str, max_expansions: int) -> list[str]:
    """Return synonyms and hypernyms from NLTK WordNet."""
    try:
        import nltk  # type: ignore
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn

        terms: set[str] = set()
        for word in query.lower().split():
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name != word:
                        terms.add(name)
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        terms.add(lemma.name().replace("_", " "))
                if len(terms) >= max_expansions * 3:
                    break
        return list(terms)[:max_expansions]
    except Exception as exc:
        logger.warning("WordNet expansion failed: %s", exc)
        return []


# ── LLM expander (Claude API) ─────────────────────────────────────────────────

def _expand_llm(query: str, max_expansions: int) -> list[str]:
    """Use Claude to generate semantically related search terms."""
    api_key = (
        settings.query_expander.get("anthropic_api_key")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set; skipping LLM query expansion")
        return []

    try:
        client = httpx.Client(timeout=20)
        prompt = (
            f"Generate {max_expansions} semantically related search query variations "
            f"for the following query: \"{query}\"\n\n"
            "Return ONLY a JSON array of strings, nothing else. "
            "Example: [\"term 1\", \"term 2\", \"term 3\"]"
        )
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.query_expander.get("llm_model", "claude-haiku-4-5-20251001"),
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = resp.json()
        text = data["content"][0]["text"].strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            terms = json.loads(match.group())
            return [str(t) for t in terms[:max_expansions]]
    except Exception as exc:
        logger.warning("LLM query expansion failed: %s", exc)
    return []


# ── Seed URL generation ───────────────────────────────────────────────────────

# A realistic browser UA avoids bot-detection blocks on HTML endpoints.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _search_ddgs_library(query: str, n: int) -> list[str]:
    """
    Use the ``duckduckgo-search`` library (pip install duckduckgo-search).
    Most reliable DuckDuckGo method — uses the internal API, not HTML scraping.
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore
        urls: list[str] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                href = r.get("href", "")
                if href.startswith("http"):
                    urls.append(href)
        if urls:
            logger.debug("DDGS library: %d URLs for '%s'", len(urls), query)
        return urls
    except ImportError:
        return []  # library not installed — fall through to next method
    except Exception as exc:
        logger.debug("DDGS library failed for '%s': %s", query, exc)
        return []


def _search_wikipedia(query: str, n: int) -> list[str]:
    """
    Wikipedia OpenSearch API — always available, no API key needed.
    Returns Wikipedia article URLs for the query.  Essential for fact-checking
    because Wikipedia is authoritative and well-indexed.
    """
    urls: list[str] = []
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": min(n, 10),
                "format": "json",
                "redirects": "resolve",
            },
            headers={"User-Agent": _BROWSER_UA},
            timeout=10,
        )
        data = resp.json()
        # Format: [query_string, [titles], [descriptions], [urls]]
        if len(data) >= 4:
            urls = [u for u in data[3] if u.startswith("http")]
        logger.debug("Wikipedia OpenSearch: %d URLs for '%s'", len(urls), query)
    except Exception as exc:
        logger.warning("Wikipedia OpenSearch failed for '%s': %s", query, exc)
    return urls


def _search_duckduckgo_html(query: str, n: int) -> list[str]:
    """
    DuckDuckGo HTML fallback.  Tries multiple CSS selector patterns because
    DuckDuckGo occasionally changes its markup.
    """
    urls: list[str] = []
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers={"User-Agent": _BROWSER_UA},
            timeout=15,
            follow_redirects=True,
        )
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")

        # Try multiple selector patterns DuckDuckGo has used over time
        selectors = [
            ("a", {"class": "result__url"}),
            ("a", {"class": "result__a"}),
            ("h2", {"class": "result__title"}),
        ]
        for tag, attrs in selectors:
            for elem in soup.find_all(tag, attrs):
                # For <h2>, look for nested <a>
                a = elem if tag == "a" else elem.find("a")
                if not a:
                    continue
                href = a.get("href", "")
                # DuckDuckGo sometimes wraps URLs in redirect links
                if "uddg=" in href:
                    from urllib.parse import unquote, urlparse, parse_qs
                    parsed = urlparse(href)
                    uddg = parse_qs(parsed.query).get("uddg", [""])
                    href = unquote(uddg[0]) if uddg[0] else href
                if href.startswith("http") and href not in urls:
                    urls.append(href)
                    if len(urls) >= n:
                        break
            if urls:
                break

        logger.debug("DuckDuckGo HTML: %d URLs for '%s'", len(urls), query)
    except Exception as exc:
        logger.warning("DuckDuckGo HTML scrape failed for '%s': %s", query, exc)
    return urls


def _search_google_api(query: str, n: int) -> list[str]:
    """Use Google Custom Search API if configured."""
    api_key = settings.seed_urls.get("google_api_key", "")
    cx = settings.seed_urls.get("google_cx", "")
    if not api_key or not cx:
        return []
    try:
        resp = httpx.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx, "q": query, "num": min(n, 10)},
            timeout=10,
        )
        data = resp.json()
        return [item["link"] for item in data.get("items", [])]
    except Exception as exc:
        logger.warning("Google API search failed: %s", exc)
        return []


def _discover_seed_urls(query: str, n: int, backend: str) -> list[str]:
    """
    Try each discovery method in reliability order until we get results.
    Wikipedia is always tried for fact-checking queries (authoritative source).
    """
    urls: list[str] = []

    if backend == "google_api":
        urls = _search_google_api(query, n)
        if urls:
            return urls

    # 1. duckduckgo-search library (best, no scraping)
    urls = _search_ddgs_library(query, n)
    if urls:
        return urls

    # 2. Wikipedia OpenSearch — crucial for fact-checking
    wiki_urls = _search_wikipedia(query, n // 2 + 1)
    urls.extend(wiki_urls)

    # 3. DuckDuckGo HTML scraping
    ddg_urls = _search_duckduckgo_html(query, n)
    seen = set(urls)
    for u in ddg_urls:
        if u not in seen:
            urls.append(u)
            seen.add(u)

    return urls[:n]


# ── Public API ────────────────────────────────────────────────────────────────

class QueryExpander:
    """Expands a seed query into related terms and returns seed URLs."""

    def __init__(self) -> None:
        cfg = settings.query_expander
        self._backend: str = cfg.get("backend", "wordnet")
        self._max_expansions: int = cfg.get("max_expansions", 5)
        self._search_backend: str = settings.seed_urls.get("search_backend", "duckduckgo")
        self._results_per_query: int = settings.seed_urls.get("results_per_query", 10)

    def expand(self, query: str) -> list[str]:
        """Return a list of related query terms (not including the original)."""
        logger.info("Expanding query: '%s' via %s", query, self._backend)
        if self._backend == "wordnet":
            return _expand_wordnet(query, self._max_expansions)
        elif self._backend == "llm":
            return _expand_llm(query, self._max_expansions)
        return []

    def get_seed_urls(self, query: str, expanded_terms: Optional[list[str]] = None) -> list[str]:
        """
        Fetch seed URLs for *query* (and optionally expanded terms).
        Returns deduplicated list of URLs.
        """
        queries_to_search = [query]
        if expanded_terms:
            queries_to_search.extend(expanded_terms[:2])

        seen: set[str] = set()
        all_urls: list[str] = []

        for q in queries_to_search:
            time.sleep(0.5)  # polite pacing
            urls = _discover_seed_urls(q, self._results_per_query, self._search_backend)
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)

        logger.info("Found %d seed URLs for query '%s'", len(all_urls), query)
        return all_urls

    def get_query_terms(self, query: str) -> list[str]:
        """Return the original query split into terms plus expansions."""
        base_terms = [t for t in re.split(r"\s+", query.lower()) if len(t) > 2]
        expanded = self.expand(query)
        extra: list[str] = []
        for term in expanded:
            extra.extend(t for t in term.lower().split() if len(t) > 2)
        return list(dict.fromkeys(base_terms + extra))
