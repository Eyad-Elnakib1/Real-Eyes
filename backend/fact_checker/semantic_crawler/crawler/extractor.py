"""
crawler/extractor.py
---------------------
Content extraction from raw HTML.

Strategy
--------
1. Try ``trafilatura`` (best recall/precision for main content).
2. Fall back to ``readability-lxml`` (Mozilla Readability port).
3. Final fallback: BeautifulSoup heuristic extraction.

All three approaches strip scripts, styles, nav, ads, and boilerplate.

Output
------
ExtractedContent with:
    title       – page <title> or <h1>
    text        – clean plain text (newline-separated paragraphs)
    html        – optional cleaned HTML (for downstream processing)
    lang        – detected language (ISO 639-1)
    word_count  – approximate word count
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Tags whose content we always discard
_JUNK_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "aside", "form", "button", "input", "select",
    "header",
}

_WHITESPACE_RE = re.compile(r"\s{3,}")
_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass
class ExtractedContent:
    url: str
    title: str = ""
    text: str = ""          # clean plain text
    html: Optional[str] = None
    lang: str = "en"
    word_count: int = 0
    extraction_method: str = "unknown"
    published_date: str = ""  # ISO date string ("" if unknown) — used for freshness ranking

    def __post_init__(self) -> None:
        if self.text and not self.word_count:
            self.word_count = len(self.text.split())

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 200


# ── Extractor backends ────────────────────────────────────────────────────────

def _extract_trafilatura(html: str, url: str) -> Optional[ExtractedContent]:
    """Primary extraction using trafilatura."""
    try:
        import trafilatura  # type: ignore
        from trafilatura.settings import use_config  # type: ignore

        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=False,
            favor_recall=True,
            config=config,
        )
        if not result or len(result.strip()) < 100:
            return None

        # Get title + publish date
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = (meta.title or "") if meta else ""
        published = (meta.date or "") if meta else ""

        return ExtractedContent(
            url=url,
            title=title,
            text=_clean_text(result),
            published_date=published or "",
            extraction_method="trafilatura",
        )
    except Exception as exc:
        logger.debug("trafilatura extraction failed for %s: %s", url, exc)
        return None


def _extract_readability(html: str, url: str) -> Optional[ExtractedContent]:
    """Fallback extraction using readability-lxml."""
    try:
        from readability import Document  # type: ignore
        from bs4 import BeautifulSoup

        doc = Document(html, url=url)
        title = doc.title() or ""
        content_html = doc.summary(html_partial=False)

        soup = BeautifulSoup(content_html, "lxml")
        text = soup.get_text(separator="\n", strip=True)
        text = _clean_text(text)

        if len(text) < 100:
            return None

        return ExtractedContent(
            url=url,
            title=title,
            text=text,
            html=content_html,
            extraction_method="readability",
        )
    except Exception as exc:
        logger.debug("readability extraction failed for %s: %s", url, exc)
        return None


def _extract_beautifulsoup(html: str, url: str) -> Optional[ExtractedContent]:
    """Last-resort extraction using BeautifulSoup heuristics."""
    try:
        from bs4 import BeautifulSoup, Tag

        soup = BeautifulSoup(html, "lxml")

        # Remove junk tags in-place
        for tag in soup.find_all(_JUNK_TAGS):
            tag.decompose()

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Try <article> or <main> first
        main = soup.find("article") or soup.find("main") or soup.find("body")
        if not main:
            return None

        # Collect <p> and heading text
        paragraphs: list[str] = []
        for elem in main.find_all(["p", "h1", "h2", "h3", "li"]):
            t = elem.get_text(separator=" ", strip=True)
            if len(t) > 40:  # skip very short fragments
                paragraphs.append(t)

        text = _clean_text("\n".join(paragraphs))
        if len(text) < 200:
            # Last resort: all text from body
            body = soup.find("body")
            text = _clean_text(body.get_text(separator="\n", strip=True) if body else "")

        if len(text) < 100:
            return None

        return ExtractedContent(
            url=url,
            title=title,
            text=text,
            extraction_method="beautifulsoup",
        )
    except Exception as exc:
        logger.debug("BeautifulSoup extraction failed for %s: %s", url, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise whitespace in extracted text."""
    text = _WHITESPACE_RE.sub("  ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_content(html: str, url: str) -> Optional[ExtractedContent]:
    """
    Extract main textual content from *html*.

    Tries backends in order of quality, falling back automatically.
    Returns None if no backend produces usable content.
    """
    for extractor in (_extract_trafilatura, _extract_readability, _extract_beautifulsoup):
        result = extractor(html, url)
        if result and not result.is_empty:
            logger.debug(
                "Extracted %d words from %s via %s",
                result.word_count, url, result.extraction_method,
            )
            return result

    logger.warning("All extraction methods failed for %s", url)
    return None
