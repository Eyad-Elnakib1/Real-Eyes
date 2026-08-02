"""
utils/url_utils.py
------------------
URL normalisation, canonicalisation, and fingerprinting helpers.

Design decisions
----------------
* We canonicalise to lowercase scheme + host, strip default ports,
  sort query params, and remove well-known tracker params.
* We produce a stable fingerprint (SHA-256 prefix) so duplicates can be
  detected cheaply in the frontier.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import (
    ParseResult,
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

# Query-string keys that are purely tracking artefacts and carry no
# semantic meaning for the resource.
_STRIP_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "msclkid", "ref", "referrer", "source",
        "_ga", "_gl", "mc_eid", "mc_cid",
    }
)

# File extensions that point to non-HTML resources we cannot crawl.
_NON_HTML_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
        ".mp4", ".mp3", ".wav", ".ogg", ".avi", ".mov",
        ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
        ".exe", ".dmg", ".pkg", ".deb", ".rpm",
        ".css", ".js", ".json", ".xml", ".rss", ".atom",
    }
)

_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def canonicalise_url(url: str, base_url: Optional[str] = None) -> Optional[str]:
    """
    Return a canonical, normalised form of *url* or ``None`` if the URL is
    unsuitable for crawling.

    Steps:
    1. Resolve relative URLs against *base_url*.
    2. Lower-case scheme and host.
    3. Remove default port.
    4. Remove fragment identifier.
    5. Strip tracking query parameters.
    6. Sort remaining query parameters for stable comparison.
    7. Decode %XX escapes for unreserved characters.
    """
    url = url.strip()
    if not url or url.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
        return None

    # Resolve relative URLs
    if base_url and not url.startswith(("http://", "https://")):
        from urllib.parse import urljoin
        url = urljoin(base_url, url)

    try:
        parsed: ParseResult = urlparse(url)
    except Exception:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    # Strip default ports
    port = parsed.port
    netloc = host if (port is None or port == _DEFAULT_PORTS.get(scheme)) else f"{host}:{port}"

    # Clean path — collapse duplicate slashes, keep trailing slash consistent
    path = re.sub(r"/+", "/", parsed.path) or "/"

    # Reject non-HTML file extensions
    path_lower = path.lower()
    if any(path_lower.endswith(ext) for ext in _NON_HTML_EXTENSIONS):
        return None

    # Strip tracking params and sort
    qs_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _STRIP_PARAMS
    ]
    qs_pairs.sort()
    query = urlencode(qs_pairs)

    canonical = urlunparse((scheme, netloc, path, "", query, ""))
    return canonical


def url_fingerprint(url: str) -> str:
    """Return a 16-char hex fingerprint (SHA-256 prefix) of a canonical URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def extract_domain(url: str) -> str:
    """Return the registered domain (host without www. prefix)."""
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def is_same_domain(url_a: str, url_b: str) -> bool:
    """Return True if both URLs share the same registered domain."""
    return extract_domain(url_a) == extract_domain(url_b)


def is_valid_http_url(url: str) -> bool:
    """Cheap check — does not canonicalise."""
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False
