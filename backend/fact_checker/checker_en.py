"""
test_live_web3.py
------------------
Live web fact-checker using the 3-class DeBERTa model
(nli_model: LABEL_0=SUPPORTS, LABEL_1=REFUTES, LABEL_2=NEI).

Differences from test_live_web2.py (binary model):
  - NLI model returns P(SUPPORTS) + P(REFUTES) + P(NEI)  — they sum to 1
  - Aggregation denominator includes NEI mass → genuine uncertainty lowers
    both sup_share and ref_share, making NEI verdict more likely
  - Strong-refutation override still applies
  - Source credibility weighting still applies
  - "fact check" secondary search query still applies

Usage:
    python test_live_web3.py --claim "Vaccines cause autism."
    python test_live_web3.py --claim "..." --verbose
    python test_live_web3.py --batch
    python test_live_web3.py --batch --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Shared trust scoring (single source of truth across crawler + retriever + checker)
sys.path.insert(0, str(Path(__file__).resolve().parent / "semantic_crawler"))
from utils.trust import trust_score as _credibility  # noqa: E402  (1.0 neutral, fact-checkers 3.0, social 0.2)
from utils.trust import RSP_BLOCKED as _RSP_BLOCKED   # noqa: E402  WP:RSP unreliable/deprecated/blacklisted
from utils.trust import RSP_TRUSTED as _RSP_TRUSTED   # noqa: E402  WP:RSP generally-reliable (searched first)

# ── Console colours ───────────────────────────────────────────────────────────
_C = {
    "SUPPORTS":        "\033[92m",
    "REFUTES":         "\033[91m",
    "NOT ENOUGH INFO": "\033[93m",
    "RESET":           "\033[0m",
    "DIM":             "\033[2m",
    "BOLD":            "\033[1m",
}

def _col(label: str, text: str) -> str:
    return f"{_C.get(label, '')}{text}{_C['RESET']}"


# ── 3-Class NLI model ─────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent / "nli_model"

# Discovered by probing: LABEL_0=SUPPORTS, LABEL_1=REFUTES, LABEL_2=NEI
_LABEL_MAP = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

_nli_tok   = None
_nli_model = None

def _get_nli():
    global _nli_tok, _nli_model
    if _nli_model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("  Loading 3-class NLI model…", file=sys.stderr)
        _nli_tok   = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _nli_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _nli_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _nli_model.to(device)
    return _nli_tok, _nli_model


def _predict_nli(claim: str, texts: list[str], batch_size: int = 8) -> list[dict]:
    """
    Run 3-class NLI on (evidence, claim) pairs.
    Returns list of {label, score, probabilities} dicts.
    """
    import torch
    import torch.nn.functional as F

    tok, mdl = _get_nli()
    device = next(mdl.parameters()).device
    results: list[dict] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(
            batch,                    # premise = evidence
            [claim] * len(batch),     # hypothesis = claim
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            probs = F.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()

        for row in probs:
            prob_dict = {_LABEL_MAP[i]: round(float(row[i]), 4) for i in range(3)}
            best_idx  = int(row.argmax())
            results.append({
                "label":         _LABEL_MAP[best_idx],
                "score":         round(float(row[best_idx]), 4),
                "probabilities": prob_dict,
            })

    return results


# ── Embedding model ───────────────────────────────────────────────────────────
_embed_model = None

def _get_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def _embed(texts: list[str]) -> np.ndarray:
    return _get_embedder().encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    ).astype(np.float32)


# ── Cross-encoder re-ranker ───────────────────────────────────────────────────
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        except Exception:
            _reranker = False
    return _reranker if _reranker is not False else None


# ── Sentence chunker ──────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def _chunk_text(text: str, max_tokens: int = 120, overlap: int = 1) -> list[str]:
    sentences = [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 20]
    if not sentences:
        stride = max_tokens * 4
        return [text[i:i+stride].strip() for i in range(0, len(text), stride)
                if len(text[i:i+stride].strip()) > 80]
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        t = len(sent.split())
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            current_tokens = sum(len(s.split()) for s in current)
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 60]


# Source credibility tiers now live in semantic_crawler/utils/trust.py and are
# imported above as `_credibility` (trust_score). One source of truth shared by
# the crawler frontier, the search retriever, and this fact-checker.


# ── Web search ────────────────────────────────────────────────────────────────
_UA      = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_WIKI_UA = "FactCheckerBot/1.0 (grad-project; malakeid235@gmail.com) httpx/0.27"

# Domains known to publish misinformation — URLs from these are silently dropped.
_DOMAIN_BLOCKLIST: frozenset[str] = frozenset({
    # anti-vax / medical misinformation
    "stopmandatoryvaccination.com", "naturalblaze.com", "greenmedinfo.com",
    "childrenshealthdefense.org", "nvic.org", "mercola.com", "informedparent.co.uk",
    "westonaprice.org", "thevaccinereaction.org", "vaccineimpact.com",
    "kirschsubstack.com", "stevekirsch.substack.com",
    "vactruth.com", "vaxxter.com", "ageofautism.com", "healthimpactnews.com",
    "vaccines.news", "medicalkidnap.com", "thetruthaboutcancer.com",
    "healthnutnews.com", "realfarmacy.com", "anh-usa.org",
    # flat-earth / pseudoscience
    "flat-earther.com", "flat-earth.org", "theflatearthsociety.org",
    "aplanetruth.info", "flatearthforum.com",
    "fevids.com", "ezekieldiet.com",
    # conspiracy / fringe
    "infowars.com", "naturalnews.com", "beforeitsnews.com", "zerohedge.com",
    "globalresearch.ca", "activistpost.com", "shtfplan.com",
    "prophecynewswatch.com", "wnd.com", "newstarget.com",
    "operationdisclosureofficial.com", "whale.to", "conspiracytheory.net",
    "thefocalpoints.com", "kyletothemoon.com",
    "davidicke.com", "rense.com", "humansarefree.com", "worldtruth.tv",
    "collective-evolution.com", "wakingtimes.com", "21stcenturywire.com",
    "veteranstoday.com", "thelibertybeacon.com",
    # fake-news content farms (formerly YourNewsWire, etc.)
    "newspunch.com", "yournewswire.com", "newswars.com", "summit.news",
    "banned.video", "gnews.org", "gtv.org", "thebl.com",
    # state-controlled propaganda outlets
    "rt.com", "sputniknews.com", "sputnikglobe.com", "presstv.ir",
    # alt-video / UGC platforms that host re-uploaded misinformation
    "bitchute.com", "brighteon.com", "rumble.com", "odysee.com",
    # Deprecated / generally-unreliable per Wikipedia WP:RSP (Perennial Sources)
    # https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources
    # — community-maintained list of sources with no reputation for fact-checking.
    "dailymail.co.uk", "thesun.co.uk", "dailystar.co.uk",          # UK tabloids
    "breitbart.com", "thegatewaypundit.com", "newsmax.com",        # deprecated US
    "oann.com", "theblaze.com", "dailycaller.com", "frontpagemag.com",
    "thefederalist.com", "pjmedia.com", "occupydemocrats.com",
    "projectveritas.com", "vdare.com", "unz.com", "lifesitenews.com",
    "theepochtimes.com", "ntd.com", "mintpressnews.com", "thegrayzone.com",
    "tass.com", "cgtn.com", "globaltimes.cn", "journal-neo.org",    # state media
    "almayadeen.net", "anna-news.info",
    # low-quality / content-farm
    "allnewspipeline.com", "thenewamerican.com", "whydontyoutrythis.com",
    # parenting/community forums — anecdotal content confuses NLI on medical claims
    "babycenter.com", "babycenter.ca", "community.babycenter.com",
    "thebump.com", "whattoexpect.com",
})

# Trusted sources for the first-pass restricted search = the WP:RSP
# generally-reliable list (_RSP_TRUSTED). DuckDuckGo's site: operator can't take
# all ~166 domains in one query, so we split them into batches of _SITE_BATCH
# domains; each batch becomes one "claim terms site:a OR site:b OR …" query.
_SITE_BATCH   = 25
_SEARCH_DELAY = 1.5   # seconds to wait between DuckDuckGo queries (avoid rate-limiting)

# Domains used for the trusted-first site: search. Defaults to WP:RSP; the
# Arabic variant (test_live_web4_ar.py) extends this with Arabic trusted sources.
_TRUSTED_SEARCH_DOMAINS: frozenset[str] = _RSP_TRUSTED

def _trusted_site_queries(query: str) -> list[str]:
    """Yield batched site:-restricted queries covering every trusted domain."""
    doms = sorted(_TRUSTED_SEARCH_DOMAINS)
    out: list[str] = []
    for i in range(0, len(doms), _SITE_BATCH):
        clause = " OR ".join(f"site:{d}" for d in doms[i:i + _SITE_BATCH])
        out.append(f"{query} {clause}")
    return out


def _is_blocked(url: str) -> bool:
    """Return True if the URL belongs to a blocklisted domain.

    Two layers: the hand-curated _DOMAIN_BLOCKLIST above, plus every source
    WP:RSP marks deprecated / blacklisted / generally-unreliable (_RSP_BLOCKED).
    """
    try:
        host = url.split("/")[2].lower().replace("www.", "")
    except IndexError:
        return False
    parts = host.split(".")
    # Test host and each parent domain (edition.cnn.com → cnn.com) against both sets.
    for i in range(len(parts) - 1):
        d = ".".join(parts[i:])
        if d in _DOMAIN_BLOCKLIST or d in _RSP_BLOCKED:
            return True
    return False


def _ddgs_search(query: str, n: int) -> list[str]:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []
    try:
        with DDGS() as ddgs:
            results = [r.get("href", "") for r in ddgs.text(query, max_results=n)
                       if r.get("href", "").startswith("http")]
        filtered = [u for u in results if not _is_blocked(u)]
        if len(results) != len(filtered):
            dropped = len(results) - len(filtered)
            print(f"  [DDGS] blocked {dropped} domain(s) from results", file=sys.stderr)
        return filtered
    except Exception as exc:
        print(f"  [DDGS] {exc}", file=sys.stderr)
        return []


def _short_query(query: str, max_words: int = 8) -> str:
    return " ".join(query.split()[:max_words])


# ── Social-media post preprocessing ──────────────────────────────────────────
#
# Two separate outputs from a raw pasted post:
#   _clean_claim(raw)       → clean readable text for the NLI model / embedding
#   _build_search_query(raw) → focused keyword query for web search
#
# Search query uses Strategy E (hashtag-first + NER + YAKE):
#   Layer 1 — hashtags from the raw post (#5G, #autism) are the author's own
#              topic tags; always include them first regardless of position.
#   Layer 2 — named entities from spaCy NER (Bill Gates, MMR, NASA, COVID-19)
#              always appear even when buried mid-paragraph.
#   Layer 3 — YAKE statistical keywords fill remaining slots; no stopword list
#              needed because YAKE down-weights high-frequency function words.
#
# This replaces the old hardcoded _STOPWORDS + _NOISE_PHRASES approach which
# required manual updates for every new meme, slang, or CTA variant.

_SPACY_NLP = None   # lazy singleton


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy  # type: ignore
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def _clean_claim(raw: str) -> str:
    """Strip structural social-media noise → clean text for NLI / embedding."""
    t = re.sub(r"https?://\S+", " ", raw)        # URLs
    t = re.sub(r"@\w+", " ", t)                  # @mentions
    t = t.replace("#", " ")                       # keep hashtag words, drop '#'
    t = re.sub(r"[^\w\s.,!?'\"-]", " ", t)       # emojis / symbols
    t = re.sub(r"[!?]{2,}", ". ", t)             # !!! / ??? → sentence break
    t = re.sub(r"\.{2,}", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _build_search_query(raw: str, max_terms: int = 15) -> str:
    """Build a web-search query using hashtag-first + NER + YAKE (Strategy E).

    Layer 1 — hashtags: the post author's own topic tags (#5G, #autism).
    Layer 2 — spaCy NER: named entities regardless of position in the post.
    Layer 3 — YAKE: statistical keywords to fill remaining slots.
    Falls back to document-order keyword extraction if libraries are missing.
    """
    # Layer 1: hashtags from raw post (before any cleaning). \w is Unicode-aware,
    # so this captures non-Latin tags too (e.g. Arabic #كأس_العالم, #قطر2022) —
    # an ASCII-only class would silently drop them. Underscores join multi-word
    # tags, so split them into separate search terms (#كأس_العالم → كأس العالم).
    hashtags = re.findall(r"#(\w{2,})", raw, re.UNICODE)
    seen: set[str] = set()
    priority: list[str] = []
    for t in hashtags:
        tag = t.replace("_", " ").strip().lower()
        if tag and tag not in seen and not tag.isdigit():
            seen.add(tag); priority.append(tag)

    cleaned = _clean_claim(raw)

    # Layer 2: NER (named entities, always included regardless of position)
    try:
        nlp = _get_spacy_nlp()
        doc = nlp(cleaned)
        for ent in doc.ents:
            key = ent.text.lower().strip()
            if key not in seen and len(key) > 1 and not key.replace(" ", "").isdigit():
                seen.add(key); priority.append(key)
    except Exception:
        pass  # spaCy unavailable — layers 1 + 3 still run

    # Layer 3: YAKE statistical keywords for remaining slots
    remaining = max_terms - len(priority)
    if remaining > 0:
        try:
            import yake  # type: ignore
            extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.7,
                                              top=remaining + 5)
            for kw, _ in extractor.extract_keywords(cleaned):
                kw_low = kw.lower()
                if kw_low not in seen and remaining > 0:
                    seen.add(kw_low); priority.append(kw_low); remaining -= 1
        except Exception:
            # YAKE unavailable — fill with document-order tokens
            for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", cleaned):
                low = tok.lower()
                has_digit = any(c.isdigit() for c in tok)
                if low.isdigit() or low in seen or (len(low) < 3 and not has_digit):
                    continue
                seen.add(low); priority.append(low); remaining -= 1
                if remaining <= 0:
                    break

    return " ".join(priority[:max_terms]) if priority else _short_query(cleaned)


def _search_web(query: str, n: int = 10) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []

    def add(u: str) -> None:
        if u.startswith("http") and u not in seen:
            seen.add(u); urls.append(u)

    # Step 1 — TRUSTED-ONLY search. Restrict DuckDuckGo to the WP:RSP
    # generally-reliable domains via the site: operator, batched because the
    # full list is too long for one query. We stop early once we have enough
    # trusted URLs to fill the result budget, so we don't fire all ~7 batches
    # on every claim. These URLs are added first → they occupy the front slots.
    for i, tq in enumerate(_trusted_site_queries(query)):
        if i > 0:
            time.sleep(_SEARCH_DELAY)   # space out queries so DDG doesn't throttle us
        for u in _ddgs_search(tq, n):
            add(u)
        if len(urls) >= n + 5:
            break
    print(f"  [search] {len(urls)} trusted URLs from RSP_TRUSTED", file=sys.stderr)

    # Step 2 — open-web search for everything else (fills remaining slots)
    time.sleep(_SEARCH_DELAY)
    ddgs_urls = _ddgs_search(query, n)
    if not ddgs_urls:
        import time as _t; _t.sleep(2)
        sq = _short_query(query)
        print(f"  [DDGS] retrying: \"{sq}\"", file=sys.stderr)
        ddgs_urls = _ddgs_search(sq, n)
    for u in ddgs_urls:
        add(u)

    # Query 3 — Wikipedia
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": _short_query(query, 6),
                    "limit": 5, "format": "json", "redirects": "resolve"},
            headers={"User-Agent": _WIKI_UA}, timeout=10,
        )
        resp.raise_for_status()
        for u in resp.json()[3]:
            add(u)
    except Exception as exc:
        print(f"  [Wikipedia] {exc}", file=sys.stderr)

    # Trusted-first ordering. The site: query above can only name ~14 domains,
    # so it can't restrict search to all of RSP_TRUSTED. Instead we sort the
    # whole pooled result set by credibility (trust_score): any WP:RSP-reliable
    # / fact-checker / authority URL that surfaced in *either* query floats to
    # the front, so trusted sources survive the urls[:n+5] cutoff and are
    # fetched/scored first. Stable sort keeps within-query order on ties.
    # (The same credibility also weights the verdict in _aggregate, so trusted
    # sources come FIRST here and count for MORE there.)
    urls.sort(key=_credibility, reverse=True)
    n_trusted = sum(1 for u in urls[:n + 5] if _credibility(u) >= 2.0)
    print(f"  [rank] {n_trusted}/{len(urls[:n + 5])} kept URLs are trusted "
          f"(credibility ≥ 2.0)", file=sys.stderr)
    return urls[:n + 5]


# ── Download & extract ────────────────────────────────────────────────────────

def _fetch_extract(url: str) -> tuple[str, str]:
    try:
        resp = httpx.get(url, headers={"User-Agent": _UA},
                         timeout=15, follow_redirects=True)
        import trafilatura  # type: ignore
        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=True, favor_recall=True)
        if not text or len(text.strip()) < 150:
            return "", ""
        meta = trafilatura.extract_metadata(resp.text)
        return (meta.title or "") if meta else "", text.strip()
    except Exception:
        return "", ""


# ── Chunk → embed → select ────────────────────────────────────────────────────
MIN_RELEVANCE = 0.30


def _select_best_evidence(
    claim: str, pages: list[dict], top_k: int, chunks_per_page: int = 2,
) -> list[dict]:
    claim_emb  = _embed([claim])[0]
    candidates: list[dict] = []

    for page in pages:
        chunks = _chunk_text(page["text"])
        if not chunks:
            continue
        chunk_embs = _embed(chunks)
        scores     = chunk_embs @ claim_emb
        for idx in np.argsort(scores)[::-1][:chunks_per_page]:
            score = float(scores[idx])
            if score < MIN_RELEVANCE:
                continue
            candidates.append({
                "url":             page["url"],
                "title":           page["title"],
                "text":            chunks[idx],
                "retrieval_score": round(score, 4),
                "credibility":     _credibility(page["url"]),
            })

    if not candidates:
        return []

    reranker = _get_reranker()
    if reranker and len(candidates) > 1:
        try:
            pairs     = [(claim, c["text"][:512]) for c in candidates]
            logits    = reranker.predict(pairs)
            ce_scores = [1.0 / (1.0 + math.exp(-float(l))) for l in logits]
            for c, s in zip(candidates, ce_scores):
                c["retrieval_score"] = round(0.6 * c["retrieval_score"] + 0.4 * s, 4)
        except Exception:
            pass

    candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)

    # Domain dedup: keep at most MAX_PER_DOMAIN chunks per source so a single
    # site can't dominate the vote (e.g. 3 identical articles from one domain).
    MAX_PER_DOMAIN = 2
    per_domain: dict[str, int] = {}
    deduped: list[dict] = []
    for c in candidates:
        d = c["url"].split("/")[2].replace("www.", "")
        if per_domain.get(d, 0) >= MAX_PER_DOMAIN:
            continue
        per_domain[d] = per_domain.get(d, 0) + 1
        deduped.append(c)
        if len(deduped) >= top_k:
            break
    return deduped


# ── 3-Class aggregation ───────────────────────────────────────────────────────
NEI_MARGIN           = 0.55
STRONG_REF_THRESHOLD = 0.85
STRONG_REF_SHARE     = 0.35

# Multi-source verification thresholds
MIN_INDEPENDENT_SOURCES = 2     # distinct domains that must agree for high confidence
TRUSTED_CREDIBILITY     = 2.0   # credibility >= this counts as a "trusted" source


def _aggregate(evidence: list[dict], preds: list[dict]) -> tuple[str, float]:
    """
    3-class weighted aggregation.

    Key difference from binary aggregation:
        denominator = weighted_SUP + weighted_REF + weighted_NEI
    When the model is genuinely uncertain (spreads probability across all 3
    classes), both sup_share and ref_share drop, making NOT ENOUGH INFO more
    likely — which is the correct behaviour.

    Strong-refutation override still applies for myth-busting cases.
    Source credibility multiplier applied to all weights.
    """
    weights = [ev["retrieval_score"] * ev.get("credibility", 1.0) for ev in evidence]

    sup = sum(p["probabilities"].get("SUPPORTS",        0.0) * w for p, w in zip(preds, weights))
    ref = sum(p["probabilities"].get("REFUTES",         0.0) * w for p, w in zip(preds, weights))
    nei = sum(p["probabilities"].get("NOT ENOUGH INFO", 0.0) * w for p, w in zip(preds, weights))

    total = sup + ref + nei
    if total < 1e-6:
        return "NOT ENOUGH INFO", 0.0

    sup_share = sup / total
    ref_share = ref / total

    # ── Strong-refutation override ────────────────────────────────────────────
    max_ref_prob = max(p["probabilities"].get("REFUTES", 0.0) for p in preds)
    if max_ref_prob >= STRONG_REF_THRESHOLD and ref_share >= STRONG_REF_SHARE:
        return "REFUTES", max_ref_prob

    # ── Normal weighted verdict ───────────────────────────────────────────────
    if sup_share >= NEI_MARGIN:
        return "SUPPORTS", sup_share
    if ref_share >= NEI_MARGIN:
        return "REFUTES", ref_share
    return "NOT ENOUGH INFO", max(sup_share, ref_share)


def _verify_sources(evidence: list[dict], preds: list[dict],
                    verdict: str) -> dict:
    """Multi-source verification.

    Counts how many *distinct* domains agree with the verdict, how many of
    those are trusted, and whether any trusted source contradicts it. A claim
    backed by one site is weaker than the same claim backed by four
    independent ones — this turns that intuition into an explainable score.
    """
    agree_domains: set[str] = set()
    trusted_agree: set[str] = set()
    trusted_contradict: set[str] = set()

    for ev, pred in zip(evidence, preds):
        domain = ev["url"].split("/")[2].replace("www.", "")
        label  = pred["label"]
        cred   = ev.get("credibility", 1.0)
        if label == verdict:
            agree_domains.add(domain)
            if cred >= TRUSTED_CREDIBILITY:
                trusted_agree.add(domain)
        elif label in ("SUPPORTS", "REFUTES") and verdict in ("SUPPORTS", "REFUTES"):
            # a contradicting definite verdict from a trusted source
            if cred >= TRUSTED_CREDIBILITY:
                trusted_contradict.add(domain)

    n_independent = len(agree_domains)
    enough_sources = n_independent >= MIN_INDEPENDENT_SOURCES
    has_trusted    = len(trusted_agree) > 0

    # Verified = enough independent agreement, at least one trusted source,
    # and no trusted source pulling the other way.
    verified = (
        verdict in ("SUPPORTS", "REFUTES")
        and enough_sources
        and has_trusted
        and not trusted_contradict
    )

    if verdict == "NOT ENOUGH INFO":
        statement = "Not enough reliable evidence to verify this claim."
    elif verified:
        statement = (
            f"{verdict} — {n_independent} independent sources agree "
            f"({len(trusted_agree)} trusted)."
        )
    elif n_independent >= 1 and not has_trusted:
        statement = (
            f"Leans {verdict}, but only low-authority sources agree "
            f"({n_independent}). Treat with caution."
        )
    elif trusted_contradict:
        statement = (
            f"Conflicting evidence: {verdict} disputed by "
            f"{len(trusted_contradict)} trusted source(s)."
        )
    else:
        statement = f"Leans {verdict} ({n_independent} source(s)) — weak support."

    return {
        "verified":            verified,
        "independent_sources": n_independent,
        "trusted_agree":       sorted(trusted_agree),
        "trusted_contradict":  sorted(trusted_contradict),
        "statement":           statement,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run(claim: str, top_k: int = 5, verbose: bool = False) -> dict:
    W = 72
    # Preprocess: a pasted post → clean hypothesis + focused search query.
    cleaned = _clean_claim(claim)
    query   = _build_search_query(claim)   # uses raw post for hashtag extraction

    print(f"\n{'═' * W}")
    print(f"  {_C['BOLD']}POST{_C['RESET']}  {claim[:W - 8]}")
    for off in range(W - 8, min(len(claim), W * 4), W):
        print(f"        {claim[off:off + W]}")
    if cleaned != claim.strip():
        print(f"  {_C['DIM']}CLAIM  {cleaned[:W * 2]}{_C['RESET']}")
    print(f"  {_C['DIM']}QUERY  {query}{_C['RESET']}")
    print(f"{'═' * W}")

    # 1. Search (use the extracted keyword query, not the raw post)
    print(f"\n  {_C['BOLD']}[1/4] Searching the web…{_C['RESET']}")
    t0   = time.monotonic()
    urls = _search_web(query, n=top_k + 5)
    print(f"       {len(urls)} URLs  ({time.monotonic() - t0:.1f}s)")

    # 2. Download
    print(f"\n  {_C['BOLD']}[2/4] Downloading & extracting…{_C['RESET']}")
    pages: list[dict] = []
    for i, url in enumerate(urls, 1):
        title, text = _fetch_extract(url)
        domain = url.split("/")[2].replace("www.", "")
        if not text:
            print(f"       [{i:2}] {_C['DIM']}SKIP{_C['RESET']}  {domain}")
            continue
        pages.append({"url": url, "title": title, "text": text})
        snippet = text[:80].replace("\n", " ")
        print(f"       [{i:2}] {_C['DIM']}OK {len(text):>6} chars{_C['RESET']}"
              f"  {domain:<32}  \"{snippet}…\"")

    if not pages:
        print("\n       No pages could be downloaded.")
        return _result(claim, "NOT ENOUGH INFO", 0.0, [], "No content retrieved.")

    # 3. Chunk → embed → select
    print(f"\n  {_C['BOLD']}[3/4] Chunking, embedding, selecting passages…{_C['RESET']}")
    t1       = time.monotonic()
    evidence = _select_best_evidence(cleaned, pages, top_k)
    print(f"       {len(evidence)}/{len(pages)} pages passed relevance filter "
          f"(threshold={MIN_RELEVANCE})  ({time.monotonic() - t1:.1f}s)")

    if not evidence:
        return _result(claim, "NOT ENOUGH INFO", 0.0, [],
                       "No relevant evidence found.")

    for ev in evidence:
        domain  = ev["url"].split("/")[2].replace("www.", "")
        snippet = ev["text"][:90].replace("\n", " ")
        cred    = ev["credibility"]
        ctag    = f" [×{cred:.0f}]" if cred >= 2.0 else ""
        print(f"       {ev['retrieval_score']:.3f}  {domain:<32}{ctag}  \"{snippet}…\"")

    # 4. NLI
    print(f"\n  {_C['BOLD']}[4/4] Running 3-class NLI on {len(evidence)} chunks…{_C['RESET']}")
    texts = [ev["text"] for ev in evidence]
    t2    = time.monotonic()
    preds = _predict_nli(cleaned, texts)
    print(f"       NLI done  ({time.monotonic() - t2:.1f}s)")

    verdict, conf = _aggregate(evidence, preds)
    verification  = _verify_sources(evidence, preds, verdict)

    # Build rich output
    rich: list[dict] = []
    for ev, pred in zip(evidence, preds):
        rich.append({
            "url":             ev["url"],
            "title":           ev["title"],
            "retrieval_score": ev["retrieval_score"],
            "credibility":     ev["credibility"],
            "nli_label":       pred["label"],
            "confidence":      pred["score"],
            "probabilities":   pred["probabilities"],
            "snippet":         ev["text"][:300],
        })

    # Verdict banner
    badge = f"{_C['BOLD']}✓ VERIFIED{_C['RESET']}" if verification["verified"] else f"{_C['DIM']}unverified{_C['RESET']}"
    print(f"\n  {'─' * W}")
    print(f"  VERDICT    {_col(verdict, verdict)}   {badge}")
    print(f"  CONFIDENCE {conf * 100:.1f}%")
    print(f"  SOURCES    {verification['independent_sources']} independent"
          f"  ({len(verification['trusted_agree'])} trusted)")
    print(f"  CREDIBILITY {verification['statement']}")
    print(f"  EVIDENCE   {len(evidence)} chunks from {len(pages)} pages")
    print(f"  {'─' * W}\n")

    for r in rich:
        domain  = r["url"].split("/")[2].replace("www.", "")
        lbl     = r["nli_label"]
        lbl_col = _col(lbl, f"[{lbl}]")
        sup_pct = r["probabilities"].get("SUPPORTS",        0) * 100
        ref_pct = r["probabilities"].get("REFUTES",         0) * 100
        nei_pct = r["probabilities"].get("NOT ENOUGH INFO", 0) * 100
        cred    = r["credibility"]
        ctag    = f" {_C['BOLD']}[trusted×{cred:.0f}]{_C['RESET']}" if cred >= 2.0 else ""
        print(f"  {lbl_col:<28} S:{sup_pct:>5.1f}%  R:{ref_pct:>5.1f}%  N:{nei_pct:>4.1f}%"
              f"  ret={r['retrieval_score']:.3f}  {domain}{ctag}")
        if verbose:
            dim, rst = _C["DIM"], _C["RESET"]
            print(f"  {dim}\"{r['snippet'][:120]}…{rst}")

    print(f"  {'─' * W}\n")
    return _result(claim, verdict, conf, rich, verification=verification)


def _result(claim, verdict, conf, evidence, note="", verification=None) -> dict:
    return {
        "claim":            claim,
        "final_prediction": verdict,
        "confidence":       round(conf, 4),
        "evidence":         evidence,
        "verification":     verification or {},
        "note":             note,
    }


# ── Built-in test battery ─────────────────────────────────────────────────────
TESTS = [
    {"claim": "The RMS Titanic sank on April 15, 1912, after hitting an iceberg.",
     "expected": "SUPPORTS", "note": "Core historical fact"},
    {"claim": "The Titanic was built and launched in Southampton, England.",
     "expected": "REFUTES",  "note": "Built in Belfast (Harland & Wolff)"},
    {"claim": "Over 1,500 people died in the Titanic disaster.",
     "expected": "SUPPORTS", "note": "Well-documented death toll"},
    {"claim": "The Titanic had enough lifeboats for all passengers and crew on board.",
     "expected": "REFUTES",  "note": "Famous for having far too few lifeboats"},
    {"claim": "Albert Einstein failed mathematics in school as a child.",
     "expected": "REFUTES",  "note": "Common myth — he actually excelled"},
    {"claim": "Vaccines cause autism.",
     "expected": "REFUTES",  "note": "Debunked by extensive scientific research"},
    {"claim": "The Great Wall of China is visible from space with the naked eye.",
     "expected": "REFUTES",  "note": "NASA confirmed it is NOT visible"},
]


# ── Social-media-style misinformation battery ─────────────────────────────────
# The kind of viral claims that spread on Facebook/WhatsApp/X. These exercise
# the trust weighting (fact-checkers like snopes/reuters debunk them, while
# social platforms amplify them) and the multi-source verification logic.
SOCIAL_TESTS = [

    # ── Government misinformation ─────────────────────────────────────────────
    {"claim": (
        "BREAKING 🚨 The government has been adding fluoride to our drinking "
        "water since the 1940s and Harvard studies PROVE it lowers children's IQ!! "
        "It's literally classified as a neurotoxin. The same people pushing "
        "fluoride are pushing vaccines and chemtrails. Wake up — they're "
        "poisoning us on purpose!! SHARE before this gets removed #fluoride "
        "#waterispoison #governmentlies"),
     "expected": "REFUTES", "note": "Fluoride mind-control / IQ conspiracy"},

    {"claim": (
        "🔴 PROOF the 2020 election was STOLEN!! Dominion voting machines were "
        "secretly connected to the internet and flipped millions of votes from "
        "Trump to Biden. Sidney Powell has the server evidence and the mainstream "
        "media is suppressing it. 74 million Americans were cheated. This is "
        "treason!! REPOST before they delete it #StopTheSteal #ElectionFraud #Dominion"),
     "expected": "REFUTES", "note": "2020 election fraud / Dominion machines conspiracy"},

    {"claim": (
        "Obama was NOT born in the United States — his own Kenyan grandmother "
        "said on tape she was PRESENT at his birth in Mombasa!! His birth "
        "certificate posted online is a proven digital forgery — multiple forensic "
        "experts confirmed the layers in the PDF. He was never legally eligible "
        "to be president. The entire presidency was unconstitutional!! "
        "#Birther #Obama #IllegalPresident"),
     "expected": "REFUTES", "note": "Obama birther conspiracy"},

    # ── Celebrity misinformation ──────────────────────────────────────────────
    {"claim": (
        "Y'ALL the Pentagon confirmed Taylor Swift is a psychological operation 😱 "
        "The Travis Kelce romance was STAGED to push vaccine propaganda and get "
        "young people to vote Democrat. She has government security clearances "
        "and reports directly to the DOD. This is not a theory anymore — "
        "journalists have the receipts!! #TaylorSwift #Psyop #DeepState #NFL"),
     "expected": "REFUTES", "note": "Taylor Swift Pentagon psyop conspiracy"},

    {"claim": (
        "The real Paul McCartney DIED in a car crash in November 1966 and was "
        "replaced by a lookalike named William Campbell 😢 The Beatles hid clues "
        "everywhere — he's barefoot on the Abbey Road cover, John whispers "
        "'I buried Paul' on Strawberry Fields, and the license plate reads "
        "28 IF (as in 28 IF Paul had lived). Listen again and you'll hear it. "
        "#PaulIsDead #Beatles #HiddenTruth"),
     "expected": "REFUTES", "note": "Paul McCartney death hoax conspiracy"},

    {"claim": (
        "Beyoncé NEVER carried Blue Ivy herself — her baby bump literally "
        "FOLDED on live TV because it was a PROSTHETIC 😂😂 She hired a "
        "surrogate and faked the whole pregnancy for publicity. They couldn't "
        "even get the prop right on camera. Hollywood elites do this all the "
        "time and the media covers for them!! #Beyonce #FakePregnancy "
        "#SurrogateGate #CelebrityLies"),
     "expected": "REFUTES", "note": "Beyoncé fake pregnancy conspiracy"},

    # ── TRUE — Government / history ───────────────────────────────────────────
    {"claim": (
        "On this day in history 🕊️ Nelson Mandela walked free from Victor Verster "
        "Prison on February 11, 1990, after 27 years of imprisonment for his "
        "anti-apartheid activism. His release was broadcast live around the world "
        "and marked a turning point for South Africa. One of the most powerful "
        "moments of the 20th century. #Mandela #SouthAfrica #Freedom"),
     "expected": "SUPPORTS", "note": "TRUE — Mandela release date and imprisonment length"},

    {"claim": (
        "November 9, 1989 — the Berlin Wall fell 🧱✊ After 28 years of dividing "
        "East and West Germany, thousands of citizens began tearing it down with "
        "their own hands after the East German government announced open borders. "
        "It marked the beginning of the end of the Cold War. An extraordinary "
        "night that changed history forever. #BerlinWall #ColdWar #History"),
     "expected": "SUPPORTS", "note": "TRUE — Berlin Wall fall date and context"},

    {"claim": (
        "A reminder for everyone 🚀 The Apollo 11 mission successfully landed "
        "astronauts on the Moon on July 20, 1969. Neil Armstrong was the first "
        "human to set foot on the lunar surface, followed by Buzz Aldrin. "
        "Michael Collins orbited above in the command module. NASA has preserved "
        "all original mission data and footage. #Apollo11 #MoonLanding #NASA"),
     "expected": "SUPPORTS", "note": "TRUE — Apollo 11 moon landing date and crew"},

    {"claim": (
        "The US Congress certified the 2020 presidential election results on "
        "January 7, 2021, confirming Joe Biden as the winner with 306 electoral "
        "votes. All 50 states had certified their results. More than 60 lawsuits "
        "challenging the election were dismissed by courts, including judges "
        "appointed by both parties. #Election2020 #ElectoralCollege"),
     "expected": "SUPPORTS", "note": "TRUE — 2020 election certification facts"},

    # ── TRUE — Celebrity ──────────────────────────────────────────────────────
    {"claim": (
        "Just a reminder that Elvis Presley passed away on August 16, 1977, "
        "at his Graceland estate in Memphis, Tennessee. The official cause of "
        "death was cardiac arrhythmia. He was 42 years old. Rest in peace, "
        "the King of Rock and Roll 🎸👑"),
     "expected": "SUPPORTS", "note": "TRUE — Elvis death date and cause"},

    {"claim": (
        "Sending so much love to Celine Dion 💙 She confirmed she has been "
        "diagnosed with Stiff Person Syndrome, an extremely rare neurological "
        "disorder that affects roughly 1 in a million people and causes severe "
        "muscle stiffness and spasms. This is why she had to cancel her world "
        "tour. Wishing her strength and recovery 🙏 #CelineDion #StiffPersonSyndrome"),
     "expected": "SUPPORTS", "note": "TRUE — Celine Dion Stiff Person Syndrome diagnosis"},

    {"claim": (
        "Robin Williams passed away on August 11, 2014 at his home in Paradise "
        "Cay, California 💙 His death was ruled a suicide. His wife later revealed "
        "he had been suffering from Lewy Body Dementia — a devastating disease "
        "that had gone undiagnosed. His family shared his story publicly to raise "
        "awareness. Truly one of the greatest 🌟 #RobinWilliams #LewyBodyDementia"),
     "expected": "SUPPORTS", "note": "TRUE — Robin Williams death and Lewy Body Dementia"},
]


def run_battery(top_k: int = 5, verbose: bool = False, json_out: str | None = None,
                tests: list[dict] | None = None, title: str = "") -> None:
    tests = tests if tests is not None else TESTS
    if title:
        print(f"\n  {_C['BOLD']}{title}{_C['RESET']}")
    print(f"\n  Warming up models for {len(tests)} tests…")
    _get_embedder()
    _get_reranker()
    _get_nli()

    results, passed = [], 0
    for tc in tests:
        result   = run(tc["claim"], top_k=top_k, verbose=verbose)
        actual   = result["final_prediction"]
        expected = tc["expected"]
        ok       = actual == expected
        if ok:
            passed += 1
        result.update({"expected": expected, "passed": ok, "note": tc["note"]})
        results.append(result)
        status = _col("SUPPORTS", "✓ PASS") if ok else _col("REFUTES", "✗ FAIL")
        print(f"  {status}  expected={expected:<16} actual={actual:<16} — {tc['note']}")

    print(f"\n  {'=' * 50}")
    print(f"  {passed}/{len(tests)} passed")
    print(f"  {'=' * 50}\n")

    if json_out:
        Path(json_out).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Results written to {json_out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Live web fact-checker — 3-class DeBERTa model."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--claim",  help="Single claim to verify.")
    mode.add_argument("--batch",  action="store_true",
                      help="Run the built-in test battery.")
    mode.add_argument("--social", action="store_true",
                      help="Run the social-media misinformation battery.")
    parser.add_argument("--top-k",   type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json",    metavar="PATH",
                        help="Write full results to a JSON file.")
    args = parser.parse_args()

    if args.batch:
        run_battery(top_k=args.top_k, verbose=args.verbose, json_out=args.json)
    elif args.social:
        run_battery(top_k=args.top_k, verbose=args.verbose, json_out=args.json,
                    tests=SOCIAL_TESTS, title="SOCIAL-MEDIA MISINFORMATION BATTERY")
    elif args.claim:
        result = run(args.claim, top_k=args.top_k, verbose=args.verbose)
        if args.json:
            Path(args.json).write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Results written to {args.json}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()