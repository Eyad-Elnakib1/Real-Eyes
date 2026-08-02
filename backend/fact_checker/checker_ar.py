"""
checker_ar.py
--------------------
Arabic-aware wrapper around checker_en.py.

Adds ONE capability: Groq-powered Arabic→English translation, so an Arabic
claim is verified by the English DeBERTa model exactly as an English claim.

The SEARCH stays in the claim's ORIGINAL language (the top priority): the query
is built from the untranslated Arabic claim, so Arabic-language sources are
retrieved. The trusted-first search also includes Arabic trusted sources
(utils.trust.TRUSTED_AR — Arabic fact-checkers + reputable Arabic news), since
WP:RSP alone has no coverage of Arabic regional topics.

Translation (Groq) is applied to English at three points, kept cheap by only
ever touching Arabic text:

  1. The retrieved ARABIC pages → English (capped at _MAX_PAGE_CHARS) right after
     download, BEFORE embedding — so an Arabic source can actually be selected
     by the English-centric embedder. English pages skip Groq (free).
  2. The CLAIM → English, so selection compares English claim vs English pages.
  3. Any selected chunk still containing Arabic (beyond the cap) → English, as a
     safety net just before NLI.

Auto-detection: only Arabic text is sent to Groq. Graceful fallback: missing
key / API error → original text used (pipeline still runs).

Auto-detection: only text containing Arabic characters is sent to Groq, so
English claims/evidence skip the API entirely (no cost, no latency).

Graceful fallback: if GROQ_API_KEY is unset or the API errors, the original
text is used unchanged (a warning is logged) — the pipeline still runs.

Everything else (search, download, selection, NLI, aggregation, verification,
test batteries) is reused unchanged from the base module.

Usage:
    setx GROQ_API_KEY "your-key"      (Windows, once)
    python checker_ar.py --claim "لقاحات كوفيد تسبب التوحد"
    python checker_ar.py --claim "..." --verbose
    python checker_ar.py --social
    python checker_ar.py --batch
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time

# ── Groq API keys (loaded from environment) ───────────────────────────────────
# Set GROQ_API_KEY in your environment (single key) or GROQ_API_KEYS (comma-
# separated list for rotation). Never hardcode keys in source code.
_env_keys = os.environ.get("GROQ_API_KEYS", "").strip()
if _env_keys:
    GROQ_API_KEYS = [k.strip() for k in _env_keys.split(",") if k.strip()]
else:
    GROQ_API_KEYS = [os.environ.get("GROQ_API_KEY", "").strip()]
    GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]

import checker_en as base
from checker_en import (
    _C, _col, _credibility, _search_web, _fetch_extract, _select_best_evidence,
    _predict_nli, _aggregate, _verify_sources, _result, _clean_claim,
    _build_search_query, MIN_RELEVANCE, SOCIAL_TESTS, TESTS,
)
from utils.trust import TRUSTED_AR  # Arabic fact-checkers / news (added to trust.py)

# Include Arabic trusted sources in the trusted-first search for this pipeline
# (the base English pipeline is left unchanged).
base._TRUSTED_SEARCH_DOMAINS = base._RSP_TRUSTED | TRUSTED_AR


# ── Groq Arabic→English translation ───────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"   # strong multilingual; override with --model
_MAX_PAGE_CHARS = 6000   # cap per Arabic page sent to Groq (bounds token cost)

# Arabic (incl. supplement) Unicode ranges.
_AR_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

_groq_client = None
_groq_warned = False


def _has_arabic(text: str) -> bool:
    return bool(_AR_RE.search(text or ""))


def _get_groq():
    """Lazy Groq client singleton. Returns None if unavailable (no key / no lib)."""
    global _groq_client, _groq_warned
    if _groq_client is None:
        try:
            from groq import Groq  # type: ignore
        except ImportError:
            if not _groq_warned:
                print("  [groq] 'groq' package not installed — skipping translation",
                      file=sys.stderr)
                _groq_warned = True
            return None
        keys = [k.strip() for k in GROQ_API_KEYS if k and k.strip()]
        key = random.choice(keys) if keys else os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            if not _groq_warned:
                print("  [groq] GROQ_API_KEY not set — using original text",
                      file=sys.stderr)
                _groq_warned = True
            return None
        _groq_client = Groq(api_key=key)
    return _groq_client


_SYS_AR2EN = (
    "You are a precise translator. Translate the user's text from "
    "Arabic to English. Preserve all names, dates, numbers, and the "
    "exact meaning (including negation). Output ONLY the English "
    "translation — no quotes, no notes, no preamble."
)
_SYS_EN2AR = (
    "You are a precise translator. Translate the user's text from "
    "English to Modern Standard Arabic. Preserve all names, dates, "
    "numbers, and the exact meaning (including negation). Output ONLY "
    "the Arabic translation — no quotes, no notes, no preamble."
)


def _translate_one(text: str, client, model: str, system: str = _SYS_AR2EN) -> str:
    """Translate a single string using the given system prompt; original on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,            # deterministic — same input → same output
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": text},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or text
    except Exception as exc:
        print(f"  [groq] translation failed ({exc}) — using original", file=sys.stderr)
        return text


def _translate_arabic(texts: list[str], model: str = GROQ_MODEL) -> list[str]:
    """Translate only the Arabic items in `texts` to English; non-Arabic pass through."""
    if not any(_has_arabic(t) for t in texts):
        return list(texts)
    client = _get_groq()
    if client is None:
        return list(texts)
    return [_translate_one(t, client, model, _SYS_AR2EN) if _has_arabic(t) else t
            for t in texts]


def _translate_to_arabic(text: str, model: str = GROQ_MODEL) -> str:
    """Translate an English string to Arabic; original on failure / no client."""
    if not text or _has_arabic(text):
        return text
    client = _get_groq()
    if client is None:
        return text
    return _translate_one(text, client, model, _SYS_EN2AR)


def _bilingual_claim(claim: str, model: str = GROQ_MODEL) -> tuple[str, str]:
    """Return (claim_en, claim_ar) regardless of the input language.

    If the input is Arabic → translate to English. If English → translate to
    Arabic. If Groq is unavailable, the missing side falls back to the original
    (search in that language is then degraded but the pipeline still runs).
    """
    if _has_arabic(claim):
        claim_ar = claim
        claim_en = _translate_arabic([claim], model)[0]
    else:
        claim_en = claim
        claim_ar = _translate_to_arabic(claim, model)
    return claim_en, claim_ar


# ── Main runner (mirrors base.run, with translation inserted) ──────────────────

def run(claim: str, top_k: int = 5, verbose: bool = False,
        model: str = GROQ_MODEL) -> dict:
    W = 72
    cleaned = _clean_claim(claim)

    # Translate the claim to BOTH languages up front, regardless of input
    # language. Each version drives a separate web search so we retrieve
    # English-language AND Arabic-language sources for every claim.
    claim_en_raw, claim_ar_raw = _bilingual_claim(claim, model)
    claim_en_clean = _clean_claim(claim_en_raw)
    claim_ar_clean = _clean_claim(claim_ar_raw)
    query_en = _build_search_query(claim_en_raw)
    query_ar = _build_search_query(claim_ar_raw)

    print(f"\n{'═' * W}")
    print(f"  {_C['BOLD']}POST{_C['RESET']}  {claim[:W - 8]}")
    for off in range(W - 8, min(len(claim), W * 4), W):
        print(f"        {claim[off:off + W]}")
    if cleaned != claim.strip():
        print(f"  {_C['DIM']}CLAIM   {cleaned[:W * 2]}{_C['RESET']}")
    print(f"  {_C['DIM']}EN      {claim_en_clean[:W * 2]}{_C['RESET']}")
    print(f"  {_C['DIM']}AR      {claim_ar_clean[:W * 2]}{_C['RESET']}")
    print(f"  {_C['DIM']}Q(EN)   {query_en}{_C['RESET']}")
    print(f"  {_C['DIM']}Q(AR)   {query_ar}{_C['RESET']}")
    print(f"{'═' * W}")

    # 1. Search — run both queries and merge URLs (dedup, preserve order).
    print(f"\n  {_C['BOLD']}[1/4] Searching the web (EN + AR)…{_C['RESET']}")
    t0 = time.monotonic()
    urls_en = _search_web(query_en, n=top_k + 5)
    urls_ar = (_search_web(query_ar, n=top_k + 5)
               if query_ar and query_ar != query_en else [])
    seen: set[str] = set()
    urls: list[str] = []
    for u in urls_en + urls_ar:
        if u not in seen:
            seen.add(u); urls.append(u)
    print(f"       {len(urls)} URLs ({len(urls_en)} EN + {len(urls_ar)} AR, "
          f"deduped)  ({time.monotonic() - t0:.1f}s)")

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

    # Translate ONLY the Arabic retrieved pages → English (capped), so Arabic
    # sources can be matched by the English embedder. English pages skip Groq
    # (free); discarded-but-translated cost is bounded by _MAX_PAGE_CHARS and by
    # how few Arabic pages a trusted/English-leaning search returns.
    ar_idx = [i for i, p in enumerate(pages) if _has_arabic(p["text"])]
    if ar_idx:
        print(f"  {_C['DIM']}[groq] translating {len(ar_idx)} Arabic page(s) "
              f"→ English…{_C['RESET']}", file=sys.stderr)
        snippets   = [pages[i]["text"][:_MAX_PAGE_CHARS] for i in ar_idx]
        translated = _translate_arabic(snippets, model)
        for i, t in zip(ar_idx, translated):
            pages[i]["text"] = t

    # Claim is already in English from the up-front bilingual translation.
    claim_en = claim_en_clean

    # 3. Chunk → embed → select against the English claim.
    print(f"\n  {_C['BOLD']}[3/4] Chunking, embedding, selecting passages…{_C['RESET']}")
    t1       = time.monotonic()
    evidence = _select_best_evidence(claim_en, pages, top_k)
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

    # Translate ONLY the selected chunks that are still Arabic (e.g. an Arabic
    # source that made the top-k) just before NLI. The claim is already English.
    texts = [ev["text"] for ev in evidence]
    n_ar  = sum(1 for t in texts if _has_arabic(t))
    if n_ar:
        print(f"  {_C['DIM']}[groq] translating {n_ar} selected chunk(s) "
              f"→ English…{_C['RESET']}", file=sys.stderr)
        texts = _translate_arabic(texts, model)

    # 4. NLI (English claim vs English evidence)
    print(f"\n  {_C['BOLD']}[4/4] Running 3-class NLI on {len(evidence)} chunks…{_C['RESET']}")
    t2    = time.monotonic()
    preds = _predict_nli(claim_en, texts)
    print(f"       NLI done  ({time.monotonic() - t2:.1f}s)")

    verdict, conf = _aggregate(evidence, preds)
    verification  = _verify_sources(evidence, preds, verdict)

    rich: list[dict] = []
    for ev, pred, txt in zip(evidence, preds, texts):
        rich.append({
            "url":             ev["url"],
            "title":           ev["title"],
            "retrieval_score": ev["retrieval_score"],
            "credibility":     ev["credibility"],
            "nli_label":       pred["label"],
            "confidence":      pred["score"],
            "probabilities":   pred["probabilities"],
            "snippet":         txt[:300],
        })

    badge = (f"{_C['BOLD']}✓ VERIFIED{_C['RESET']}" if verification["verified"]
             else f"{_C['DIM']}unverified{_C['RESET']}")
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


# ── Bilingual social-media news battery (3 English + 3 Arabic, mixed truth) ───
# Viral-post style (emojis, hashtags, engagement bait) like SOCIAL_TESTS, so the
# preprocessing/translation path is exercised the same way real posts hit it.
NEWS_TESTS = [
    # ── English ───────────────────────────────────────────────────────────────
    {"claim": (
        "On this day 🚀 NASA's Perseverance rover touched down on Mars on "
        "February 18, 2021 after a 7-month journey! The 'seven minutes of terror' "
        "landing went flawlessly and it's still exploring Jezero Crater today. "
        "Humanity on another planet 🔴🛰️ #Mars #Perseverance #NASA"),
     "expected": "SUPPORTS", "note": "EN TRUE — Perseverance Mars landing date"},

    # {"claim": (
    #     "What a summer ☀️🏅 Paris hosted the 2024 Summer Olympic Games and that "
    #     "opening ceremony floating down the River Seine was UNREAL! France threw "
    #     "an unforgettable show for the whole world 🇫🇷 #Paris2024 #Olympics"),
    #  "expected": "SUPPORTS", "note": "EN TRUE — Paris 2024 Summer Olympics"},

    # {"claim": (
    #     "WAKE UP 🚨 Doctors are HIDING this!! Drinking a little bleach KILLS the "
    #     "COVID virus in your body within hours — Big Pharma buries it because "
    #     "they can't profit from a cheap cure!! My cousin tried it and recovered. "
    #     "SHARE to save lives before they delete this!! #COVID #NaturalCure #DoYourResearch"),
    #  "expected": "REFUTES", "note": "EN FALSE — bleach 'cure', dangerous & debunked"},

    # ── Arabic ────────────────────────────────────────────────────────────────
    {"claim": (
        "ذكرى لا تُنسى ⚽🏆 استضافت قطر بطولة كأس العالم لكرة القدم 2022، وكانت "
        "أول دولة عربية تنظّم الحدث على الإطلاق! حفل الافتتاح كان مبهراً وتوّجت "
        "الأرجنتين باللقب. فخر لكل العرب 🇶🇦 #كأس_العالم#قطر2022"),
     "expected": "SUPPORTS", "note": "AR TRUE — 2022 FIFA World Cup held in Qatar"},

    # {"claim": (
    #     "أخيراً 🚇✨ افتُتح مترو الرياض للجمهور خلال عام 2024 بشبكة ضخمة تغطي "
    #     "العاصمة بالكامل! نقلة حضارية كبرى للمملكة وتسهيل هائل للتنقل اليومي. "
    #     "مبروك للرياض 🇸🇦 #مترو_الرياض #الرياض"),
    #  "expected": "SUPPORTS", "note": "AR TRUE — Riyadh Metro opened to public in 2024"},

    # {"claim": (
    #     "عاجل 🚨 الأطباء يخفون الحقيقة عنكم!! لقاح كورونا يسبب العقم عند النساء "
    #     "ويدمّر الخصوبة نهائياً — الشركات تُخفي الدراسات والإعلام متواطئ!! انشروا "
    #     "قبل الحذف لتحذير بناتكم وأخواتكم!! #لا_للقاح #كورونا #استفيقوا"),
    #  "expected": "REFUTES", "note": "AR FALSE — vaccine/infertility myth, debunked"},
]


def run_battery(tests: list[dict], verbose: bool = False,
                model: str = GROQ_MODEL) -> None:
    print(f"\n  {_C['BOLD']}ARABIC-AWARE FACT-CHECK BATTERY (web4_ar){_C['RESET']}")
    print(f"\n  Warming up models for {len(tests)} tests…")
    base._get_embedder(); base._get_reranker(); base._get_nli()

    passed = 0
    for tc in tests:
        result   = run(tc["claim"], verbose=verbose, model=model)
        actual   = result["final_prediction"]
        expected = tc["expected"]
        ok       = actual == expected
        passed  += ok
        status = _col("SUPPORTS", "✓ PASS") if ok else _col("REFUTES", "✗ FAIL")
        print(f"  {status}  expected={expected:<16} actual={actual:<16} — {tc['note']}")

    print(f"\n  {'=' * 50}")
    print(f"  {passed}/{len(tests)} passed")
    print(f"  {'=' * 50}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Arabic-aware live web fact-checker (Groq translation + 3-class DeBERTa)."
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--claim",  type=str, help="single claim to check (Arabic or English)")
    mode.add_argument("--news",   action="store_true",
                      help="run the bilingual social-media news battery (EN + AR)")
    mode.add_argument("--social", action="store_true", help="run the social battery")
    mode.add_argument("--batch",  action="store_true", help="run the built-in battery")
    ap.add_argument("--model",   default=GROQ_MODEL, help=f"Groq model (default: {GROQ_MODEL})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.news:
        run_battery(NEWS_TESTS, verbose=args.verbose, model=args.model)
    elif args.social:
        run_battery(SOCIAL_TESTS, verbose=args.verbose, model=args.model)
    elif args.batch:
        run_battery(TESTS, verbose=args.verbose, model=args.model)
    elif args.claim:
        run(args.claim, verbose=args.verbose, model=args.model)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
