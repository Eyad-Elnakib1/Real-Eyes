"""
Local HTTP bridge for the "Real Eyes" Chrome extension.

It imports `run_inference` from test_classification.py WITHOUT modifying it,
saves the incoming image to a temp file, calls the function, and returns
the result as JSON.

Run it once before using the extension:

    cd D:\\grad_fainal_work\\classification_model
    pip install flask flask-cors requests
    python server.py

The server listens on http://127.0.0.1:5001
"""

import base64
import concurrent.futures
import os
import re
import sys
import tempfile
import threading
import time as _time
import traceback
import types as _types
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Portable project root: server.py lives at <root>/classification_model/server.py,
# so .parent.parent is the project root regardless of drive / OS / install path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the segmentation + screenshot + news-verification packages importable.
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "models" / "classification"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "models" / "segmentation"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "screenshot"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "fact_checker"))

# Import the user's file as-is. Do NOT modify test_classification.py.
# We reuse its building blocks so the model loads ONCE per process.
from test_classification import (
    OUT_DIR,
    CKPT_PATH,
    USE_EMA,
    CLASS_NAMES,
    SwinForensicModel,
    predict as model_predict,
    load_image,
    make_report,
)

# Segmentation pipeline (same import-only pattern)
from test_segmentation import (
    load_model       as seg_load_model,
    preprocess_image as seg_preprocess,
    predict_mask     as seg_predict_mask,
    analyze_regions  as seg_analyze_regions,
    save_heatmap_card as seg_save_card,
)

SEG_CKPT_PATH = str(PROJECT_ROOT / "backend" / "models" / "segmentation" / "checkpoint_epoch_21.pth")
REAL_TRIGGER_THRESHOLD = 0.85   # if real_prob <= 85%, also run segmentation
_SERVER_START_TIME = _time.time()



# Use the real llm.py (process_news cleans OCR + translates Arabic→English).
# Fall back to a pass-through stub only if groq isn't installed.
try:
    import llm as _real_llm  # noqa: F401  (registers in sys.modules)
    print("[server] real llm.py loaded (groq available)")
except Exception as _llm_err:
    if "llm" not in sys.modules:
        print(f"[server] llm.py unavailable ({_llm_err}); using pass-through stub",
              file=sys.stderr)
        _llm_stub = _types.ModuleType("llm")
        _llm_stub.process_news = lambda text: (text, text)
        sys.modules["llm"] = _llm_stub

from handler import run_extraction

# News-verification pipeline (Arabic-aware wrapper around the DeBERTa fact-checker).
from checker_ar import run as verify_claim  # noqa: E402
import checker_ar as _news_ar                # for _bilingual_claim etc.
import checker_en as _news_base      # for warm-up + internals

_news_lock = threading.Lock()

# --- URL fetch cache ------------------------------------------------------
URL_CACHE_TTL = 3600           # seconds
URL_CACHE_MAX = 200            # bounded LRU-ish
_url_cache: dict = {}          # url -> (timestamp, (title, text))
_url_cache_lock = threading.Lock()

def _cached_fetch_extract(url: str):
    now = _time.time()
    with _url_cache_lock:
        cached = _url_cache.get(url)
        if cached and now - cached[0] < URL_CACHE_TTL:
            return cached[1]
    try:
        result = _news_base._fetch_extract(url)
    except Exception:
        result = (None, None)
    with _url_cache_lock:
        _url_cache[url] = (now, result)
        if len(_url_cache) > URL_CACHE_MAX:
            oldest = sorted(_url_cache.items(), key=lambda kv: kv[1][0])[:50]
            for k, _ in oldest:
                _url_cache.pop(k, None)
    return result


def _parallel_fetch(urls, *, overall_timeout=20.0, max_workers=8, progress_cb=None):
    """Fetch URLs concurrently. Returns list[{url,title,text}] for successes."""
    pages = []
    completed = 0
    total = len(urls)
    if not total:
        return pages
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_url = {pool.submit(_cached_fetch_extract, u): u for u in urls}
        try:
            for fut in concurrent.futures.as_completed(future_to_url,
                                                       timeout=overall_timeout):
                completed += 1
                url = future_to_url[fut]
                try:
                    title, text = fut.result()
                    if text:
                        pages.append({"url": url, "title": title, "text": text})
                except Exception:
                    pass
                if progress_cb:
                    progress_cb(completed, total)
        except concurrent.futures.TimeoutError:
            # Drop whatever is still pending — partial result is fine.
            for fut in future_to_url:
                if not fut.done():
                    fut.cancel()
    return pages


# --- Split a packed region into sub-photos via seam detection -------------
# grad1.detect_image_regions uses a 40x40 morphological close that merges
# tightly-packed photos (e.g. wedding collages with 1-2 px borders) into a
# single region. We post-process here without touching grad1.py.

def _find_low_var_bands(stds, threshold=8.0, min_thickness=1):
    """Return [(start, end), ...] index ranges where std stays below threshold."""
    bands = []
    in_band = False
    s = 0
    for i, sd in enumerate(stds):
        if sd < threshold and not in_band:
            in_band = True
            s = i
        elif sd >= threshold and in_band:
            in_band = False
            if i - s >= min_thickness:
                bands.append((s, i))
    if in_band and len(stds) - s >= min_thickness:
        bands.append((s, len(stds)))
    return bands


def _split_region_into_subphotos(crop_bgr, min_side=80):
    """
    Given a cropped region (BGR uint8 numpy array), return a list of
    (x, y, w, h) sub-region rects inside the crop. If no clean split is
    found, returns a single rect covering the whole crop.
    """
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Ignore seams touching the image edges (those are the crop's own borders).
    def midpoints(bands, total):
        pts = []
        for s, e in bands:
            if s <= 0 or e >= total:
                continue
            pts.append((s + e) // 2)
        return pts

    row_stds = gray.std(axis=1)
    h_cuts = [0] + midpoints(_find_low_var_bands(row_stds), h) + [h]
    h_cuts = sorted(set(h_cuts))

    rects = []
    for i in range(len(h_cuts) - 1):
        y1, y2 = h_cuts[i], h_cuts[i + 1]
        if y2 - y1 < min_side:
            continue
        band = gray[y1:y2]
        col_stds = band.std(axis=0)
        v_cuts = [0] + midpoints(_find_low_var_bands(col_stds), w) + [w]
        v_cuts = sorted(set(v_cuts))
        for j in range(len(v_cuts) - 1):
            x1, x2 = v_cuts[j], v_cuts[j + 1]
            if x2 - x1 < min_side:
                continue
            rects.append((x1, y1, x2 - x1, y2 - y1))

    if not rects:
        return [(0, 0, w, h)]
    return rects

# --- Cached model (load once, reuse for every request) --------------------
_model_lock = threading.Lock()
_infer_lock = threading.Lock()   # serialize GPU calls; cheap because no reload
_model = None
_device = None

def get_model():
    global _model, _device
    with _model_lock:
        if _model is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[server] loading classification model on {_device} ...")
            m = SwinForensicModel().to(_device)
            ckpt = torch.load(CKPT_PATH, map_location=_device, weights_only=False)
            use_ema = USE_EMA and "ema" in ckpt
            state = ckpt["ema"] if use_ema else ckpt["model"]
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            m.load_state_dict(state, strict=False)
            m.eval()
            _model = m
            print(f"[server] classification model ready "
                  f"({'EMA' if use_ema else 'RAW'} weights)")
        return _model, _device


# --- Cached segmentation model -------------------------------------------
_seg_lock = threading.Lock()
_seg_infer_lock = threading.Lock()
_seg_model = None
_seg_device = None

def get_seg_model():
    global _seg_model, _seg_device
    with _seg_lock:
        if _seg_model is None:
            _seg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[server] loading segmentation model on {_seg_device} ...")
            _seg_model = seg_load_model(SEG_CKPT_PATH, _seg_device)
            print("[server] segmentation model ready")
        return _seg_model, _seg_device

app = Flask(__name__)
CORS(app)  # allow the extension to call us


DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<b64>.+)$", re.DOTALL)


def _save_data_url(data_url: str) -> str:
    m = DATA_URL_RE.match(data_url)
    if not m:
        raise ValueError("Not a base64 data URL")
    ext = {
        "image/jpeg": ".jpg",
        "image/jpg":  ".jpg",
        "image/png":  ".png",
        "image/webp": ".webp",
        "image/gif":  ".gif",
        "image/bmp":  ".bmp",
    }.get(m.group("mime").lower(), ".png")
    raw = base64.b64decode(m.group("b64"))
    fd, path = tempfile.mkstemp(suffix=ext, prefix="realeyes_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _download_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 RealEyes/0.1"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    mime = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = {
        "image/jpeg": ".jpg",
        "image/jpg":  ".jpg",
        "image/png":  ".png",
        "image/webp": ".webp",
        "image/gif":  ".gif",
        "image/bmp":  ".bmp",
    }.get(mime)
    if ext is None:
        ext = os.path.splitext(url.split("?")[0])[1].lower() or ".png"
    fd, path = tempfile.mkstemp(suffix=ext, prefix="realeyes_")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "status": "online",
        "version": "1.0.0",
        "models": {
            "classification": _model is not None,
            "segmentation": _seg_model is not None,
            "fact_checker": sys.modules.get("checker_en") is not None,
            "llm_translator": sys.modules.get("llm") is not None and hasattr(sys.modules.get("llm"), "client"),
        },
        "gpu_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "uptime_seconds": round(_time.time() - _SERVER_START_TIME, 1)
    })



@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True, silent=True) or {}
        image_url = payload.get("imageUrl")
        force_seg = payload.get("forceSeg", False)
        if not image_url:
            return jsonify({"ok": False, "error": "missing imageUrl"}), 400

        if image_url.startswith("data:"):
            tmp_path = _save_data_url(image_url)
        else:
            tmp_path = _download_url(image_url)

        try:
            model, device = get_model()  # cached: loads once, reused forever

            pil = load_image(tmp_path)

            # Serialize the GPU call. Cheap because the model is already loaded.
            with _infer_lock:
                pred, probs = model_predict(pil, model, device)

            os.makedirs(OUT_DIR, exist_ok=True)
            stem = os.path.splitext(os.path.basename(tmp_path))[0]
            out_path = os.path.join(OUT_DIR, f"{stem}_report.png")
            make_report(pil, probs, int(pred), out_path)

            # --- Run segmentation if classifier isn't confident it's real OR if force_seg is True ---
            seg_path = None
            real_prob = float(probs[0])
            if force_seg or real_prob <= REAL_TRIGGER_THRESHOLD:
                print(f"[server] force_seg={force_seg} or real={real_prob*100:.2f}% <= "
                      f"{REAL_TRIGGER_THRESHOLD*100:.0f}% -> running segmentation")
                seg_model_obj, seg_device = get_seg_model()
                seg_tensor, seg_pil, orig_size = seg_preprocess(pil)
                with _seg_infer_lock:
                    seg_pred = seg_predict_mask(seg_model_obj, seg_tensor, seg_device)
                seg_regions = seg_analyze_regions(seg_pred, orig_size)
                seg_path = os.path.join(OUT_DIR, f"{stem}_segmentation.png")
                seg_save_card(seg_pil, seg_pred, seg_regions, seg_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        print(f"[server] {stem} -> {CLASS_NAMES[int(pred)]} "
              f"real={float(probs[0])*100:.2f}% fake={float(probs[1])*100:.2f}%"
              + (f"  (+ seg)" if seg_path else ""))

        return jsonify({
            "ok": True,
            "label": CLASS_NAMES[int(pred)],
            "pred": int(pred),
            "probs": [float(p) for p in probs],
            "out_path": out_path.replace("\\", "/"),
            "seg_path": seg_path.replace("\\", "/") if seg_path else None,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


def verify_claim_fast(claim: str, top_k: int = 5, progress_cb=None) -> dict:
    """Faster reimplementation of test_live_web4_ar.run.

    Replaces the serial download loop with a 8-way parallel fetch + URL cache,
    and emits progress events. Everything else (search, embed, NLI, aggregate)
    reuses the imported helpers as-is — no edits to the original files.
    """
    base = _news_base
    ar   = _news_ar

    def push(stage: str, pct: int, message: str = ""):
        if progress_cb:
            try:
                progress_cb({"stage": stage, "pct": pct, "message": message})
            except Exception:
                pass

    push("starting", 4, "Preparing claim…")
    cleaned = base._clean_claim(claim)  # noqa: F841 (parity with base.run)

    # Translate claim to both languages so search retrieves EN + AR sources.
    claim_en_raw, claim_ar_raw = ar._bilingual_claim(claim)
    claim_en_clean = base._clean_claim(claim_en_raw)
    claim_ar_clean = base._clean_claim(claim_ar_raw)  # noqa: F841
    query_en = base._build_search_query(claim_en_raw)
    query_ar = base._build_search_query(claim_ar_raw)

    push("searching", 12, "Searching the web (EN + AR)…")
    urls_en = base._search_web(query_en, n=top_k + 5)
    urls_ar = (base._search_web(query_ar, n=top_k + 5)
               if query_ar and query_ar != query_en else [])
    seen, urls = set(), []
    for u in urls_en + urls_ar:
        if u not in seen:
            seen.add(u); urls.append(u)

    if not urls:
        push("done", 100, "No sources found")
        return base._result(claim, "NOT ENOUGH INFO", 0.0, [], "No URLs returned.")

    push("fetching", 25, f"Downloading 0/{len(urls)} pages…")
    def on_fetch(done_n, total_n):
        # Map fetch progress to 25%-65%.
        pct = 25 + int(40 * done_n / max(total_n, 1))
        push("fetching", pct, f"Downloaded {done_n}/{total_n} pages…")
    pages = _parallel_fetch(urls, overall_timeout=20.0, max_workers=8,
                            progress_cb=on_fetch)

    if not pages:
        push("done", 100, "No pages downloaded")
        return base._result(claim, "NOT ENOUGH INFO", 0.0, [],
                            "No content could be retrieved.")

    push("translating", 70, "Translating Arabic sources…")
    ar_idx = [i for i, p in enumerate(pages) if ar._has_arabic(p["text"])]
    if ar_idx:
        snippets   = [pages[i]["text"][:ar._MAX_PAGE_CHARS] for i in ar_idx]
        translated = ar._translate_arabic(snippets)
        for i, t in zip(ar_idx, translated):
            pages[i]["text"] = t

    claim_en = claim_en_clean

    push("selecting", 80, "Selecting the most relevant evidence…")
    evidence = base._select_best_evidence(claim_en, pages, top_k)
    if not evidence:
        push("done", 100, "No relevant evidence")
        return base._result(claim, "NOT ENOUGH INFO", 0.0, [],
                            "No relevant evidence found.")

    texts = [ev["text"] for ev in evidence]
    if any(ar._has_arabic(t) for t in texts):
        texts = ar._translate_arabic(texts)

    push("nli", 90, "Comparing claim to evidence (NLI)…")
    preds   = base._predict_nli(claim_en, texts)
    verdict, conf = base._aggregate(evidence, preds)
    verification  = base._verify_sources(evidence, preds, verdict)

    rich = []
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

    push("done", 100, "Done")
    return base._result(claim, verdict, conf, rich, verification=verification)


def _explain_with_llm(claim: str, verdict: str, evidence: list) -> str | None:
    """Have Groq summarise WHY the verdict holds, using only the gathered sources."""
    if not evidence or verdict not in ("SUPPORTS", "REFUTES"):
        return None
    llm_mod = sys.modules.get("llm")
    client = getattr(llm_mod, "client", None) if llm_mod else None
    if client is None:
        return None

    # Prefer snippets whose NLI label matches the final verdict (they're the
    # ones driving the decision). Cap at 5 to keep the prompt small.
    matching = [e for e in evidence if e.get("nli_label") == verdict]
    use      = (matching or evidence)[:5]

    bullets = []
    for e in use:
        url = e.get("url") or ""
        domain = url.split("/")[2].replace("www.", "") if url.count("/") >= 2 else "source"
        snippet = (e.get("snippet") or "")[:300]
        bullets.append(f"- [{domain}] {snippet}")
    evidence_block = "\n".join(bullets)

    verdict_word = {"REFUTES": "FALSE", "SUPPORTS": "TRUE"}.get(verdict, verdict)

    user_prompt = (
        f"USER CLAIM: \"{claim}\"\n\n"
        f"A fact-checking pipeline judged this claim to be {verdict_word}, "
        f"based on these trusted sources:\n\n{evidence_block}\n\n"
        f"In 2-3 short sentences, explain to the user WHY the claim is "
        f"{verdict_word}. Reference what the sources actually say (e.g. "
        f"\"All gathered sources state that …\"). Do not invent facts — only "
        f"use the evidence above. Plain text only, no markdown, no bullet "
        f"points, no quotes around the whole answer."
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[
                {"role": "system",
                 "content": "You are a concise fact-check explainer."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as _e:
        print(f"[server] explanation failed ({_e})", file=sys.stderr)
        return None


def _has_claim(text: str) -> bool:
    """Cheap heuristic: does this look like an actual claim worth verifying?

    LLM cleaning strips dates, byline metadata and OCR noise — anything left
    empty / too short / made of punctuation is treated as 'no claim'.
    """
    if not text:
        return False
    stripped = re.sub(r"\s+", " ", text).strip()
    if len(stripped) < 15:
        return False
    # Must have at least one run of 3+ Latin or Arabic letters.
    if not re.search(r"[A-Za-z؀-ۿ]{3,}", stripped):
        return False
    return True


_CLAIM_CLASSIFIER_PROMPT = (
    "You decide whether a text contains a CHECKABLE CLAIM — a statement, "
    "allegation or hypothesis whose truth could be researched in news, "
    "encyclopedias, scientific literature, or fact-checking sites.\n\n"
    "Answer YES (it IS a claim) for any factual statement about the world, "
    "people, events, science, history, health, politics, etc. — EVEN IF YOU "
    "ALREADY KNOW IT IS FALSE. The pipeline exists precisely to fact-check "
    "false statements, so misinformation must be passed through.\n\n"
    "Examples that MUST be YES:\n"
    "  - 'NASA landed Perseverance on Mars in 2021.'\n"
    "  - 'The Titanic sank in 1912 after hitting an iceberg.'\n"
    "  - 'Qatar hosted the 2022 FIFA World Cup.'\n"
    "  - 'Vaccines cause autism.'                    (false health claim — YES)\n"
    "  - 'Coronavirus causes AIDS.'                  (false health claim — YES)\n"
    "  - 'The Earth is flat.'                        (false science claim — YES)\n"
    "  - '5G towers spread COVID-19.'                (false conspiracy — YES)\n"
    "  - 'Einstein failed mathematics in school.'    (false biographical — YES)\n"
    "  - News headlines, viral posts, conspiracy theories, social-media rumors.\n\n"
    "Answer NO (no claim) ONLY for text that makes no factual assertion at all:\n"
    "  - Greetings & well-wishes: 'Happy New Year', 'Eid Mubarak', 'Good morning'.\n"
    "  - Personal/private messages: 'I congratulate my brother on his success.'\n"
    "  - Opinions / feelings: 'I love this movie', 'this song is amazing'.\n"
    "  - First-person life updates: 'I went to the gym today'.\n"
    "  - Questions, prayers, ads, UI labels, captions, instructions, jokes.\n"
    "  - Anything purely subjective or about the speaker's private life.\n\n"
    "Default to YES whenever the text states a fact about the external world, "
    "even if short or obviously wrong.\n\n"
    "Respond with EXACTLY one word: YES or NO. No explanation."
)


def _is_factual_claim(text: str) -> bool:
    """Ask the LLM whether `text` is a fact-checkable factual claim.

    Falls back to True (let it through) if the LLM client isn't available, so
    the heuristic stage still gates obvious garbage.
    """
    if not text:
        return False
    llm_mod = sys.modules.get("llm")
    client = getattr(llm_mod, "client", None) if llm_mod else None
    if client is None:
        return True   # no LLM — permissive fallback
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[
                {"role": "system", "content": _CLAIM_CLASSIFIER_PROMPT},
                {"role": "user",   "content": text.strip()},
            ],
        )
        out = (resp.choices[0].message.content or "").strip().upper()
        return out.startswith("YES")
    except Exception as _e:
        print(f"[server] claim detection failed ({_e}); proceeding anyway",
              file=sys.stderr)
        return True


def _trim_news_result(result: dict) -> dict:
    """Reduce verify_claim's dict to JSON-safe primitives + capped snippets."""
    if not isinstance(result, dict):
        return {"ok": False, "error": "invalid result"}
    evidence = []
    for ev in result.get("evidence", []) or []:
        url = ev.get("url") or ""
        domain = url.split("/")[2].replace("www.", "") if url.count("/") >= 2 else ""
        evidence.append({
            "url":             url,
            "title":           ev.get("title"),
            "domain":          domain,
            "retrieval_score": float(ev.get("retrieval_score", 0)),
            "credibility":     float(ev.get("credibility", 1.0)),
            "nli_label":       ev.get("nli_label"),
            "confidence":      float(ev.get("confidence", 0)),
            "probabilities":   {k: float(v) for k, v in (ev.get("probabilities") or {}).items()},
            "snippet":         (ev.get("snippet") or "")[:400],
        })
    return {
        "ok":               True,
        "claim":            result.get("claim"),
        "final_prediction": result.get("final_prediction"),
        "confidence":       float(result.get("confidence", 0)),
        "verification":     result.get("verification") or {},
        "note":             result.get("note") or "",
        "evidence":         evidence,
    }


# --- Async task system for /verify-text -----------------------------------
_tasks_lock = threading.Lock()
_tasks: dict = {}   # task_id -> {progress, result, error, done, started}

def _task_set(tid: str, **kw):
    with _tasks_lock:
        if tid not in _tasks:
            _tasks[tid] = {}
        _tasks[tid].update(kw)

def _task_get(tid: str):
    with _tasks_lock:
        t = _tasks.get(tid)
        return None if t is None else dict(t)

def _task_gc():
    """Drop tasks older than 10 minutes so the dict doesn't grow forever."""
    cutoff = _time.time() - 600
    with _tasks_lock:
        for k in [k for k, v in _tasks.items() if v.get("started", 0) < cutoff]:
            _tasks.pop(k, None)


def _run_gates(claim: str):
    """Apply process_news + 2-stage claim gate. Returns (cleaned, english,
    claim_probe, no_claim_flag). claim_probe is the string to send to the
    pipeline (None when no_claim_flag is True)."""
    cleaned, english = claim, claim
    llm_mod = sys.modules.get("llm")
    if llm_mod and hasattr(llm_mod, "process_news"):
        try:
            cleaned, english = llm_mod.process_news(claim)
        except Exception as _e:
            print(f"[server] process_news failed ({_e})", file=sys.stderr)

    if not (_has_claim(english) or _has_claim(cleaned)):
        return cleaned, english, None, True   # gate 1
    claim_probe = english if _has_claim(english) else cleaned
    if not _is_factual_claim(claim_probe):
        return cleaned, english, None, True   # gate 2
    return cleaned, english, claim_probe, False


@app.route("/verify-text-start", methods=["POST", "OPTIONS"])
def verify_text_start():
    """Kick off async verification. Returns either an immediate no-claim
    answer or a task_id the client should poll via /verify-progress/<id>."""
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(force=True, silent=True) or {}
    claim = (payload.get("text") or "").strip()
    if not claim:
        return jsonify({"ok": False, "error": "empty text"}), 400

    cleaned, english, claim_probe, no_claim = _run_gates(claim)
    if no_claim:
        return jsonify({
            "ok": True, "no_claim": True, "claim": claim,
            "cleaned": cleaned, "english": english, "done": True,
        })

    _task_gc()
    tid = uuid.uuid4().hex
    _task_set(tid,
              started=_time.time(),
              progress={"stage": "queued", "pct": 1, "message": "Queued…"},
              result=None, error=None, done=False)

    def worker(task_id: str, c_probe: str, raw_claim: str):
        try:
            def pcb(d):
                _task_set(task_id, progress=d)
            with _news_lock:
                result = verify_claim_fast(c_probe, progress_cb=pcb)
            trimmed = _trim_news_result(result)
            try:
                explanation = _explain_with_llm(
                    result.get("claim", c_probe),
                    result.get("final_prediction"),
                    result.get("evidence", []) or [],
                )
                if explanation:
                    trimmed["explanation"] = explanation
            except Exception as _e:
                print(f"[server] explanation error: {_e}", file=sys.stderr)
            _task_set(task_id, result=trimmed, done=True,
                      progress={"stage": "done", "pct": 100, "message": "Done"})
        except Exception as _e:
            traceback.print_exc()
            _task_set(task_id, error=str(_e), done=True)

    threading.Thread(target=worker, args=(tid, claim_probe, claim), daemon=True).start()
    return jsonify({"ok": True, "task_id": tid})


@app.route("/verify-progress/<task_id>", methods=["GET"])
def verify_progress(task_id):
    t = _task_get(task_id)
    if not t:
        return jsonify({"ok": False, "error": "unknown task"}), 404
    return jsonify({
        "ok":       True,
        "done":     bool(t.get("done")),
        "progress": t.get("progress") or {},
        "result":   t.get("result"),
        "error":    t.get("error"),
    })


@app.route("/verify-text", methods=["POST", "OPTIONS"])
def verify_text():
    """Run the DeBERTa-based fact-checker on a user-supplied claim.

    Two-stage gate before hitting the news pipeline:
      1. Cheap heuristic: empty / too short / no letters -> no claim.
      2. LLM classifier: not a fact-checkable claim -> no claim.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True, silent=True) or {}
        claim = (payload.get("text") or "").strip()
        if not claim:
            return jsonify({"ok": False, "error": "empty text"}), 400

        # Clean + translate via llm.process_news so Arabic gets evaluated in
        # English (the same way the screenshot flow does).
        cleaned, english = claim, claim
        llm_mod = sys.modules.get("llm")
        if llm_mod and hasattr(llm_mod, "process_news"):
            try:
                cleaned, english = llm_mod.process_news(claim)
            except Exception as _e:
                print(f"[server] process_news failed ({_e})", file=sys.stderr)

        # Gate 1: heuristic.
        if not (_has_claim(english) or _has_claim(cleaned)):
            print("[server] /verify-text: no claim (heuristic), skipping")
            return jsonify({
                "ok":       True,
                "no_claim": True,
                "claim":    claim,
                "cleaned":  cleaned,
                "english":  english,
            })

        # Gate 2: LLM classifier.
        claim_probe = english if _has_claim(english) else cleaned
        if not _is_factual_claim(claim_probe):
            print("[server] /verify-text: not a factual claim, skipping")
            return jsonify({
                "ok":       True,
                "no_claim": True,
                "claim":    claim,
                "cleaned":  cleaned,
                "english":  english,
            })

        # Both gates passed — run the news pipeline.
        with _news_lock:
            result = verify_claim(claim_probe)
        return jsonify(_trim_news_result(result))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/process-screenshot", methods=["POST", "OPTIONS"])
def process_screenshot():
    """Run grad1.run_extraction on a screenshot and return the cropped image."""
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        payload = request.get_json(force=True, silent=True) or {}
        data_url = payload.get("dataUrl")
        if not data_url:
            return jsonify({"ok": False, "error": "missing dataUrl"}), 400

        tmp_path = _save_data_url(data_url)
        try:
            extraction = run_extraction(tmp_path)
        finally:
            pass  # keep temp file until after we've copied the crop

        # Clean up grad1's auto-saved first crop — we'll crop every region ourselves.
        first_crop = extraction.get("cropped_image")
        if first_crop and os.path.isfile(first_crop):
            try:
                os.remove(first_crop)
            except OSError:
                pass

        os.makedirs(OUT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(tmp_path))[0]

        # Load the original screenshot once; crop every detected region.
        src_pil = load_image(tmp_path)
        sw, sh = src_pil.size
        regions = extraction.get("image_regions") or []
        if not regions:
            print("[server] No image regions detected by extraction, falling back to full screenshot as single region.")
            regions = [{
                "id": 1,
                "type": "photo / main image",
                "x": 0,
                "y": 0,
                "width": sw,
                "height": sh,
            }]

        # Expand each grad1 region into its sub-photos (collage splitter).
        expanded = []
        for r in regions:
            x1 = max(0, int(r["x"]))
            y1 = max(0, int(r["y"]))
            x2 = min(sw, x1 + int(r["width"]))
            y2 = min(sh, y1 + int(r["height"]))
            if x2 <= x1 or y2 <= y1:
                continue
            crop_rgb = np.array(src_pil.crop((x1, y1, x2, y2)))
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            subs = _split_region_into_subphotos(crop_bgr)
            for (dx, dy, sw_, sh_) in subs:
                expanded.append({
                    "type":   r.get("type", "image block") + (" (split)" if len(subs) > 1 else ""),
                    "x":      x1 + dx,
                    "y":      y1 + dy,
                    "width":  sw_,
                    "height": sh_,
                })

        per_region = []
        for idx, r in enumerate(expanded, start=1):
            rid = idx
            x1 = r["x"]; y1 = r["y"]
            x2 = x1 + r["width"]; y2 = y1 + r["height"]

            pil_crop = src_pil.crop((x1, y1, x2, y2))

            # Classify this crop.
            model, device = get_model()
            with _infer_lock:
                pred, probs = model_predict(pil_crop, model, device)

            cls_path = os.path.join(OUT_DIR, f"{stem}_r{rid}_report.png")
            make_report(pil_crop, probs, int(pred), cls_path)

            seg_path = None
            real_prob = float(probs[0])
            if real_prob <= REAL_TRIGGER_THRESHOLD:
                print(f"[server] (ss r{rid}) real={real_prob*100:.2f}% -> segmentation")
                seg_model_obj, seg_device = get_seg_model()
                seg_tensor, seg_pil, orig_size = seg_preprocess(pil_crop)
                with _seg_infer_lock:
                    seg_pred = seg_predict_mask(seg_model_obj, seg_tensor, seg_device)
                seg_regions = seg_analyze_regions(seg_pred, orig_size)
                seg_path = os.path.join(OUT_DIR, f"{stem}_r{rid}_segmentation.png")
                seg_save_card(seg_pil, seg_pred, seg_regions, seg_path)

            per_region.append({
                "id":       rid,
                "type":     r.get("type"),
                "label":    CLASS_NAMES[int(pred)],
                "probs":    [float(p) for p in probs],
                "out_path": cls_path.replace("\\", "/"),
                "seg_path": seg_path.replace("\\", "/") if seg_path else None,
            })

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        # --- Text verification --------------------------------------------
        # For each extracted text bucket (outside / inside images), clean it
        # via llm.process_news then run the DeBERTa news-verification pipeline.
        text_outside = (extraction.get("text") or "").strip()
        text_inside  = (extraction.get("text_inside") or "").strip()

        text_results = []
        llm_mod = sys.modules.get("llm")
        for source_name, raw_text in (("outside", text_outside),
                                      ("inside",  text_inside)):
            if len(raw_text) < 10:
                continue

            cleaned, english = raw_text, raw_text
            if llm_mod and hasattr(llm_mod, "process_news"):
                try:
                    cleaned, english = llm_mod.process_news(raw_text)
                except Exception as _e:
                    print(f"[server] process_news failed ({_e})", file=sys.stderr)

            claim_for_check = (english or cleaned or raw_text).strip()

            # Gate 1 (cheap): empty / too short / no letters -> no claim.
            if not (_has_claim(english) or _has_claim(cleaned)):
                print(f"[server] {source_name} text: no claim (heuristic), skipping")
                text_results.append({
                    "source":   source_name,
                    "raw":      raw_text,
                    "cleaned":  cleaned,
                    "english":  english,
                    "no_claim": True,
                    "result":   None,
                })
                continue

            # Gate 2 (LLM): is it actually a fact-checkable factual claim?
            # Catches personal greetings, opinions, ads, etc. that pass gate 1.
            claim_probe = english if _has_claim(english) else cleaned
            if not _is_factual_claim(claim_probe):
                print(f"[server] {source_name} text: not a factual claim, skipping")
                text_results.append({
                    "source":   source_name,
                    "raw":      raw_text,
                    "cleaned":  cleaned,
                    "english":  english,
                    "no_claim": True,
                    "result":   None,
                })
                continue

            print(f"[server] verifying {source_name} text claim "
                  f"({len(claim_for_check)} chars)")
            try:
                with _news_lock:
                    nv = verify_claim_fast(claim_for_check)
                trimmed = _trim_news_result(nv)
                try:
                    explanation = _explain_with_llm(
                        nv.get("claim", claim_for_check),
                        nv.get("final_prediction"),
                        nv.get("evidence", []) or [],
                    )
                    if explanation:
                        trimmed["explanation"] = explanation
                except Exception as _e:
                    print(f"[server] explanation error (ss): {_e}", file=sys.stderr)
            except Exception as _e:
                traceback.print_exc()
                trimmed = {"ok": False, "error": str(_e)}

            text_results.append({
                "source":  source_name,
                "raw":     raw_text,
                "cleaned": cleaned,
                "english": english,
                "result":  trimmed,
            })

        print(f"[server] screenshot processed: {extraction.get('summary', {})} "
              f"regions_classified={len(per_region)} text_claims={len(text_results)}")

        return jsonify({
            "ok": True,
            "regions":       per_region,
            "text_results":  text_results,
            "text":          extraction.get("text"),
            "text_inside":   extraction.get("text_inside"),
            "summary":       extraction.get("summary"),
        })
    except Exception as e:
        traceback.print_exc()

def extract_frames(video_path, min_frames=8, max_frames=60, frames_per_second=1.0):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    duration_s   = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("video has no frames")

    target = int(round(duration_s * frames_per_second))
    n = max(min_frames, min(max_frames, target))
    n = min(n, total_frames)

    indices = [int(round(i * (total_frames - 1) / max(n - 1, 1))) for i in range(n)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append({
            "idx": int(idx),
            "ts":  round(idx / fps, 2) if fps > 0 else None,
            "pil": Image.fromarray(frame_rgb),
            "bgr": frame_bgr,
        })
    cap.release()
    return {"frames": frames, "total_frames": total_frames, "fps": fps, "duration_s": duration_s, "n_sampled": len(frames)}

@app.route("/classify-video", methods=["POST", "OPTIONS"])
def classify_video():
    """Process uploaded video file, sample 12 frames, save thumbnails, and run Swin V2-B classification."""
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        if 'video' not in request.files:
            return jsonify({"ok": False, "error": "No video file provided"}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"ok": False, "error": "Empty filename"}), 400

        # Save video to a temporary file
        ext = os.path.splitext(video_file.filename)[1] or ".mp4"
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="realeyes_video_")
        with os.fdopen(fd, "wb") as f:
            video_file.save(f)

        try:
            extraction = extract_frames(tmp_path)
            frames_data = extraction["frames"]

            os.makedirs(OUT_DIR, exist_ok=True)
            stem = os.path.splitext(os.path.basename(tmp_path))[0]

            probs_list = []
            thumbnails = []

            for i, f_data in enumerate(frames_data):
                pil_frame = f_data["pil"]
                
                model, device = get_model()
                with _infer_lock:
                    pred, probs = model_predict(pil_frame, model, device)

                probs_list.append(probs)

                thumb_path = os.path.join(OUT_DIR, f"{stem}_f{i+1}.png")
                pil_frame.save(thumb_path, format="PNG")
                thumbnails.append(thumb_path.replace("\\", "/"))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if not probs_list:
            return jsonify({"ok": False, "error": "Failed to extract and classify video frames"}), 500

        # Average probabilities across all frames
        avg_probs = np.mean(probs_list, axis=0)
        avg_real_prob = float(avg_probs[0])
        avg_fake_prob = float(avg_probs[1])

        verdict = "REAL" if avg_real_prob > avg_fake_prob else "FAKE"
        confidence = avg_real_prob if verdict == "REAL" else avg_fake_prob

        print(f"[server] video {stem} analyzed -> {verdict} (real={avg_real_prob*100:.2f}%, fake={avg_fake_prob*100:.2f}%)")

        return jsonify({
            "ok": True,
            "verdict": "Verified Video" if verdict == "REAL" else "Deepfake Detected",
            "confidence": confidence,
            "probs": [avg_real_prob, avg_fake_prob],
            "thumbnails": thumbnails
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/report", methods=["GET"])
def report():
    """Serve the generated report PNG so the extension can preview it."""
    path = request.args.get("path", "")
    if not path:
        return ("not found", 404)

    # Resolve OUT_DIR (relative in test_classification.py) against the
    # process's cwd, since that's where run_inference writes the report.
    out_root = Path(OUT_DIR).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_file():
        return ("not found", 404)

    try:
        candidate.relative_to(out_root)
    except ValueError:
        return ("forbidden", 403)

    return send_file(str(candidate), mimetype="image/png")


if __name__ == "__main__":
    # Preload everything so the first request is as fast as the rest.
    get_model()
    get_seg_model()
    try:
        _news_base._get_embedder()
        _news_base._get_reranker()
        _news_base._get_nli()
        print("[server] news-verification models ready")
    except Exception as _e:
        print(f"[server] news warmup skipped: {_e}")
    # threaded=True lets multiple images run in parallel. GPU work is still
    # serialized by the infer locks, but I/O (download, preprocess, encode)
    # and report drawing can overlap freely.
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
# End of server configuration
