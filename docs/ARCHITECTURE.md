# RealEyes — System Architecture

> **Last Updated:** July 2026  
> **Status:** Production (Local Deployment)

---

## 1. Overview

RealEyes is a multi-modal misinformation detection platform that analyzes text, images, social media posts, and videos for AI generation or factual inaccuracy. It operates as a local-first system with three client surfaces (web dashboard, Chrome extension, direct API) communicating with a single Python Flask backend that orchestrates four independent ML pipelines.

---

## 2. High-Level Architecture

The system follows a **monolithic API gateway** pattern where a single Flask process (`server.py`) serves all endpoints and manages all ML model lifecycles. This is a deliberate architectural decision documented in [ADR-001](ADR-001-monolith-flask-server.md).

### 2.1 Layers

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  Next.js Dashboard  │  Chrome Extension  │  Direct API  │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP REST (JSON)
┌──────────────────────▼──────────────────────────────────┐
│                    API GATEWAY LAYER                     │
│              Flask + CORS (server.py :5001)              │
│                                                          │
│  Routes: /predict, /verify-text-start, /verify-progress, │
│          /process-screenshot, /classify-video, /report   │
└──────────┬──────────┬───────────┬───────────────────────┘
           │          │           │
┌──────────▼──┐ ┌─────▼─────┐ ┌──▼──────────────────────┐
│ VISION      │ │ TEXT/NLI  │ │ SCREENSHOT              │
│ PIPELINE    │ │ PIPELINE  │ │ PIPELINE                │
│             │ │           │ │                          │
│ Swin-V2-B  │ │ Groq LLM  │ │ EasyOCR → CV detection  │
│ AXUNet     │ │ DuckDuckGo│ │ → Vision + Text pipeline │
│             │ │ MiniLM-L6 │ │                          │
│             │ │ DeBERTa   │ │                          │
└─────────────┘ └───────────┘ └──────────────────────────┘
```

### 2.2 Concurrency Model

All ML models share a single CUDA device. To prevent GPU OOM errors under concurrent HTTP requests, the server uses **fine-grained thread locks**:

| Lock | Scope | Reason |
|---|---|---|
| `_model_lock` | Classification model initialization | Prevents double-loading on first concurrent requests |
| `_infer_lock` | Classification GPU forward pass | Serializes CUDA calls |
| `_seg_lock` | Segmentation model initialization | Same pattern as above |
| `_seg_infer_lock` | Segmentation GPU forward pass | Serializes CUDA calls |
| `_news_lock` | Entire fact-checking pipeline | NLI model is not thread-safe |
| `_url_cache_lock` | URL fetch cache reads/writes | Bounded LRU cache (200 entries, 1h TTL) |

Flask runs with `threaded=True`, so HTTP I/O, image preprocessing, and report card generation can overlap freely — only the GPU forward passes are serialized.

### 2.3 Model Loading Strategy

All models use a **lazy singleton** pattern: loaded once on first access, then cached in module-level globals for the lifetime of the process. The `if __name__ == "__main__"` block pre-loads everything at startup so the first user request is as fast as subsequent ones.

---

## 3. Pipeline Details

### 3.1 Image Classification + Segmentation

```
Input (base64 / URL)
  ↓ decode & save temp file
  ↓ PIL load_image()
  ↓ Swin-V2-B forward pass → [real_prob, fake_prob]
  ↓ make_report() → PNG report card
  ↓ if real_prob ≤ 85% OR forceSeg:
  │   ↓ AXUNet forward pass → pixel mask
  │   ↓ analyze_regions() → area stats
  │   ↓ save_heatmap_card() → PNG overlay
  ↓ JSON response {label, probs, out_path, seg_path}
```

**Threshold logic:** If the classifier is more than 85% confident the image is real, segmentation is skipped (it would just highlight noise). This saves ~2s of GPU time per real image.

### 3.2 Text Fact-Checking

```
Input (raw claim text)
  ↓ Gate 1: heuristic (length, character check)
  ↓ Gate 2: LLM claim classifier (Groq Llama-3.3-70b)
  ↓ Bilingual cleaning (llm.process_news: OCR fix + Arabic→EN)
  ↓ Search query building (hashtags + spaCy NER + YAKE keywords)
  ↓ Web search:
  │   ↓ Trusted-first: WP:RSP site:-restricted DuckDuckGo queries
  │   ↓ Open-web: standard DuckDuckGo query
  │   ↓ Wikipedia: OpenSearch API
  ↓ Parallel fetch (8 workers, 20s timeout, URL cache)
  ↓ Arabic source translation (if detected)
  ↓ Evidence selection:
  │   ↓ Sentence chunking (120 tokens, 1-sentence overlap)
  │   ↓ MiniLM-L6 cosine similarity
  │   ↓ ms-marco cross-encoder reranking (0.6 bi-encoder + 0.4 cross-encoder)
  │   ↓ Domain dedup (max 2 chunks/domain)
  ↓ 3-class DeBERTa NLI → {SUPPORTS, REFUTES, NEI} per chunk
  ↓ Credibility-weighted aggregation → final verdict + confidence
  ↓ Multi-source verification → independent source count + trusted agreement
  ↓ LLM explanation (Groq summarizes WHY the verdict holds)
  ↓ JSON response
```

### 3.3 Screenshot Analysis

Combines both pipelines above:

1. **EasyOCR** extracts text blocks with bounding boxes (English + Arabic).
2. **CV region detection:** Gaussian variance map → morphological close/open → contour detection → UI bar exclusion → bounding box expansion.
3. **Collage splitter:** Post-processes tightly packed photos (e.g., 2×2 wedding collages) by detecting low-variance seam lines.
4. **Per-region classification:** Each detected sub-image runs through the full Swin-V2-B + AXUNet pipeline.
5. **Per-text verification:** Text blocks are classified as "outside" (captions) or "inside" (embedded in images), then each runs through the fact-checking pipeline.

### 3.4 Video Analysis

1. Save uploaded video to temp file.
2. Extract frames using OpenCV at ~1fps (adaptive: min 8, max 60 frames).
3. Each frame is independently classified by Swin-V2-B.
4. Probabilities are averaged across all frames.
5. Final verdict is based on majority probability.

---

## 4. Data Flow: Async Text Verification

The text verification endpoint uses a **task-based async pattern** to avoid HTTP timeout on long-running fact-checks:

```
POST /verify-text-start
  → immediate response: {task_id}
  → spawns daemon thread: worker(task_id, claim)

GET /verify-progress/{task_id}  (client polls every 2s)
  → returns: {done, progress: {stage, pct, message}, result, error}
```

Progress stages: `queued → starting → searching → fetching → translating → selecting → nli → done`

Tasks are garbage-collected after 10 minutes.

---

## 5. Security Considerations

- **API keys** are stored in environment variables, never hardcoded.
- **Report file serving** (`/report`) enforces path traversal protection: resolved paths must be children of `OUT_DIR`.
- **Domain blocklist:** 80+ known misinformation domains are silently dropped from search results (InfoWars, Natural News, state propaganda outlets, etc.).
- **WP:RSP blocklist:** Additional Wikipedia-community-maintained unreliable source list.
- **CORS** is enabled for extension communication but runs on localhost only.

---

## 6. Technology Inventory

| Component | Technology | Version |
|---|---|---|
| API Server | Flask + flask-cors | 3.0+ |
| Image Classifier | Swin Transformer V2-B (timm) | — |
| Anomaly Segmenter | AXUNet (Xception + UNet + Attention) | — |
| NLI Model | DeBERTa (3-class, fine-tuned) | — |
| Embedder | sentence-transformers/all-MiniLM-L6-v2 | — |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | — |
| LLM | Groq Cloud (Llama-3.3-70b-versatile) | — |
| OCR | EasyOCR (EN + AR) | 1.7+ |
| NER | spaCy (en_core_web_sm) | 3.7+ |
| Keywords | YAKE | 0.4.8+ |
| Web Search | DuckDuckGo (ddgs) | 6.0+ |
| Content Extract | Trafilatura | 1.10+ |
| Frontend | Next.js 16, React 19, Tailwind 4 | — |
| Animation | Framer Motion | 12+ |
| Extension | Chrome Manifest V3 | — |
