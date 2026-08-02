# RealEyes — REST API Reference

> **Base URL:** `http://127.0.0.1:5001`  
> **Content-Type:** `application/json` (unless otherwise noted)  
> **CORS:** Enabled for all origins

---

## Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/predict` | Classify an image as Real or AI-Generated |
| `POST` | `/verify-text-start` | Start async text fact-checking |
| `GET` | `/verify-progress/:id` | Poll async verification progress |
| `POST` | `/verify-text` | Synchronous text fact-checking (blocking) |
| `POST` | `/process-screenshot` | Analyze a social media screenshot |
| `POST` | `/classify-video` | Analyze a video for deepfakes |
| `GET` | `/report` | Serve a generated report image |

---

## `GET /health`

Check if the server is running and models are loaded.

### Response

```json
{ "ok": true }
```

---

## `POST /predict`

Classify a single image as Real or AI-Generated. Optionally generates a segmentation heatmap.

### Request Body

```json
{
  "imageUrl": "data:image/png;base64,iVBOR...",
  "forceSeg": false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `imageUrl` | `string` | ✅ | Base64 data URL (`data:image/...;base64,...`) or remote HTTP URL |
| `forceSeg` | `boolean` | ❌ | Force segmentation regardless of classification confidence. Default: `false` |

### Response (200)

```json
{
  "ok": true,
  "label": "FAKE",
  "pred": 1,
  "probs": [0.08, 0.92],
  "out_path": "results/realeyes_abc123_report.png",
  "seg_path": "results/realeyes_abc123_segmentation.png"
}
```

| Field | Type | Description |
|---|---|---|
| `label` | `string` | `"Real Image"` or `"FAKE"` |
| `pred` | `int` | `0` = Real, `1` = Fake |
| `probs` | `[float, float]` | `[real_probability, fake_probability]` |
| `out_path` | `string` | Path to classification report card PNG |
| `seg_path` | `string \| null` | Path to segmentation heatmap (null if skipped) |

### Segmentation Trigger Logic

Segmentation runs automatically when:
- `forceSeg` is `true`, **OR**
- `probs[0]` (real probability) ≤ 0.85

---

## `POST /verify-text-start`

Start an asynchronous text fact-checking task. Returns immediately with a `task_id` for polling.

### Request Body

```json
{
  "text": "Vaccines cause autism."
}
```

### Response — Claim Detected (200)

```json
{
  "ok": true,
  "task_id": "a1b2c3d4e5f6..."
}
```

### Response — No Claim Detected (200)

```json
{
  "ok": true,
  "no_claim": true,
  "claim": "Happy New Year!",
  "cleaned": "Happy New Year!",
  "english": "Happy New Year!",
  "done": true
}
```

The two-stage claim gate filters out:
- **Gate 1 (heuristic):** Empty, too short, or non-textual input.
- **Gate 2 (LLM classifier):** Greetings, opinions, ads, personal messages.

---

## `GET /verify-progress/:task_id`

Poll the progress of an async verification task.

### Response — In Progress (200)

```json
{
  "ok": true,
  "done": false,
  "progress": {
    "stage": "fetching",
    "pct": 45,
    "message": "Downloaded 4/10 pages…"
  },
  "result": null,
  "error": null
}
```

### Progress Stages

| Stage | % Range | Description |
|---|---|---|
| `queued` | 1% | Task created, waiting for worker thread |
| `starting` | 4% | Preparing claim |
| `searching` | 12% | Web search (EN + AR) |
| `fetching` | 25-65% | Parallel page download |
| `translating` | 70% | Arabic source translation |
| `selecting` | 80% | Evidence embedding & reranking |
| `nli` | 90% | DeBERTa NLI inference |
| `done` | 100% | Complete |

### Response — Complete (200)

```json
{
  "ok": true,
  "done": true,
  "progress": { "stage": "done", "pct": 100, "message": "Done" },
  "result": {
    "ok": true,
    "claim": "Vaccines cause autism.",
    "final_prediction": "REFUTES",
    "confidence": 0.89,
    "explanation": "All gathered sources state that extensive scientific research...",
    "verification": {
      "verified": true,
      "independent_sources": 4,
      "trusted_agree": ["reuters.com", "snopes.com"],
      "trusted_contradict": [],
      "statement": "REFUTES — 4 independent sources agree (2 trusted)."
    },
    "evidence": [
      {
        "url": "https://www.reuters.com/...",
        "title": "Vaccines do not cause autism",
        "domain": "reuters.com",
        "retrieval_score": 0.87,
        "credibility": 3.0,
        "nli_label": "REFUTES",
        "confidence": 0.94,
        "probabilities": {
          "SUPPORTS": 0.02,
          "REFUTES": 0.94,
          "NOT ENOUGH INFO": 0.04
        },
        "snippet": "Multiple large-scale studies involving millions of children..."
      }
    ]
  },
  "error": null
}
```

---

## `POST /verify-text`

Synchronous (blocking) text fact-checking. Same logic as the async variant, but blocks until complete. Suitable for programmatic/API use where polling is inconvenient.

### Request/Response

Same schema as `/verify-text-start` results, returned directly.

---

## `POST /process-screenshot`

Analyze a social media post screenshot. Extracts text (EN + AR), detects image regions, classifies each region, and fact-checks each text block.

### Request Body

```json
{
  "dataUrl": "data:image/png;base64,iVBOR..."
}
```

### Response (200)

```json
{
  "ok": true,
  "regions": [
    {
      "id": 1,
      "type": "photo / main image",
      "label": "FAKE",
      "probs": [0.12, 0.88],
      "out_path": "results/screenshot_r1_report.png",
      "seg_path": "results/screenshot_r1_segmentation.png"
    }
  ],
  "text_results": [
    {
      "source": "outside",
      "raw": "Breaking: New study confirms...",
      "cleaned": "New study confirms...",
      "english": "New study confirms...",
      "result": {
        "ok": true,
        "final_prediction": "REFUTES",
        "confidence": 0.85,
        "explanation": "...",
        "evidence": [...]
      }
    }
  ],
  "text": "Breaking: New study confirms...",
  "text_inside": "",
  "summary": {
    "total_text_blocks": 5,
    "total_outside_text": 4,
    "total_inside_text": 1,
    "total_image_regions": 1
  }
}
```

---

## `POST /classify-video`

Analyze a video for deepfakes by sampling and classifying individual frames.

### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `video` | `file` | ✅ | Video file (MP4, WEBM, MOV). Max 12 MB. |

### Example (curl)

```bash
curl -X POST http://127.0.0.1:5001/classify-video \
  -F "video=@suspect_video.mp4"
```

### Response (200)

```json
{
  "ok": true,
  "verdict": "Deepfake Detected",
  "confidence": 0.87,
  "probs": [0.13, 0.87],
  "thumbnails": [
    "results/video_f1.png",
    "results/video_f2.png"
  ]
}
```

---

## `GET /report`

Serve a generated report image (classification card or segmentation heatmap).

### Query Parameters

| Param | Type | Required | Description |
|---|---|---|---|
| `path` | `string` | ✅ | Path to the report PNG (as returned by `/predict` or `/process-screenshot`) |

### Security

Path traversal is blocked: the resolved path must be a child of `OUT_DIR`. Returns `403 Forbidden` otherwise.

### Response

- **200:** PNG image file
- **403:** Path outside allowed directory
- **404:** File not found

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "ok": false,
  "error": "descriptive error message"
}
```

| HTTP Code | Meaning |
|---|---|
| `400` | Bad request (missing field, empty input) |
| `404` | Resource not found (unknown task ID, missing file) |
| `403` | Forbidden (path traversal attempt) |
| `500` | Internal server error (model crash, exception) |
