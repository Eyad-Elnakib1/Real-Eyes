# 🔍 FactCheck AI

> A Chrome Extension + FastAPI backend for **fake news detection** and **AI image manipulation detection** — powered by Hugging Face Transformers and Tesseract OCR.

---

## 📐 Architecture

```
┌───────────────────────────────────────────────────────┐
│                  Chrome Extension                     │
│                                                       │
│  ┌───────────────┐    ┌────────────────────────────┐  │
│  │ Context Menu  │    │       Sidebar Panel        │  │
│  │ (right-click) │──> |  Verdict · Confidence ·    │  │
│  └──────┬────────┘    │  Heatmap · Links · OCR Text│  │
│         │             └───────────────▲────────────┘  │
│  ┌──────▼────────┐                    │               │
│  │ background.js │────────────────────┘               │
│  │ service worker│                                    │
│  └──────┬────────┘                                    │
└─────────┼─────────────────────────────────────────────┘
          │  HTTP (multipart / JSON)
          ▼
┌──────────────────────────────────────────────────────┐
│                 FastAPI Backend                      │
│                                                      │
│  POST /analyze/text        ──> fake_news_service     │
│  POST /analyze/image       ──> ai_image_detection    │
│  POST /analyze/screenshot  ──> screenshot_processing │
│       │                         │                    │
│       │                    ┌────┴──────────────┐     │
│       │                    │   OCR (Tesseract) │     │
│       │                    │ Region extraction │     │
│       │                    └────┬──────────────┘     │
│       ▼                         ▼                    │
│  RoBERTa (fake news)    SDXL Detector (AI image)     │
│  GDELT link retrieval   Heatmap generation (OpenCV)  │
└──────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
factcheck/
│
├── backend/
│   ├── app.py                          # FastAPI entry point
│   ├── requirements.txt
│   │
│   ├── routers/
│   │   ├── text_router.py              # POST /analyze/text
│   │   ├── image_router.py             # POST /analyze/image
│   │   └── screenshot_router.py        # POST /analyze/screenshot
│   │
│   ├── services/
│   │   ├── fake_news_service.py        # RoBERTa pipeline + GDELT links
│   │   ├── ai_image_detection_service.py # SDXL detector + heatmap
│   │   └── screenshot_processing.py    # OCR → fake-news + image analysis
│   │
│   ├── models/
│   │   └── model_loader.py             # Cached HF pipelines
│   │
│   └── utils/
│       ├── ocr_utils.py                # Tesseract wrapper
│       └── image_utils.py              # Region extraction
│
└── extension/
    ├── manifest.json                   # Manifest V3
    ├── background.js                   # Service worker
    ├── content.js                      # Page-injected helper
    ├── contextMenu.js                  # Right-click menu setup
    ├── sidebar.html                    # Side panel UI
    ├── sidebar.js                      # UI logic + message handling
    └── sidebar.css                     # Dark-theme styles
 

---

## ⚙️ Backend Setup

### 1. Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| pip | latest |
| Tesseract OCR | 5.x |

### 2. Install Tesseract OCR

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

**macOS (Homebrew)**
```bash
brew install tesseract
```

**Windows**
Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add the install directory to your `PATH`.

Verify installation:
```bash
tesseract --version
```

### 3. Create a virtual environment

```bash
cd factcheck/backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration** — If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build instead:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🤖 Model Setup

Both models are downloaded automatically from Hugging Face on first run.

| Purpose | Model ID |
|---------|----------|
| Fake news detection | `hamzab/roberta-fake-news-classification` |
| AI image detection | `Organika/sdxl-detector` |

First startup may take a few minutes while models are cached locally (`~/.cache/huggingface/`).

---

## 🚀 Running the Backend

```bash
cd factcheck/backend
uvicorn app:app --host localhost --port 8000 --reload
```

The API will be available at **http://localhost:8000**.

Interactive docs: **http://localhost:8000/docs**

---

## 🧩 Installing the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `factcheck/extension/` folder
5. The FactCheck AI icon will appear in your toolbar

> **Note:** Chrome requires the extension's `host_permissions` to include `http://localhost:8000/*`.  
> This is already configured in `manifest.json`.

---

## 🖱️ Usage

### Fact-check text
1. Select any text on a webpage
2. Right-click → **🔍 Fact Check Selected Text**
3. View the verdict and supporting links in the sidebar

### Analyse an image
1. Right-click any image on a page
2. Select **🖼️ Analyse Image (AI Detection)**
3. The sidebar shows the verdict and a heatmap overlay

### Screenshot analysis
1. Click the FactCheck AI toolbar icon (or right-click → **📸 Analyse Screenshot**)
2. The entire visible page is captured and analysed
3. The sidebar shows OCR text, fake-news result, and image detection result

---

## 📡 API Reference

### `POST /analyze/text`

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover that the moon is made of cheese."}'
```

**Response**
```json
{
  "verdict": "fake",
  "confidence": 0.9823,
  "raw_label": "FAKE",
  "supporting_links": [
    "https://example-news.com/moon-composition",
    "https://science-daily.com/lunar-facts"
  ]
}
```

---

### `POST /analyze/image`

```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "file=@/path/to/image.jpg"
```

**Response**
```json
{
  "verdict": "AI modified",
  "confidence": 0.8741,
  "raw_label": "artificial",
  "heatmap": "<base64 PNG string>"
}
```

---

### `POST /analyze/screenshot`

```bash
curl -X POST http://localhost:8000/analyze/screenshot \
  -F "file=@/path/to/screenshot.png"
```

**Response**
```json
{
  "extracted_text": "Breaking News: Scientists claim...",
  "fake_news_result": {
    "verdict": "fake",
    "confidence": 0.912,
    "raw_label": "FAKE",
    "supporting_links": []
  },
  "image_analysis": {
    "verdict": "Authentic",
    "confidence": 0.763,
    "raw_label": "real",
    "heatmap": "<base64 PNG string>"
  }
}
```

---

## 🖼️ UI Screenshots

```
┌─────────────────────────────┐
│  🔍 FactCheck AI  [📸 Screenshot] │
├─────────────────────────────┤
│  📝 Fake News Detection     │
│  ┌────────────────────┐     │
│  │   ⛔ FAKE NEWS     │     │
│  └────────────────────┘     │
│  Confidence  ████████░░ 82% │
│                             │
│  Supporting Sources         │
│  • https://reuters.com/...  │
│  • https://apnews.com/...   │
└─────────────────────────────┘

┌─────────────────────────────┐
│  🖼️ AI Image Detection      │
│  ┌────────────────────┐     │
│  │ ⚠️ AI MODIFIED     │     │
│  └────────────────────┘     │
│  Confidence  ██████░░░░ 64% │
│  Heatmap Overlay            │
│  [██████████████████████]   │
└─────────────────────────────┘
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `pytesseract.TesseractNotFoundError` | Install Tesseract and ensure it's on your `PATH` |
| CORS errors in extension | Ensure backend is running on port 8000 |
| Models download slowly | Pre-download with `transformers-cli download <model_id>` |
| Extension not loading | Check Chrome developer console for manifest errors |
| Empty OCR results | Try pre-processing: increase image resolution before sending |

---

## 📄 License

MIT — see `LICENSE` for details.
