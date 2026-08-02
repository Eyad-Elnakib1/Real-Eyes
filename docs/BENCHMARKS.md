# RealEyes — Benchmarks & Evaluation

> **Last Updated:** July 2026  
> **Hardware:** NVIDIA GPU (CUDA), Intel/AMD CPU, 16 GB RAM

---

## 1. Image Classification (Swin-V2-B)

### 1.1 Test Set Performance

| Metric | Score |
|---|---|
| **Accuracy** | 95.2% |
| **Precision (Fake)** | 94.8% |
| **Recall (Fake)** | 95.6% |
| **F1 Score** | 95.2% |
| **AUC-ROC** | 0.987 |

> *Evaluated on a held-out test set of real vs. AI-generated images. Model uses EMA weights from best validation checkpoint (`bestv_3.3.pth`).*

### 1.2 Training Configuration

| Parameter | Value |
|---|---|
| Architecture | `swinv2_base_window12to16_192to256` (timm) |
| Input Resolution | 256 × 256 |
| Optimizer | AdamW |
| Weight Strategy | Exponential Moving Average (EMA) |
| Checkpoint | Epoch with best validation loss |

> *Full training log available at [`docs/training_log.txt`](training_log.txt).*

---

## 2. Anomaly Segmentation (AXUNet)

### 2.1 Segmentation Performance

| Metric | Score |
|---|---|
| **Mean IoU** | 0.72 |
| **Dice Coefficient** | 0.78 |
| **Pixel Accuracy** | 96.1% |

> *Evaluated on manipulated image segmentation test set. Model checkpoint: epoch 21.*

### 2.2 Architecture Details

| Parameter | Value |
|---|---|
| Backbone | Xception (timm: `legacy_xception`) |
| Decoder | UNet with attention gates |
| Weights | `checkpoint_epoch_21.pth` (1.5 GB) |
| Output | Binary mask (manipulated vs. authentic pixels) |

---

## 3. Text Fact-Checking (DeBERTa NLI Pipeline)

### 3.1 Built-in Test Battery Results

The `checker_en.py` file includes a 7-claim test battery of well-documented factual statements:

| # | Claim | Expected | Pipeline Output | Match |
|---|---|---|---|---|
| 1 | "The RMS Titanic sank on April 15, 1912, after hitting an iceberg." | SUPPORTS | SUPPORTS | ✅ |
| 2 | "The Titanic was built and launched in Southampton, England." | REFUTES | REFUTES | ✅ |
| 3 | "Over 1,500 people died in the Titanic disaster." | SUPPORTS | SUPPORTS | ✅ |
| 4 | "The Titanic had enough lifeboats for all passengers and crew." | REFUTES | REFUTES | ✅ |
| 5 | "Albert Einstein failed mathematics in school." | REFUTES | REFUTES | ✅ |
| 6 | "Vaccines cause autism." | REFUTES | REFUTES | ✅ |
| 7 | "The Great Wall of China is visible from space." | REFUTES | REFUTES | ✅ |

**Test Battery Accuracy: 7/7 (100%)**

### 3.2 Social Media Misinformation Battery

Additional viral claim tests (from `SOCIAL_TESTS` in `checker_en.py`) covering anti-vax conspiracies, flat-earth claims, and government misinformation posts with hashtags and emoji — designed to stress-test the NLP preprocessing and trust-weighting.

### 3.3 Pipeline Component Benchmarks

| Component | Model | Latency (avg) | Purpose |
|---|---|---|---|
| Claim Classification | Groq Llama-3.3-70b | ~0.5s | Determines if text is a checkable claim |
| Web Search | DuckDuckGo (ddgs) | 2-5s | Retrieves evidence URLs |
| Page Download | httpx (8-way parallel) | 3-8s | Fetches page content |
| Embedding | all-MiniLM-L6-v2 | < 50ms | Embeds claim + evidence chunks |
| Reranking | ms-marco-MiniLM-L-6-v2 | < 100ms | Cross-encoder score fusion |
| NLI Inference | DeBERTa 3-class | < 200ms | Per-chunk verdict generation |
| LLM Explanation | Groq Llama-3.3-70b | ~1s | Human-readable result summary |

**End-to-End Latency (text claim):** 8-20 seconds (dominated by web search + download)

---

## 4. End-to-End Latency Summary

| Pipeline | Input | Avg Latency | GPU Required |
|---|---|---|---|
| Image Classification | Single image | ~1.5s | Yes |
| Image + Segmentation | Single image | ~3.5s | Yes |
| Text Fact-Checking | Text claim | 8-20s | Yes (NLI) |
| Screenshot Analysis | Post screenshot | 15-40s | Yes |
| Video Analysis | 12 frames | ~18s | Yes |

> *Latencies measured on consumer GPU (NVIDIA RTX-class). CPU fallback adds approximately 3-5× to inference times.*

---

## 5. Trust Scoring Validation

The source credibility system (`semantic_crawler/utils/trust.py`) was validated by checking that known misinformation domains score low and known reliable domains score high:

| Domain | Expected Tier | `trust_score()` | Status |
|---|---|---|---|
| reuters.com | Trusted | 3.0 | ✅ |
| snopes.com | Fact-Checker | 3.0 | ✅ |
| bbc.com | Trusted | 2.0 | ✅ |
| nytimes.com | Trusted | 2.0 | ✅ |
| infowars.com | Blocked | 0.0 | ✅ |
| naturalnews.com | Blocked | 0.0 | ✅ |
| reddit.com | Social | 0.2 | ✅ |
| unknownblog.xyz | Neutral | 1.0 | ✅ |
