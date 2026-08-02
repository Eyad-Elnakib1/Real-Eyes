# ADR-003: Arabic→English Translation Before NLI

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-07 |
| **Decision** | Translate Arabic claims and evidence to English before running the NLI model, rather than training a bilingual NLI model |
| **Deciders** | RealEyes Engineering Team |

---

## Context

RealEyes must handle both English and Arabic text. The fact-checking pipeline relies on a fine-tuned DeBERTa NLI model that was trained exclusively on English-language premise-hypothesis pairs. Arabic content enters the system from two sources:

1. **Direct text input:** A user pastes an Arabic news claim.
2. **Screenshot OCR:** EasyOCR extracts Arabic text from social media screenshots.

We need a strategy to handle Arabic without degrading the NLI model's accuracy.

## Decision

We use a **translate-then-verify** approach:

1. **`llm.process_news(text)`** (Groq Llama-3.3-70b) cleans the text, fixes OCR errors, and translates Arabic→English.
2. **`checker_ar._bilingual_claim(claim)`** generates both Arabic and English versions for dual-language web search.
3. **`checker_ar._translate_arabic(texts)`** translates Arabic evidence pages to English before embedding and NLI.
4. The English-only DeBERTa model then runs on all-English inputs.

## Rationale

### 1. No Retraining Required

Training a bilingual NLI model requires:
- A large Arabic NLI dataset (scarce — XNLI has only 5K Arabic pairs vs. 392K English SNLI).
- GPU compute for fine-tuning DeBERTa on multilingual data.
- Validation that Arabic performance matches English (risk of regression).

Translation via Groq LLM is zero-cost in terms of training, and Llama-3.3-70b has strong Arabic→English translation quality.

### 2. Evidence Quality

Arabic web search returns Arabic pages. These pages must be compared against the claim using cosine similarity (MiniLM-L6 embedder) and NLI. Both models are English-only. Translating the pages to English ensures the full retrieval-ranking-NLI pipeline operates in its trained language.

### 3. Dual-Language Search Coverage

By generating both Arabic and English search queries, we retrieve evidence from both Arabic news outlets (Al Jazeera, BBC Arabic) and English fact-checkers (Snopes, Reuters). This significantly increases evidence coverage for claims that originate in Arabic media.

### 4. Latency Tradeoff

The Groq API call adds ~0.5-1s per translation batch. This is acceptable because:
- The web search + download phase already takes 5-15s.
- Translation is batched (all Arabic pages translated in one API call).
- The async task pattern (`/verify-text-start`) means the user sees progress updates, not a blocked UI.

## Consequences

### Positive
- Zero ML training cost.
- Leverages state-of-the-art LLM translation quality.
- Same NLI model serves both languages with consistent accuracy.
- Dual-language search improves evidence recall.

### Negative
- Depends on Groq API availability (graceful fallback: `llm.py` stub passes text through untranslated).
- Translation errors can propagate to NLI (mitigated by Llama-3.3's high Arabic quality).
- Groq API has rate limits on the free tier (mitigated by batching).

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| **Fine-tune multilingual DeBERTa (mDeBERTa)** | Requires Arabic NLI training data and compute; risk of English regression |
| **Use XLM-R for NLI** | Lower accuracy than English-only DeBERTa on English pairs; compromises the majority use case |
| **Google Translate API** | Paid service; Groq free tier is sufficient |
| **No Arabic support** | Unacceptable — significant portion of target users are Arabic speakers |
