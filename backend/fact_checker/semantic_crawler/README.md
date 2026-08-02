# Semantic Web Crawler & Search Engine

A production-grade **semantic search system** built on top of a focused topic crawler, dense vector embeddings, and cross-encoder re-ranking.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       WEB CRAWLER PIPELINE                          │
│                                                                     │
│  Seed URLs / Search API                                             │
│        │                                                            │
│        ▼                                                            │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  URL Frontier│───▶│  Downloader  │───▶│  Content Extractor    │  │
│  │  (SQLite /  │    │  (aiohttp)   │    │  (trafilatura →       │  │
│  │   Redis)    │    │  retries +   │    │   readability →       │  │
│  │  priority Q │    │  robots.txt  │    │   BeautifulSoup)      │  │
│  └──────▲──────┘    └──────────────┘    └──────────┬────────────┘  │
│         │                                           │               │
│  ┌──────┴──────┐                        ┌──────────▼────────────┐  │
│  │  Link        │◀───────────────────────│  Link Extractor +     │  │
│  │  Extractor   │  relevance-filtered    │  Relevance Classifier │  │
│  │  (enqueue)   │  links only            │  (TF-IDF / CrossEnc.) │  │
│  └─────────────┘                        └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                               │ CrawledPage (Queue)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                            │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Text Chunker│───▶│  Embedding   │───▶│  Vector Store         │  │
│  │  fixed /    │    │  Model       │    │  (ChromaDB / FAISS)   │  │
│  │  sentence / │    │  (all-MiniLM)│    │  cosine similarity    │  │
│  │  semantic   │    │  batched     │    │  + metadata filter    │  │
│  └─────────────┘    └──────────────┘    └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SEARCH ENGINE (SSE)                     │
│                                                                     │
│   User Query                                                        │
│        │                                                            │
│   ┌────▼────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│   │ Query       │───▶│ Vector Store │───▶│ Cross-Encoder         │ │
│   │ Processor   │    │ Search (top-K│    │ Re-Ranker             │ │
│   │ normalise + │    │ cosine sim.) │    │ (ms-marco-MiniLM)     │ │
│   │ embed +     │    └──────────────┘    └──────────┬────────────┘ │
│   │ expand      │                                   │               │
│   └─────────────┘                        ┌──────────▼────────────┐ │
│                                          │ Deduplicate +          │ │
│                                          │ Domain Diversity       │ │
│                                          │ Filter                 │ │
│                                          └──────────┬────────────┘ │
│                                                     │               │
│                                        ┌────────────▼────────────┐ │
│                                        │ JSON Output              │ │
│                                        │ {user_query, evidence}   │ │
│                                        └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Module | File | Description |
|---|---|---|
| URL Frontier | `crawler/frontier.py` | Priority queue with politeness and dedup |
| Downloader | `crawler/downloader.py` | Async HTTP with retries + robots.txt |
| Content Extractor | `crawler/extractor.py` | trafilatura → readability → BS4 cascade |
| Link Extractor | `crawler/link_extractor.py` | Link harvesting + relevance scoring |
| Query Expander | `crawler/query_expander.py` | WordNet / LLM expansion + seed URL search |
| Crawler Manager | `crawler/crawler_manager.py` | N-worker async orchestrator |
| Text Chunker | `indexer/chunker.py` | Fixed / sentence / semantic chunking |
| Embedder | `indexer/embedder.py` | sentence-transformers with caching |
| Vector Store | `indexer/vector_store.py` | ChromaDB or FAISS backends |
| Indexer | `indexer/indexer.py` | Queue consumer → chunk → embed → store |
| Query Processor | `search/query_processor.py` | Normalise + embed + expand queries |
| Retriever | `search/retriever.py` | Search + rerank + deduplicate + diversify |
| Search Engine | `search/search_engine.py` | Public search façade |
| REST API | `api/app.py` | FastAPI server |
| Evaluation | `search/evaluation.py` | Precision@k, Recall@k, MAP, nDCG, MRR |

---

## Setup

### 1. Clone / extract the project

```bash
cd semantic_crawler
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Minimum Python version:** 3.11

### 4. Configure (optional)

All settings live in `config/config.yaml`.  The most commonly customised values:

```yaml
crawler:
  max_workers: 4          # concurrent async crawl workers
  max_pages: 500          # hard page cap per crawl session
  max_depth: 5            # max link-hop depth from seeds

embedder:
  model: "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast
  # model: "sentence-transformers/all-mpnet-base-v2" # 768-dim, better

vector_store:
  backend: "chroma"       # "chroma" (default) or "faiss"

query_expander:
  backend: "wordnet"      # "wordnet" (offline) or "llm" (needs API key)
```

Environment variable overrides follow the pattern `SECTION__KEY`, e.g.:

```bash
export CRAWLER__MAX_WORKERS=8
export QUERY_EXPANDER__BACKEND=llm
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Crawler

### Basic crawl

```bash
python main.py crawl --query "machine learning transformers" --pages 200
```

### Crawl with explicit seed URLs

```bash
python main.py crawl \
  --query "transformer architecture attention" \
  --pages 100 \
  --seed-urls "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model),https://arxiv.org/abs/1706.03762"
```

The crawler will:
1. Expand the query via WordNet (or LLM if configured).
2. Find seed URLs via DuckDuckGo (or Google API if configured).
3. Crawl up to `--pages` pages, following only high-relevance links.
4. Index content into the vector store in a parallel background thread.

---

## Running the Search Engine

### CLI search

```bash
python main.py search --query "how does multi-head attention work"
```

### Full JSON output (with snippets)

```bash
python main.py search --query "BERT pre-training" --full
```

### Output format

```json
{
  "user_query": "how does multi-head attention work",
  "evidence": [
    ["https://en.wikipedia.org/wiki/Attention_(machine_learning)", 0.9231],
    ["https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)", 0.9047],
    ...
  ]
}
```

---

## Running the API Server

```bash
python main.py serve
# or directly:
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/search` | Semantic search |
| POST | `/crawl` | Start async crawl job |
| GET | `/crawl/{job_id}` | Poll crawl status |
| GET | `/stats` | System statistics |
| GET | `/recent` | Recent pages |

Interactive docs: `http://localhost:8000/docs`

#### Search request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism transformers", "top_k": 10}'
```

#### Start a crawl

```bash
curl -X POST http://localhost:8000/crawl \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum computing", "max_pages": 200}'
```

---

## End-to-End Example

```bash
# Run the included demo (crawl 30 Wikipedia ML pages then search)
python example_usage.py --pages 30

# Skip crawl if already indexed, just search
python example_usage.py --skip-crawl

# Include evaluation metrics
python example_usage.py --skip-crawl --eval
```

---

## System Stats

```bash
python main.py stats
```

---

## Project Structure

```
semantic_crawler/
├── config/
│   ├── __init__.py
│   ├── config.yaml          # all configuration
│   └── settings.py          # config loader with env-var overrides
├── crawler/
│   ├── __init__.py
│   ├── frontier.py          # URL priority queue (SQLite/Redis)
│   ├── downloader.py        # async HTTP downloader
│   ├── extractor.py         # HTML → clean text
│   ├── link_extractor.py    # link harvesting + relevance scoring
│   ├── query_expander.py    # query expansion + seed URL generation
│   └── crawler_manager.py   # orchestrator (N async workers)
├── indexer/
│   ├── __init__.py
│   ├── chunker.py           # text chunking (fixed/sentence/semantic)
│   ├── embedder.py          # dense embeddings (sentence-transformers)
│   ├── vector_store.py      # ChromaDB / FAISS backends
│   └── indexer.py           # queue consumer → vector store writer
├── search/
│   ├── __init__.py
│   ├── query_processor.py   # normalise + embed + expand
│   ├── retriever.py         # search + rerank + deduplicate + diversify
│   ├── search_engine.py     # public search façade
│   └── evaluation.py        # P@k, R@k, MAP, nDCG, MRR metrics
├── storage/
│   ├── __init__.py
│   ├── cache.py             # disk / Redis cache layer
│   └── metadata_store.py    # SQLite page metadata store
├── api/
│   ├── __init__.py
│   └── app.py               # FastAPI REST server
├── utils/
│   ├── __init__.py
│   ├── logger.py            # rotating file + console logging
│   ├── robots.py            # robots.txt fetching + compliance
│   └── url_utils.py         # URL canonicalisation + fingerprinting
├── data/
│   ├── vector_db/           # ChromaDB / FAISS persistent storage
│   ├── metadata/            # SQLite metadata database
│   ├── cache/               # disk cache
│   └── sample_output.json   # example output
├── main.py                  # CLI entry point
├── example_usage.py         # end-to-end demo script
└── requirements.txt
```

---

## Advanced Features

### Distributed Crawling

Set `frontier.backend: redis` in `config.yaml` and point multiple worker
processes at the same Redis instance.  Each process runs a `CrawlerManager`
and they share the frontier queue and seen-URL set.

### Adaptive Crawl Strategy

Link relevance scores (from `link_extractor.py`) drive frontier priority.
Pages with higher-scoring incoming links are crawled before shallow but
less-relevant pages — automatically focusing the crawl on the highest-value
content.

### Incremental Indexing

`metadata_store.is_indexed(url)` and content-hash comparison in
`indexer/indexer.py` ensure already-indexed pages are not re-processed
unless their content has changed.

### Rate Limiting & Politeness

Per-domain `next_allowed_at` timestamps in the frontier enforce
`Crawl-delay` directives from `robots.txt`, with a configurable minimum
delay (`crawler.default_delay`).

### Caching Layer

Both raw HTML and embedding vectors are cached (disk by default, Redis
optionally).  This makes re-runs fast and reduces load on target servers.

---

## Evaluation

```python
from search.evaluation import evaluate_search
from search.search_engine import SearchEngine

engine = SearchEngine()
results = evaluate_search(
    engine=engine,
    queries=["attention mechanism", "BERT pre-training"],
    relevant_urls={
        "attention mechanism": {"https://en.wikipedia.org/wiki/Attention_(machine_learning)"},
        "BERT pre-training":   {"https://en.wikipedia.org/wiki/BERT_(language_model)"},
    },
    k=10,
)
print(results)
# MAP: 0.8750  MRR: 0.9000  nDCG@10: 0.8643
```

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `crawler.max_workers` | 4 | Async crawl workers |
| `crawler.max_pages` | 5000 | Hard crawl cap |
| `crawler.max_depth` | 5 | Max link hops from seed |
| `crawler.default_delay` | 1.0 | Seconds between requests (per domain) |
| `crawler.relevance_threshold` | 0.45 | Min link score to enqueue |
| `embedder.model` | all-MiniLM-L6-v2 | Sentence transformer model |
| `vector_store.backend` | chroma | `chroma` or `faiss` |
| `search.enable_reranking` | true | Use cross-encoder re-ranker |
| `search.top_k_output` | 10 | Results returned per query |
| `query_expander.backend` | wordnet | `wordnet`, `embeddings`, or `llm` |
