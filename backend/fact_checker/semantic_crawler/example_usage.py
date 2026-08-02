"""
example_usage.py
-----------------
End-to-end demonstration of the Semantic Web Crawler & Search Engine.

This script:
1. Crawls Wikipedia pages about machine learning (50 pages)
2. Indexes the content into the vector store
3. Runs several semantic search queries
4. Prints results in the required JSON format
5. Optionally evaluates precision@k

Run:
    python example_usage.py

Prerequisites:
    pip install -r requirements.txt
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# ── Known Wikipedia ML seed URLs for a reproducible demo ─────────────────────
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Large_language_model",
]

SEARCH_QUERIES = [
    "how does the attention mechanism work in transformers",
    "differences between RNN and transformer architectures",
    "applications of generative adversarial networks",
    "reinforcement learning reward function",
    "BERT pre-training objectives",
]


async def run_crawl(query: str, seed_urls: list[str], max_pages: int = 50) -> dict:
    """Run the crawler and indexer, return stats."""
    from queue import Queue
    from crawler.crawler_manager import CrawlerManager
    from indexer.indexer import Indexer

    print(f"\n{'='*60}")
    print(f"CRAWLING: '{query}'")
    print(f"Seed URLs: {len(seed_urls)}  |  Max pages: {max_pages}")
    print("=" * 60)

    content_queue: Queue = Queue(maxsize=200)
    crawler = CrawlerManager(content_queue=content_queue)
    crawler._max_pages = max_pages

    indexer = Indexer(content_queue=content_queue)
    indexer.start()

    pages = await crawler.crawl(query=query, seed_urls=seed_urls)
    indexer.stop(timeout=300.0)

    stats = {**crawler.stats(), **indexer.stats()}
    print("\nCrawl stats:")
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    return stats


def run_searches(queries: list[str]) -> list[dict]:
    """Run all search queries and return their responses."""
    from search.search_engine import SearchEngine

    engine = SearchEngine(auto_crawl=False)
    all_responses = []

    print(f"\n{'='*60}")
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 60)

    for query in queries:
        print(f"\n🔍 Query: {query}")
        response = engine.search(query=query)

        # Required output format
        output = {
            "user_query": response.user_query,
            "evidence": response.evidence,
        }
        all_responses.append(output)

        print(json.dumps(output, indent=2))
        print(f"   ⏱  {response.query_time_ms:.0f}ms  |  {response.total_results} results", end="")
        if response.expanded_terms:
            print(f"  |  expanded: {response.expanded_terms[:3]}")
        else:
            print()

        if response.results:
            print("\n   Top result:")
            r = response.results[0]
            print(f"   [{r.score:.4f}] {r.url}")
            print(f"   {r.snippet[:200]}...")

    return all_responses


def run_evaluation(queries: list[str], all_responses: list[dict]) -> None:
    """Compute and print precision@k against known-relevant URLs."""
    from search.evaluation import evaluate_search, EvalResults
    from search.search_engine import SearchEngine

    # Hand-curated relevance judgements for demo queries
    relevant_urls: dict[str, set[str]] = {
        "how does the attention mechanism work in transformers": {
            "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
            "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        },
        "differences between RNN and transformer architectures": {
            "https://en.wikipedia.org/wiki/Recurrent_neural_network",
            "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        },
        "applications of generative adversarial networks": {
            "https://en.wikipedia.org/wiki/Generative_adversarial_network",
        },
        "reinforcement learning reward function": {
            "https://en.wikipedia.org/wiki/Reinforcement_learning",
        },
        "BERT pre-training objectives": {
            "https://en.wikipedia.org/wiki/BERT_(language_model)",
            "https://en.wikipedia.org/wiki/Large_language_model",
        },
    }

    engine = SearchEngine(auto_crawl=False)
    results = evaluate_search(
        engine=engine,
        queries=[q for q in queries if q in relevant_urls],
        relevant_urls=relevant_urls,
        k=10,
    )
    print(str(results))


def write_sample_output(all_responses: list[dict]) -> None:
    """Write sample output to file."""
    output_path = Path("data/sample_output.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(all_responses, fh, indent=2, ensure_ascii=False)
    print(f"\n✅ Sample output written to {output_path}")


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-crawl", action="store_true", help="Skip crawling (use existing index)")
    parser.add_argument("--pages", type=int, default=30, help="Max pages to crawl")
    parser.add_argument("--eval", action="store_true", help="Run evaluation metrics")
    args = parser.parse_args()

    if not args.skip_crawl:
        await run_crawl(
            query="machine learning deep learning transformers",
            seed_urls=SEED_URLS,
            max_pages=args.pages,
        )
    else:
        print("Skipping crawl — using existing index.")

    all_responses = run_searches(SEARCH_QUERIES)
    write_sample_output(all_responses)

    if args.eval:
        run_evaluation(SEARCH_QUERIES, all_responses)

    print(f"\n✅ Done. Run 'python main.py serve' to start the API server.")


if __name__ == "__main__":
    asyncio.run(main())
