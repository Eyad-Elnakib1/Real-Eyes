"""
main.py
--------
Command-line entry point for the Semantic Web Crawler & Search Engine.

Usage
-----
    # Crawl a topic
    python main.py crawl --query "machine learning transformers" --pages 200

    # Search the index
    python main.py search --query "attention mechanism in NLP"

    # Start the API server
    python main.py serve

    # Crawl + immediately search
    python main.py crawl-search --query "quantum computing breakthroughs" --pages 100
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from queue import Queue

from config.settings import settings
from utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


# ── Crawl command ─────────────────────────────────────────────────────────────

async def cmd_crawl(args: argparse.Namespace) -> None:
    from crawler.crawler_manager import CrawlerManager
    from indexer.indexer import Indexer

    logger.info("=== CRAWL MODE ===")
    logger.info("Query   : %s", args.query)
    logger.info("Max pages: %d", args.pages)

    content_queue: Queue = Queue(maxsize=300)
    crawler = CrawlerManager(content_queue=content_queue)
    crawler._max_pages = args.pages

    if args.seed_urls:
        seed_urls = [u.strip() for u in args.seed_urls.split(",") if u.strip()]
    else:
        seed_urls = None

    indexer = Indexer(content_queue=content_queue)
    indexer.start()

    pages = await crawler.crawl(query=args.query, seed_urls=seed_urls)
    indexer.stop(timeout=300.0)

    print("\n" + "=" * 60)
    print("CRAWL COMPLETE")
    print("=" * 60)
    stats = {**crawler.stats(), **indexer.stats()}
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
    print("=" * 60)


# ── Search command ────────────────────────────────────────────────────────────

def cmd_search(args: argparse.Namespace) -> None:
    from search.search_engine import SearchEngine

    logger.info("=== SEARCH MODE ===")
    logger.info("Query: %s", args.query)

    engine = SearchEngine(auto_crawl=False)
    response = engine.search(query=args.query)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    if args.full:
        print(response.to_full_json())
    else:
        print(response.to_json())

    if response.results:
        print(f"\nQuery time: {response.query_time_ms:.0f}ms")
        if response.expanded_terms:
            print(f"Expansions: {', '.join(response.expanded_terms[:5])}")

        print("\nTop results:")
        for i, r in enumerate(response.results, 1):
            print(f"\n  [{i}] {r.url}")
            print(f"       Score : {r.score:.4f}")
            print(f"       Title : {r.title or 'N/A'}")
            print(f"       Snippet: {r.snippet[:150]}...")
    else:
        print("\nNo results found. Run a crawl first:")
        print(f'  python main.py crawl --query "{args.query}"')

    print("=" * 60)


# ── Serve command ─────────────────────────────────────────────────────────────

def cmd_serve(_args: argparse.Namespace) -> None:
    from api.app import run_server
    logger.info("=== API SERVER MODE ===")
    run_server()


# ── Crawl-then-search command ─────────────────────────────────────────────────

async def cmd_crawl_search(args: argparse.Namespace) -> None:
    logger.info("=== CRAWL + SEARCH MODE ===")
    await cmd_crawl(args)
    cmd_search(args)


# ── Stats command ─────────────────────────────────────────────────────────────

def cmd_stats(_args: argparse.Namespace) -> None:
    from indexer.vector_store import vector_store
    from storage.metadata_store import metadata_store

    print("\n" + "=" * 60)
    print("SYSTEM STATS")
    print("=" * 60)
    print(f"  Vector store chunks : {vector_store.count()}")
    print(f"  Metadata pages      : {metadata_store.count()}")
    print("=" * 60)

    recent = metadata_store.recent(limit=10)
    if recent:
        print("\nRecently crawled pages:")
        for p in recent:
            import datetime
            ts = datetime.datetime.fromtimestamp(p.crawl_timestamp).strftime("%H:%M:%S")
            print(f"  [{ts}] {'✓' if p.indexed else '○'} {p.url[:80]}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic_crawler",
        description="Semantic Web Crawler & Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # crawl
    p_crawl = sub.add_parser("crawl", help="Crawl and index pages for a query")
    p_crawl.add_argument("--query", required=True, help="Crawl focus query")
    p_crawl.add_argument("--pages", type=int, default=100, help="Max pages to crawl")
    p_crawl.add_argument("--seed-urls", default=None, help="Comma-separated seed URLs")

    # search
    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("--query", required=True, help="Search query")
    p_search.add_argument("--full", action="store_true", help="Output full JSON with snippets")

    # serve
    sub.add_parser("serve", help="Start the FastAPI REST server")

    # crawl-search
    p_cs = sub.add_parser("crawl-search", help="Crawl then immediately search")
    p_cs.add_argument("--query", required=True, help="Query for both crawl and search")
    p_cs.add_argument("--pages", type=int, default=50, help="Max pages to crawl")
    p_cs.add_argument("--seed-urls", default=None)
    p_cs.add_argument("--full", action="store_true")

    # stats
    sub.add_parser("stats", help="Show system statistics")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "crawl":
        asyncio.run(cmd_crawl(args))
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "crawl-search":
        asyncio.run(cmd_crawl_search(args))
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
