"""
indexer/vector_store.py
------------------------
Pluggable vector database for storing and querying chunk embeddings.

Two backends
------------
"chroma"  — ChromaDB (default).  Persistent, metadata-filtering, cosine
             similarity out-of-the-box.  No extra index management needed.
"faiss"   — Facebook AI Similarity Search.  Fastest for large corpora,
             but requires manual metadata management alongside the index.

Both expose the same VectorStore interface:
    add(chunks, metadata)
    search(query_vector, top_k, filter?) -> list[SearchResult]
    delete(ids)
    count() -> int
    persist()
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    url: str
    score: float               # cosine similarity in [0, 1]
    metadata: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        from utils.url_utils import extract_domain
        return extract_domain(self.url)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseVectorStore:
    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_meta: Optional[dict] = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        pass

    def is_empty(self) -> bool:
        return self.count() == 0


# ── ChromaDB backend ──────────────────────────────────────────────────────────

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store.

    ChromaDB handles persistence, cosine similarity, and metadata filtering
    natively.  We use the EphemeralClient for tests and PersistentClient for
    production.
    """

    def __init__(self, persist_path: str, collection_name: str) -> None:
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings as ChromaSettings  # type: ignore
        except ImportError as exc:
            raise ImportError("chromadb not installed: pip install chromadb") from exc

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready: collection='%s' count=%d",
            collection_name, self._collection.count(),
        )

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Chroma expects list of lists
        emb_list = embeddings.tolist()

        # Sanitise metadatas (Chroma only accepts str/int/float/bool values)
        safe_metas = [_sanitise_meta(m) for m in metadatas]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=emb_list,
                documents=texts,
                metadatas=safe_metas,
            )
        except Exception as exc:
            logger.error("ChromaVectorStore.add error: %s", exc)
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_meta: Optional[dict] = None,
    ) -> list[SearchResult]:
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": min(top_k, max(1, self._collection.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if filter_meta:
                kwargs["where"] = filter_meta

            res = self._collection.query(**kwargs)
            results: list[SearchResult] = []

            for i, doc in enumerate(res["documents"][0]):
                dist = res["distances"][0][i]
                # Chroma cosine distance = 1 - cosine_similarity
                score = max(0.0, 1.0 - dist)
                meta = res["metadatas"][0][i] or {}
                chunk_id = res["ids"][0][i]
                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=doc,
                        url=meta.get("url", ""),
                        score=score,
                        metadata=meta,
                    )
                )
            return results
        except Exception as exc:
            logger.error("ChromaVectorStore.search error: %s", exc)
            return []

    def delete(self, ids: list[str]) -> None:
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:
            logger.warning("ChromaVectorStore.delete error: %s", exc)

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def persist(self) -> None:
        pass  # PersistentClient auto-persists


# ── FAISS backend ─────────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store with a companion JSON metadata file.

    Uses IndexFlatIP (inner product) on L2-normalised vectors, which
    gives cosine similarity scores.  Persisted as two files:
        <path>/faiss.index
        <path>/faiss_meta.json
    """

    def __init__(self, persist_path: str) -> None:
        try:
            import faiss  # type: ignore
            self._faiss = faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu not installed: pip install faiss-cpu") from exc

        self._path = Path(persist_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._index_file = self._path / "faiss.index"
        self._meta_file = self._path / "faiss_meta.json"

        self._index = None
        self._meta: list[dict] = []   # parallel array to FAISS index
        self._texts: list[str] = []   # parallel text storage
        self._ids: list[str] = []
        self._dim: Optional[int] = None

        self._load()

    def _load(self) -> None:
        if self._index_file.exists():
            self._index = self._faiss.read_index(str(self._index_file))
            meta_data = json.loads(self._meta_file.read_text())
            self._meta = meta_data.get("meta", [])
            self._texts = meta_data.get("texts", [])
            self._ids = meta_data.get("ids", [])
            self._dim = self._index.d
            logger.info("FAISSVectorStore loaded: %d vectors (dim=%d)", len(self._ids), self._dim)

    def _init_index(self, dim: int) -> None:
        self._dim = dim
        # Inner-product index (cosine similarity after L2-normalisation)
        self._index = self._faiss.IndexFlatIP(dim)

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        if self._index is None:
            self._init_index(embeddings.shape[1])

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        vecs = embeddings.astype(np.float32)
        self._faiss.normalize_L2(vecs)
        self._index.add(vecs)
        self._texts.extend(texts)
        self._meta.extend(metadatas)
        self._ids.extend(ids)
        self.persist()
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_meta: Optional[dict] = None,
    ) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_embedding.astype(np.float32).reshape(1, -1)
        self._faiss.normalize_L2(q)

        k = min(top_k * 3, self._index.ntotal)  # fetch more for filtering
        scores, indices = self._index.search(q, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            meta = self._meta[idx]

            # Apply metadata filter
            if filter_meta:
                if not all(meta.get(k) == v for k, v in filter_meta.items()):
                    continue

            results.append(
                SearchResult(
                    chunk_id=self._ids[idx],
                    text=self._texts[idx],
                    url=meta.get("url", ""),
                    score=float(score),
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> None:
        # FAISS FlatIP doesn't support deletion; rebuild index without ids
        id_set = set(ids)
        keep = [i for i, cid in enumerate(self._ids) if cid not in id_set]
        if not keep:
            return
        vecs = self._index.reconstruct_n(0, self._index.ntotal)
        self._init_index(self._dim)
        self._index.add(vecs[keep])
        self._texts = [self._texts[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._ids = [self._ids[i] for i in keep]
        self.persist()

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def persist(self) -> None:
        if self._index is None:
            return
        self._faiss.write_index(self._index, str(self._index_file))
        self._meta_file.write_text(
            json.dumps({"meta": self._meta, "texts": self._texts, "ids": self._ids})
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_meta(meta: dict) -> dict:
    """Ensure all values are Chroma-compatible types."""
    out: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    return out


# ── Factory ───────────────────────────────────────────────────────────────────

def build_vector_store() -> BaseVectorStore:
    cfg = settings.vector_store
    backend = cfg.get("backend", "chroma")
    persist_path = cfg.get("persist_path", "data/vector_db")
    collection_name = cfg.get("collection_name", "semantic_web")

    if backend == "faiss":
        return FAISSVectorStore(persist_path=persist_path)
    return ChromaVectorStore(
        persist_path=persist_path,
        collection_name=collection_name,
    )


# Module-level singleton
vector_store: BaseVectorStore = build_vector_store()
