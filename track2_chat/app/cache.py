import asyncio
import hashlib
import logging
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.normalize import extract_keywords
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

# Shared thread pool for CPU-bound embedding work
_embed_pool = ThreadPoolExecutor(max_workers=4)


class ResponseCache:
    """Multi-layer response cache with semantic matching.

    Layers (checked in order):
    1. Exact-match — SHA256 hash of request (instant, lossless)
    2. Inflight dedup — coalesce concurrent identical requests
    3. Keyword similarity — Jaccard similarity of extracted keywords
    4. Embedding similarity — FAISS cosine search with MiniLM embeddings
    """

    def __init__(
        self,
        max_size: int = 16384,
        keyword_threshold: float = 0.55,
        semantic_threshold: float = 0.75,
        hard_stop: float = 0.85,
    ):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[dict]] = {}
        self._max_size = max_size

        # Counters
        self.hits = 0
        self.misses = 0
        self.dedup_hits = 0
        self.semantic_hits = 0
        self.keyword_hits = 0

        # Semantic cache config
        self._keyword_threshold = keyword_threshold
        self._semantic_threshold = semantic_threshold
        self._hard_stop = hard_stop

        # Keyword index: cache_key -> set of keywords
        self._key_keywords: dict[str, set[str]] = {}

        # FAISS embedding index
        self._embedder: SentenceTransformer | None = None
        self._faiss_index: faiss.IndexIDMap | None = None
        self._id_to_key: dict[int, str] = {}
        self._key_to_id: dict[str, int] = {}
        self._id_counter = 0

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._embedder

    def _embed_text(self, text: str) -> np.ndarray:
        model = self._get_embedder()
        emb = model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype="float32")

    def _init_faiss(self, dim: int):
        if self._faiss_index is None:
            base = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIDMap(base)

    # ── Exact-match layer ──

    def make_key(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> str | None:
        if temperature > 0:
            return None

        hasher = hashlib.sha256()
        hasher.update(struct.pack("<d", temperature))
        hasher.update(struct.pack("<I", max_tokens))
        hasher.update(struct.pack("<I", len(messages)))

        for message in messages:
            role_bytes = message.role.encode("utf-8")
            content_bytes = message.content.encode("utf-8")
            hasher.update(struct.pack("<I", len(role_bytes)))
            hasher.update(role_bytes)
            hasher.update(struct.pack("<I", len(content_bytes)))
            hasher.update(content_bytes)

        return hasher.hexdigest()

    def get_by_key(self, key: str) -> dict | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    # ── Inflight dedup layer ──

    def claim_inflight_by_key(self, key: str) -> tuple[asyncio.Future[dict], bool]:
        """Return (future, is_owner). Owner must run inference and resolve it."""
        future = self._inflight.get(key)
        if future is not None:
            self.dedup_hits += 1
            return future, False

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        return future, True

    # ── Keyword similarity layer ──

    def _keyword_search(self, query: str) -> dict | None:
        query_keywords = extract_keywords(query)
        if not query_keywords:
            return None

        best_key = None
        best_score = 0.0

        for key, item_keywords in self._key_keywords.items():
            if not item_keywords:
                continue

            intersection = query_keywords & item_keywords
            if not intersection:
                continue

            score = len(intersection) / len(query_keywords | item_keywords)

            if score >= self._hard_stop:
                self._cache.move_to_end(key)
                self.keyword_hits += 1
                return self._cache[key]

            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= self._keyword_threshold and best_key is not None:
            self._cache.move_to_end(best_key)
            self.keyword_hits += 1
            return self._cache[best_key]

        return None

    # ── Embedding similarity layer ──

    def _embedding_search(self, query: str, top_k: int = 3) -> dict | None:
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return None

        query_emb = self._embed_text(query)
        query_emb = np.expand_dims(query_emb, axis=0)

        scores, indices = self._faiss_index.search(query_emb, top_k)

        best_score = -1.0
        best_key = None

        for score, faiss_id in zip(scores[0], indices[0]):
            if faiss_id == -1:
                continue

            key = self._id_to_key.get(int(faiss_id))
            if key is None or key not in self._cache:
                continue

            if score >= self._hard_stop:
                self._cache.move_to_end(key)
                self.semantic_hits += 1
                return self._cache[key]

            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= self._semantic_threshold and best_key is not None:
            self._cache.move_to_end(best_key)
            self.semantic_hits += 1
            return self._cache[best_key]

        return None

    # ── Combined semantic lookup (keyword → embedding) ──

    def semantic_get(self, messages: list[ChatMessage]) -> dict | None:
        if not self._cache:
            return None

        query = messages[-1].content

        result = self._keyword_search(query)
        if result is not None:
            return result

        result = self._embedding_search(query)
        return result

    async def semantic_get_async(self, messages: list[ChatMessage]) -> dict | None:
        """Non-blocking version that runs full semantic search in thread pool."""
        if not self._cache:
            return None

        query = messages[-1].content

        # Run entire semantic search (keyword + embedding) in thread pool
        # to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _embed_pool, self.semantic_get, messages
        )
        return result

    # ── Store with semantic indexing ──

    def _index_entry(self, key: str, query_text: str):
        """Add keywords and embedding to semantic indices."""
        self._key_keywords[key] = extract_keywords(query_text)

        embedding = self._embed_text(query_text)
        self._init_faiss(len(embedding))

        if key in self._key_to_id:
            faiss_id = self._key_to_id[key]
        else:
            faiss_id = self._id_counter
            self._id_counter += 1
            self._key_to_id[key] = faiss_id
            self._id_to_key[faiss_id] = key

        self._faiss_index.add_with_ids(
            np.expand_dims(embedding, axis=0),
            np.array([faiss_id], dtype="int64"),
        )

    def _evict_oldest(self):
        if len(self._cache) <= self._max_size:
            return

        old_key, _ = self._cache.popitem(last=False)
        self._key_keywords.pop(old_key, None)

        old_id = self._key_to_id.pop(old_key, None)
        if old_id is not None:
            self._id_to_key.pop(old_id, None)
            if self._faiss_index is not None:
                self._faiss_index.remove_ids(np.array([old_id], dtype="int64"))

    # ── Inflight + store operations ──

    def complete_inflight_by_key(
        self, key: str, response: dict, query_text: str | None = None
    ) -> None:
        self._cache[key] = response
        self._evict_oldest()

        # Resolve inflight waiters FIRST (don't block on indexing)
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_result(response)

        # Index for semantic search (blocking but fast enough)
        if query_text is not None:
            try:
                self._index_entry(key, query_text)
            except Exception:
                logger.debug("Failed to index entry for semantic cache", exc_info=True)

    def fail_inflight_by_key(self, key: str, error: Exception) -> None:
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_exception(error)

    def log_stats(self):
        total = self.hits + self.misses + self.dedup_hits + self.semantic_hits + self.keyword_hits
        logger.info(
            "Cache stats: exact=%d, keyword=%d, semantic=%d, dedup=%d, miss=%d, total=%d, "
            "cache_size=%d, faiss_size=%d",
            self.hits, self.keyword_hits, self.semantic_hits,
            self.dedup_hits, self.misses, total,
            len(self._cache),
            self._faiss_index.ntotal if self._faiss_index else 0,
        )
