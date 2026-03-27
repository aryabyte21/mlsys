import asyncio
import hashlib
import json
from collections import OrderedDict

import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas import ChatMessage
from normalize import extract_keywords


class ResponseCache:
    """Exact-match response cache with inflight request deduplication.

    Two layers:
    1. Persistent cache — stores completed responses (LRU eviction)
    2. Inflight dedup — coalesces concurrent identical requests so only one
       hits the GPU while the rest await the same result
    """

    def __init__(self, max_size: int = 16384):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[dict]] = {}
        self._max_size = max_size
        self._embedder: SentenceTransformer | None = None
        self.hits = 0
        self.misses = 0
        self.dedup_hits = 0

    def _make_key(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> str | None:
        if temperature > 0:
            return None
        data = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages], sort_keys=True
        )
        key_str = f"{data}|t={temperature}|m={max_tokens}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder
        
    def _get_embedding(self, text:str) -> np.ndarray:
        model = self._get_embedder()
        emb = model.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype="float32")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
        
    def get(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> dict | None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def semantic_get(self, messages: list[ChatMessage]):
        query = messages[-1].content
        
        query_keywords = extract_keywords(query)
        query_embedding = self._get_embedding(query)
        
        best_score = -1
        best_response = None
        
        for item in self._cache.values():
            if "embedding" not in item:
                continue
            
            if not query_keywords.intersection(item["keywords"]):
                continue
            
            sim = self._cosine_similarity(query_embedding, item["embedding"])
            
            if sim > best_score:
                best_score = sim
                best_response = item
                
        if best_score > 0.85:
            return best_response
        
        return None
            
    def get_inflight(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> asyncio.Future[dict] | None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        future = self._inflight.get(key)
        if future is not None:
            self.dedup_hits += 1
        return future

    def start_inflight(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> asyncio.Future[dict] | None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        return future

    def complete_inflight(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        response: dict,
    ) -> None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return
        
        keywords = extract_keywords(messages[-1].content)
        embedding = self._get_embedding(messages[-1].content)
        
        self._cache[key] = {
            **response,
            "keywords": keywords,
            "embedding": embedding,
        }
        
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_result(response)

    def fail_inflight(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        error: Exception,
    ) -> None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_exception(error)
