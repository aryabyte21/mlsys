import asyncio
import hashlib
import json
import faiss
import numpy as np

from collections import OrderedDict
from sentence_transformers import SentenceTransformer

from app.schemas import ChatMessage
from app.normalize import extract_keywords


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
        
        # FAISS variable
        self._id_to_key: dict[int, str] = {}        # Map FAISS id -> original request key for debugging
        self._key_to_id: dict[str, int] = {}
        self._faiss_index = None
        self._id_counter = 0

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
    
    # Initialize FAISS if it's still empty
    def _init_faiss(self, dim: int):
        if self._faiss_index is None:
            base = faiss.IndexFlatIP(dim)
            self._faiss_index  = faiss.IndexIDMap(base)
    
    def get(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> dict | None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            cached = self._cache[key]
            return cached
        
        self.misses += 1
        return None

    def semantic_get(self, messages: list[ChatMessage], top_k: int = 5, semantic_threshold: float = 0.8):
        
        # Sanity check
        if self._faiss_index is None or len(self._cache) == 0:
            return None
        
        query = messages[-1].content
        
        query_embedding = self._get_embedding(query)
        
        # FAISS index search
        query_embedding = np.expand_dims(query_embedding, axis=0)
        scores, indices = self._faiss_index.search(query_embedding, top_k)
        
        best_score = -1
        best_response = None
        
        for score, faiss_id in zip(scores[0], indices[0]):
            if faiss_id == -1:
                continue
            
            key = self._id_to_key.get(int(faiss_id))
            if key is None:
                continue
            
            # item = self._cache.get(int(faiss_id))
            item = self._cache.get(key)
            
            if item is None:
                continue
            
            if key not in self._cache:
                continue
            
            # Keyword filter
            query_keywords = extract_keywords(query)
            
            if not query_keywords.intersection(self._cache[key].get("keywords", set())):
                continue
            
            if score > best_score:
                best_score = score
                best_key = key
            
        if best_score > semantic_threshold and best_key is not None:
            self._cache.move_to_end(best_key)
            return self._cache[best_key]
        
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
        
        
        # keywords = extract_keywords(messages[-1].content)
        # embedding = self._get_embedding(messages[-1].content)
        
        if key in self._key_to_id:
            faiss_id = self._key_to_id[key]
        else:
            faiss_id = self._id_counter
            self._id_counter += 1
            
            self._key_to_id[key] = faiss_id
            self._id_to_key[faiss_id] = key
        
        self._cache[key] = response
        
        embedding = self._get_embedding(messages[-1].content)
        
        self._init_faiss(len(embedding))
        
        self._faiss_index.add_with_ids(
            np.expand_dims(embedding, axis=0),
            np.array([faiss_id], dtype="int64")
            )
        
        if len(self._cache) > self._max_size:
            old_key, _ = self._cache.popitem(last=False)
            old_id = self._key_to_id.pop(old_key, None)
            # old_id = old_item["faiss_id"]
            
            # self._faiss_index.remove_ids(np.array([old_id], dtype="int64"))
            
            # old_key = self._id_to_key.pop(old_id, None)
            if old_key is not None:
                # self._key_to_id.pop(old_key, None)
                self._id_to_key.pop(old_id, None)
                
                if self._faiss_index is not None:
                    self._faiss_index.remove_ids(
                        np.array([old_id], dtype="int64")
                    )
            
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
