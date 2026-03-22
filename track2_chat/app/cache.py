import asyncio
import hashlib
import json
from collections import OrderedDict

from app.schemas import ChatMessage


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
        self._cache[key] = response
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
