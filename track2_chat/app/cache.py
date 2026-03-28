import asyncio
import hashlib
import struct
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

    def make_key(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> str | None:
        """Build a deterministic cache key for deterministic requests only."""
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

    def get_inflight_by_key(self, key: str) -> asyncio.Future[dict] | None:
        future = self._inflight.get(key)
        if future is not None:
            self.dedup_hits += 1
        return future

    def claim_inflight_by_key(self, key: str) -> tuple[asyncio.Future[dict], bool]:
        """Return (future, is_owner). Owner must run inference and resolve it."""
        future = self._inflight.get(key)
        if future is not None:
            self.dedup_hits += 1
            return future, False

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        return future, True

    def start_inflight_by_key(self, key: str) -> asyncio.Future[dict]:
        future: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        return future

    def complete_inflight_by_key(self, key: str, response: dict) -> None:
        self._cache[key] = response
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_result(response)

    def fail_inflight_by_key(self, key: str, error: Exception) -> None:
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_exception(error)

    def get(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> dict | None:
        key = self.make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        return self.get_by_key(key)

    def get_inflight(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> asyncio.Future[dict] | None:
        key = self.make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        return self.get_inflight_by_key(key)

    def start_inflight(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> asyncio.Future[dict] | None:
        key = self.make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        return self.start_inflight_by_key(key)

    def complete_inflight(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        response: dict,
    ) -> None:
        key = self.make_key(messages, temperature, max_tokens)
        if key is None:
            return
        self.complete_inflight_by_key(key, response)

    def fail_inflight(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        error: Exception,
    ) -> None:
        key = self.make_key(messages, temperature, max_tokens)
        if key is None:
            return
        self.fail_inflight_by_key(key, error)
