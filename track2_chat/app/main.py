import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.cache import ResponseCache
from app.chat_engine import ChatEngine
from app.constants import CACHE_MAX_SIZE
from app.schemas import ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = ChatEngine()
cache = ResponseCache(max_size=CACHE_MAX_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting engine initialization...")
    await engine.initialize()
    logger.info("Engine ready, accepting requests.")
    yield


app = FastAPI(title="Track 2: Chat Engine", lifespan=lifespan)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine is still initializing")

    # Layer 1: persistent cache hit
    cached = cache.get(request.messages, request.temperature, request.max_tokens)
    if cached is not None:
        return cached

    # Layer 2: another coroutine is already computing this exact request — wait for it
    inflight = cache.get_inflight(request.messages, request.temperature, request.max_tokens)
    if inflight is not None:
        return await inflight

    # Layer 3: we're the first — register inflight, run inference, resolve waiters
    future = cache.start_inflight(request.messages, request.temperature, request.max_tokens)
    try:
        response = await engine.generate(request)
        cache.complete_inflight(
            request.messages, request.temperature, request.max_tokens, response
        )
        return response
    except Exception as e:
        cache.fail_inflight(request.messages, request.temperature, request.max_tokens, e)
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if engine.is_ready:
        return {"status": "ready", "message": "Chat engine is initialized and ready."}
    raise HTTPException(status_code=503, detail="Engine is initializing")
