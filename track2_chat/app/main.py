import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from app.cache import ResponseCache
from app.chat_engine import ChatEngine
from app.constants import CACHE_MAX_SIZE
from app.schemas import ChatMessage, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = ChatEngine()
cache = ResponseCache(max_size=CACHE_MAX_SIZE)


async def _init_engine():
    try:
        await engine.initialize()
        # Warmup: send a dummy request to prime CUDA graphs, prefix cache, and JIT
        logger.info("Sending warmup request...")
        warmup_req = ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0,
            max_tokens=1,
        )
        await engine.generate(warmup_req)
        logger.info("Warmup complete!")
    except Exception:
        logger.exception("Engine initialization FAILED!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting engine initialization in background...")
    asyncio.create_task(_init_engine())
    yield


app = FastAPI(
    title="Track 2: Chat Engine",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


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
