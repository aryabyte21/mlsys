import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse

from app.cache import ResponseCache
from app.chat_engine import ChatEngine
from app.constants import CACHE_MAX_SIZE
from app.schemas import ChatMessage, ChatRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = ChatEngine()
cache = ResponseCache(max_size=CACHE_MAX_SIZE)


async def _init_engine():
    try:
        await engine.initialize()
        # Warmup: prime CUDA graphs, prefix cache, and JIT
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine is still initializing")

    cache_key = cache.make_key(request.messages, request.temperature, request.max_tokens)
    if cache_key is None:
        return await engine.generate(request)

    # Layer 1: persistent cache hit
    cached = cache.get_by_key(cache_key)
    if cached is not None:
        return cached

    # Layer 2+3: claim or join inflight computation in one dictionary lookup
    inflight, is_owner = cache.claim_inflight_by_key(cache_key)
    if not is_owner:
        return await inflight

    try:
        response = await engine.generate(request)
        cache.complete_inflight_by_key(cache_key, response)
        return response
    except Exception as e:
        cache.fail_inflight_by_key(cache_key, e)
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if engine.is_ready:
        return {"status": "ready", "message": "Chat engine is initialized and ready."}
    raise HTTPException(status_code=503, detail="Engine is initializing")
