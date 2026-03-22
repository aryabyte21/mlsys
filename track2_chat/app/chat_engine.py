import logging
from functools import lru_cache

from app.schemas import ChatRequest
from app.constants import (
    ENABLE_CHUNKED_PREFILL,
    ENABLE_PREFIX_CACHING,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_DTYPE,
    MAX_MODEL_LENGTH,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_NAME,
    QUANTIZATION,
    SPEC_DECODE_ENABLED,
    SPEC_NUM_TOKENS,
    SPEC_PROMPT_LOOKUP_MAX,
    SPEC_PROMPT_LOOKUP_MIN,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _get_sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
    """Cache SamplingParams — benchmark always sends the same values."""
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,
    )


class ChatEngine:
    def __init__(self):
        self.engine = None
        self.tokenizer = None
        self.is_ready = False
        self._supports_thinking = None  # cache template capability

    async def initialize(self):
        if self.is_ready:
            return

        speculative_config = None
        if SPEC_DECODE_ENABLED:
            speculative_config = {
                "method": "ngram",
                "prompt_lookup_min": SPEC_PROMPT_LOOKUP_MIN,
                "prompt_lookup_max": SPEC_PROMPT_LOOKUP_MAX,
                "num_speculative_tokens": SPEC_NUM_TOKENS,
                "disable_logprobs": False,
            }

        logger.info(
            "Initializing vLLM: model=%s, max_model_len=%d, quant=%s, kv_dtype=%s, "
            "max_num_seqs=%d, max_batched_tokens=%d, prefix_caching=%s, "
            "chunked_prefill=%s, spec_decode=%s",
            MODEL_NAME, MAX_MODEL_LENGTH, QUANTIZATION, KV_CACHE_DTYPE,
            MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, ENABLE_PREFIX_CACHING,
            ENABLE_CHUNKED_PREFILL, speculative_config,
        )

        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LENGTH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            quantization=QUANTIZATION,
            kv_cache_dtype=KV_CACHE_DTYPE,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
            disable_log_stats=True,
            speculative_config=speculative_config,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = await self.engine.get_tokenizer()

        # Probe once whether the tokenizer supports enable_thinking
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self._supports_thinking = True
        except TypeError:
            self._supports_thinking = False

        self.is_ready = True
        logger.info("Engine initialization complete!")

    async def generate(self, request: ChatRequest) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Tokenize directly to token IDs — skip double tokenization
        if self._supports_thinking:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )

        # Reuse cached SamplingParams
        sampling_params = _get_sampling_params(request.temperature, request.max_tokens)

        final_output = None
        async for output in self.engine.generate(
            {"prompt_token_ids": token_ids}, sampling_params, random_uuid()
        ):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        output_data = final_output.outputs[0]
        if output_data.logprobs is None:
            raise RuntimeError("logprobs are missing from vLLM output")

        # Build response dict directly — skip Pydantic construction
        logprobs = []
        for i, token_id in enumerate(output_data.token_ids):
            if i < len(output_data.logprobs):
                step = output_data.logprobs[i]
                if token_id in step:
                    logprobs.append(step[token_id].logprob)
                else:
                    logprobs.append(0.0)

        return {"output": output_data.text, "logprobs": logprobs}
