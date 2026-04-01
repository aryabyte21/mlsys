import os

# Model: override with MODEL_NAME env var to point to a local path on Vast.ai
# e.g. export MODEL_NAME=/workspace/hf-cache/models--Qwen.../snapshots/...
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
MAX_MODEL_LENGTH = 8192

# Speculative Decoding Constants
# Using a 0.5B Qwen model to draft tokens for your 4B model (must share the exact same tokenizer)
SPECULATIVE_MODEL_NAME = os.environ.get("SPECULATIVE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
NUM_SPECULATIVE_TOKENS = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "5"))

# Opt 2: Chunked Prefill tuning parameters
# Tune these to find the best P99 latency vs throughput tradeoff
CHUNKED_PREFILL_ENABLED = True
MAX_NUM_BATCHED_TOKENS = 2048   # Chunk size C: try 256, 512, 1024, 2048
MAX_NUM_SEQS = 256              # Max batch size: try 64, 128, 256
GPU_MEMORY_UTILIZATION = 0.90  # VRAM allocation: try 0.90, 0.95
ENABLE_PREFIX_CACHING = True    # APC: reuse KV cache for shared prefixes
