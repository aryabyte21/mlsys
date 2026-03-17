# Optimization Plan

> Last updated: 2026-03-18

## Current Goal

Build and optimize a high-throughput vLLM serving engine for Track 2 (Customer Support Chatbot)
targeting maximum throughput and minimum latency at 128 concurrency on a single RTX 5080.

## Hypothesis

The biggest performance gains will come from (in order):
1. **FP8 quantization** — halves model memory, freeing VRAM for more concurrent KV caches
2. **FP8 KV cache** — halves KV cache memory, critical for 128 concurrency
3. **Reduced max_model_len** — inputs are 6-92 chars, outputs max 256 tokens → 1024 is plenty
4. **Exact-match response cache + inflight dedup** — 36.1% duplicates; dedup coalesces concurrent identical requests
5. **N-gram speculative decoding** — 3 draft tokens per step, formulaic responses have high acceptance rate
6. **max_num_batched_tokens=8192** — default 2048 is under-tuned for high-concurrency decode
7. **APC + chunked prefill** — built-in vLLM features for prefix sharing and P99 reduction
8. **Disabling Qwen3 thinking mode** — prevents wasted tokens on `<think>` blocks
9. **V1 engine + CUDA graphs** — async scheduling + 8x throughput over eager mode on Blackwell

## Key Data Insights

- 13,435 total benchmark queries, 8,582 unique (36.1% duplicate rate)
- All single-turn: `[{"role": "user", "content": "..."}]` — no multi-turn sessions
- Benchmark uses `temperature=0` (greedy/deterministic), `max_tokens=256`
- Input lengths: 6-92 characters (~15-40 tokens after tokenization)
- Response format: `{"output": "...", "logprobs": [...]}` (NOT OpenAI format)
- Benchmark computes P99 (not P95 as spec says)

## Architecture

```
Incoming Request (conc. 128)
    │
    ▼
┌──────────────────────┐
│  Exact-Match Cache    │──hit──▶ Return instantly (0ms)
└──────────┬───────────┘
           │ miss
           ▼
┌──────────────────────┐
│  Inflight Dedup       │──dup──▶ Await first result (no GPU cost)
└──────────┬───────────┘
           │ first
           ▼
┌─────────────────────────────────────────────────┐
│  vLLM V1 AsyncLLMEngine (Blackwell-optimized)    │
│  ┌────────────────────────────────────────────┐  │
│  │ Model: Qwen3-4B-Instruct-2507 (FP8)       │  │
│  │ KV Cache: FP8, max_model_len=1024          │  │
│  │ Attention: FlashInfer                       │  │
│  │ Speculative: n-gram (3 tokens, lookup 2-4)  │  │
│  │ APC: enabled (shared prefix caching)        │  │
│  │ Chunked Prefill: enabled (P99 reduction)    │  │
│  │ max_num_seqs: 256                           │  │
│  │ max_num_batched_tokens: 8192                │  │
│  │ gpu_memory_utilization: 0.95                │  │
│  │ CUDA graphs: enabled (8x over eager)        │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │
         ▼
    Cache result, resolve dedup waiters
```

## Full Optimization Stack

### Tier 1 — High Impact

| # | Optimization | Mechanism | Status |
|---|-------------|-----------|--------|
| 1 | FP8 model quantization | `quantization="fp8"` | Done |
| 2 | FP8 KV cache | `kv_cache_dtype="fp8"` | Done |
| 3 | Reduced max_model_len | `max_model_len=1024` (from 8192) | Done |
| 4 | Exact-match response cache | SHA256 LRU cache, 36% hit rate | Done |
| 5 | Inflight request dedup | asyncio futures coalesce concurrent identical requests | Done |
| 6 | N-gram speculative decoding | 3 tokens, lookup 2-4, disable_logprobs=False | Done |
| 7 | Disable Qwen3 thinking | `enable_thinking=False` in chat template | Done |

### Tier 2 — Medium Impact

| # | Optimization | Mechanism | Status |
|---|-------------|-----------|--------|
| 8 | max_num_batched_tokens=8192 | Up from 2048 default, reduces scheduling overhead | Done |
| 9 | V1 engine | `VLLM_USE_V1=1`, async scheduling, overlapped CPU/GPU | Done |
| 10 | CUDA graphs | Default (not enforce_eager), 8x throughput on Blackwell | Done |
| 11 | Automatic Prefix Caching | `enable_prefix_caching=True` | Done |
| 12 | Chunked prefill | `enable_chunked_prefill=True` | Done |
| 13 | GPU memory utilization | `gpu_memory_utilization=0.95` (from 0.9) | Done |
| 14 | Batch size tuning | `max_num_seqs=256` | Done |
| 15 | CUDA memory defrag | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Done |

### Tier 3 — Fine-Tuning (After Benchmark)

| # | Optimization | Notes |
|---|-------------|-------|
| 16 | max_model_len reduction to 512 | If 1024 is overkill, halves KV per seq |
| 17 | spec_num_tokens tuning (2 vs 3 vs 5) | Measure acceptance rate, find sweet spot |
| 18 | gpu_memory_utilization=0.98 | Push VRAM harder if no OOM |

### Deprioritized

| # | Optimization | Reason |
|---|-------------|--------|
| - | Semantic/fuzzy cache | Risk of incorrect logprobs → perplexity degradation |
| - | Session-aware scheduler | Benchmark is single-turn, no sessions to track |
| - | CPU KV cache offload | V1 engine uses recomputation, not CPU swap |
| - | enforce_eager | Loses 8x throughput on Blackwell due to no CUDA graphs |
| - | num_scheduler_steps | V0-only, incompatible with spec decode, V1 does it natively |

## Experiment Log

| # | Date | Config Change | P50 (ms) | P99 (ms) | Throughput (req/s) | Perplexity | Verdict |
|---|------|--------------|----------|----------|-------------------|------------|---------|
| 0 | — | baseline (starter kit, no opts) | — | — | — | — | pending |
| 1 | 2026-03-17 | Tier 1+2 on Modal L4 (no FP8, no spec) | 9859 | 19053 | 13.33 | 1.2000 | L4 baseline — high latency from network + weak GPU |
| 2 | — | Full stack on RTX 5080 (FP8 + spec decode + V1) | — | — | — | — | pending |

## Discoveries & Surprises

- **36.1% duplicate rate** in training data — exact-match caching is extremely high impact
- **Benchmark is single-turn** (not multi-turn) — session-based optimizations irrelevant
- **Benchmark uses P99** (code) not P95 (spec doc) — target P99 reduction
- **temperature=0 always** — makes caching lossless (deterministic outputs)
- **max_tokens=256** is set by benchmark, not configurable from engine side
- **vLLM spec decode disable_logprobs defaults to True** — MUST set False or logprobs silently missing
- **V1 engine doesn't use CPU swap** — uses recomputation-based preemption instead
- **CUDA graphs give 8x throughput over enforce_eager on Blackwell** (from vLLM benchmarks)
- **max_num_batched_tokens default (2048) is under-tuned** — vLLM source has TODO to tune it
- **transformers>=4.53.0 breaks vLLM tokenizer** — all_special_tokens_extended removed, must pin

## Key Techniques & Skills

- FP8 quantization on RTX 5080 (Blackwell) for inference memory savings
- FlashInfer attention backend for CUDA 12.8 compatibility
- Application-layer response caching with SHA256 keys + inflight dedup
- N-gram speculative decoding for formulaic customer service responses
- vLLM V1 engine with async scheduling and CUDA graphs
- PYTORCH_CUDA_ALLOC_CONF for memory defragmentation

## Decisions

- **FP8 over INT4/AWQ**: FP8 has minimal quality loss and native hardware support on RTX 5080
- **max_model_len=1024**: Conservative but safe; max prompt+response is ~350 tokens
- **N-gram spec decode over Eagle/draft model**: No extra model needed, works in V1, low risk
- **3 speculative tokens**: Balance between acceptance rate and overhead for short prompts
- **disable_logprobs=False**: Critical — default True silently drops logprobs
- **Pin vllm==0.8.5.post1 + transformers<4.53.0**: Latest versions have breaking changes

## Dead Ends

- Session-aware scheduler — benchmark has no multi-turn sessions
- Session-TTL KV eviction — same reason
- Semantic/fuzzy caching — risks perplexity degradation for uncertain hit rate improvement
- enforce_eager — loses 8x throughput on Blackwell
- num_scheduler_steps — V0-only, V1 already does async scheduling natively

## Next Steps

1. Deploy on Vast.ai RTX 5080 with Docker image and run benchmark
2. If spec decode causes issues with logprobs, disable it via `SPEC_DECODE_ENABLED=false`
3. Fine-tune: try max_model_len=512, spec_num_tokens=5, gpu_mem=0.98
4. Final Docker image push to GHCR for submission
