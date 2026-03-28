# Optimization Plan

> Last updated: 2026-03-28

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
| 2 | 2026-03-21 | Vanilla baseline on H200 (no FP8, no V1, no spec, no FlashInfer) | 1219 | 2767 | 108.40 | 1.2001 | working baseline — all 13435 passed |
| 3 | 2026-03-21 | V1 engine only on H200 | 1226 | 2663 | 109.48 | 1.2001 | works, marginal improvement |
| 4 | 2026-03-21 | V1 + FP8 on H200 | 1140 | 2579 | 116.11 | 1.2009 | works, ~6% throughput gain |
| 5 | 2026-03-21 | V1 + FP8 on H200 (repeat) | 1142 | 2574 | 116.29 | 1.2009 | confirmed stable, all passed |
| 6 | 2026-03-22 | V1+FP8+max512+warmup on H100-47 (full mem) | 1722 | 3633 | 77.09 | 1.2009 | all passed, slower GPU than H200 |
| 7 | 2026-03-28 | V1+FP8+max512+gpu0.98+Pydantic skip on H200 | 1209 | 2696 | 110.17 | 1.2009 | all passed, app-level opts negligible on H200 — GPU is bottleneck |
| 8 | 2026-03-29 | FP8+max512+gpu0.98+swap0 on H200 | 1144 | 2604 | 115.89 | 1.2009 | ~same as run 4-5, swap_space=0 no effect |
| 9 | 2026-03-29 | FP8+max_batched=16384 on H200 | 1136 | 2608 | 116.02 | 1.2009 | identical — batch size irrelevant on H200 |
| 10 | 2026-03-29 | BF16 (no quant) on H200 | 1142 | 2592 | 115.55 | 1.2009 | identical — H200 has 141GB, memory never bottleneck |
| 11 | — | Full stack on RTX 5080 | — | — | — | — | pending — configs will diverge on 16GB |

## Discoveries & Surprises

- **36.1% duplicate rate** in training data — exact-match caching is extremely high impact
- **Benchmark is single-turn** (not multi-turn) — session-based optimizations irrelevant
- **Benchmark uses P99** (code) not P95 (spec doc) — target P99 reduction
- **temperature=0 always** — makes caching lossless (deterministic outputs)
- **max_tokens=256** is set by benchmark, not configurable from engine side
- **vLLM spec decode disable_logprobs defaults to True** — MUST set False or logprobs silently missing
- **V1 engine doesn't use CPU swap** — uses recomputation-based preemption instead
- **V1 engine + FP8 KV cache incompatible** in vLLM 0.8.5.post1 — `VLLM_USE_V1=1` raises NotImplementedError with `--kv-cache-dtype`
- **ngram spec decode on V1 is experimental** — causes EngineDeadError crashes under load
- **Eval uses P50 and P95** (not P99) per project spec, though benchmark script reports P99
- **FlashInfer + FP8 + V1 crashes** — EngineDeadError under load. FlashInfer FP8 attention with scale 1.0 causes engine core death
- **V1 + FP8 (no FlashInfer) is stable** — confirmed 116 req/s on H200, all 13435 passed, perplexity 1.2009
- **CUDA graphs give 8x throughput over enforce_eager on Blackwell** (from vLLM benchmarks)
- **max_num_batched_tokens default (2048) is under-tuned** — vLLM source has TODO to tune it
- **transformers>=4.53.0 breaks vLLM tokenizer** — all_special_tokens_extended removed, must pin
- **[CRITICAL] FP8 may be 3x SLOWER than AWQ on Blackwell (SM120)** — CUTLASS lacks SM120 FP8 GEMM kernels, online dynamic FP8 has 17.9% overhead vs BF16. RTX 5090 benchmarks: AWQ ~140 tok/s vs FP8 ~45 tok/s (vLLM issues #28234, #37242)
- **BF16 (no quant) may beat FP8 on RTX 5080** — 4B model is ~8GB in BF16, fits in 16GB with max_model_len=512. Must A/B test.
- **AWQ Marlin is the recommended quantization for Blackwell** — `quantization="awq_marlin"` with AWQ checkpoint
- **FP8 KV cache still broken on V1 + Blackwell** — PR #17005 closed, TRITON_MLA raises NotImplementedError for SM120
- **cudagraph_mode=FULL may help** — better for small models with short prompts, less memory overhead than default FULL_AND_PIECEWISE
- **max_num_batched_tokens could go to 16384+** — small model + short prompts benefit from larger batches
- **KV cache math for 16GB RTX 5080**: FP8 model (~4.2GB) → ~10.5GB KV → ~4778 blocks → ~149 max concurrent (worst case) or ~251 (typical). BF16 model (~8GB) → ~6.7GB KV → ~3037 blocks → ~94-159 concurrent.

## Key Techniques & Skills

- FP8 quantization on RTX 5080 (Blackwell) for inference memory savings
- FlashInfer attention backend for CUDA 12.8 compatibility
- Application-layer response caching with SHA256 keys + inflight dedup
- N-gram speculative decoding for formulaic customer service responses
- vLLM V1 engine with async scheduling and CUDA graphs
- PYTORCH_CUDA_ALLOC_CONF for memory defragmentation

## Decisions

- **FP8 over INT4/AWQ**: UNDER REVIEW — SM120 may lack native FP8 GEMM kernels, AWQ/BF16 could be faster
- **max_model_len=512**: Reduced from 1024; max prompt+response is ~350 tokens, safe margin
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

## RTX 5080 Benchmark Matrix (RunPod, $15 credits)

Run these in order. Each run: start fresh pod, run full 13435-request benchmark at concurrency 128.

| Run | Quantization | max_num_seqs | max_batched_tokens | Other Changes | Hypothesis |
|-----|-------------|-------------|-------------------|---------------|-----------|
| A | FP8 (current) | 256 | 8192 | baseline on 5080 | Establish 5080 baseline |
| B | None (BF16) | 160 | 8192 | QUANTIZATION="" | BF16 may beat FP8 on SM120 |
| C | FP8 | 256 | 16384 | larger batches | Better throughput for small model |
| D | Best of A/B/C | best | best | cudagraph_mode=FULL | Full CUDA graphs for small model |
| E | Best so far | best | best | all combined | Final best config |

All configs via env vars — no rebuild needed between runs.

## Next Steps

1. ~~Add Blackwell env vars to Dockerfile~~ ✅ Done
2. ~~Set swap_space=0~~ ✅ Done
3. Build & push Docker image to GHCR
4. Deploy on RunPod RTX 5080
5. Run benchmark matrix A→E
6. Pick best config, final Docker image push for submission
