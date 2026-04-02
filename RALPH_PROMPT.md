# Optimization Loop: Track 2 Chat Engine

You are optimizing a vLLM-based LLM serving engine for maximum throughput on Qwen3-4B-Instruct-2507.

## Context
- GPU: A100 80GB PCIe (target: RTX 5080 16GB — optimize for both)
- Benchmark: 13,435 customer service queries, 128 concurrency, temperature=0, max_tokens=256
- Metrics: throughput (req/s), P50 latency, P99 latency, perplexity
- Code: `/home/a/arya/mlsys/track2_chat/app/` (main.py, chat_engine.py, cache.py, constants.py, normalize.py)
- Plan: `/home/a/arya/mlsys/plan.md` — READ THIS FIRST every iteration
- Time constraint: SLURM job with limited GPU time. Be fast and decisive.

## Each Iteration

1. **Read `plan.md`** — understand what's been tried, what worked, what to try next
2. **Pick ONE optimization** from the Next Steps or your own idea
3. **Implement it** — modify code in `track2_chat/app/`
4. **Start server**: `cd /home/a/arya/mlsys/track2_chat && QUANTIZATION="" VLLM_NO_USAGE_STATS=1 CUDA_MODULE_LOADING=LAZY uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log --loop uvloop`
5. **Wait for ready**: poll `curl http://localhost:8000/ready` every 5s
6. **Run benchmark**: `cd /home/a/arya/mlsys/benchmark && uv run runner_chat.py --url http://localhost:8000 --data data/track2/train.jsonl --concurrency 128 --timeout 120`
7. **Log results** to `plan.md` experiment log immediately
8. **Kill server**: `pkill -9 -f uvicorn; sleep 3`
9. **Analyze**: Did it improve? Update plan.md with verdict and next steps

## Rules

- NEVER sacrifice perplexity for throughput — quality comes first
- ONE change per iteration — isolate variables
- If a change HURTS performance, revert it immediately
- Always kill the server before starting a new one
- Config via env vars when possible (no rebuild needed)
- Log EVERY result to plan.md — no exceptions
- Check SLURM time remaining: `squeue -u $USER` — stop if <10 min left

## Key Facts

- Decode is the bottleneck (99% of latency) — memory bandwidth bound
- Spec decode HURTS at 128 concurrency (proven by papers) — keep disabled
- FP8 broken on SM120 in vLLM 0.8.5.post1 — use BF16 for now
- 36% exact-match cache hit rate, potentially 77% with semantic caching
- Semantic cache (sentence-transformers + FAISS) is already implemented but UNTESTED
- V0+multi-step gave +4% over V1 on A100
- Inflight dedup barely fires (duplicates are 7,858 positions apart)

## Ideas to Explore (prioritized)

1. Test semantic cache as-is — does it improve throughput without hurting perplexity?
2. Tune semantic cache thresholds (keyword_threshold, semantic_threshold, hard_stop)
3. Try V0 with num_scheduler_steps=10 combined with semantic cache
4. Increase max_num_batched_tokens to 16384 or 32768
5. Pre-warm semantic cache during startup with diverse customer service queries
6. Profile actual cache hit rates during benchmark (add logging)

## Completion

When throughput exceeds 50 req/s with perplexity < 1.25, output:
<promise>OPTIMIZATION COMPLETE</promise>

If SLURM time is running out (<10 min), commit current best and output the promise.
