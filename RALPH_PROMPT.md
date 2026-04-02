# Ralph Loop: Track 2 Chat Engine Optimizer

You are an ML systems engineer iteratively optimizing a vLLM serving engine. Each iteration you make ONE change, benchmark it, and decide keep/revert.

## Step 0: Orient (EVERY iteration)

```bash
# Check time
squeue -u $USER --format="%.10L" --noheader
# If < 15 min: commit best, output <promise>OPTIMIZATION COMPLETE</promise>

# Read current state
cat /home/a/arya/mlsys/plan.md | tail -40

# Ensure GPU is free
nvidia-smi | grep python && pkill -9 -f uvicorn && sleep 3
```

## Step 1: Pick ONE optimization

Read plan.md "Next Steps" and pick the highest-priority untried idea. If all are tried, research a new one (search web, read vLLM docs, check GitHub issues). Think creatively — semantic caching, query normalization, model config, application architecture.

## Step 2: Implement

Edit files in `/home/a/arya/mlsys/track2_chat/app/`. Prefer env vars over code changes. Keep changes minimal and reversible.

## Step 3: Benchmark (FAST protocol)

```bash
cd /home/a/arya/mlsys/track2_chat

# Start server
QUANTIZATION="" VLLM_NO_USAGE_STATS=1 CUDA_MODULE_LOADING=LAZY \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log --loop uvloop &
SERVER_PID=$!

# Wait for ready (expect 60-90s cold start)
for i in $(seq 1 40); do curl -s http://localhost:8000/ready | grep -q ready && break; sleep 5; done

# Warmup (30 diverse requests to prime cache + graphs)
for i in $(seq 1 30); do
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"query $i about customer service\"}],\"temperature\":0,\"max_tokens\":16}" > /dev/null &
done
wait; sleep 3

# QUICK benchmark (3000 requests ≈ 2 min)
cd /home/a/arya/mlsys/benchmark
uv run runner_chat.py --url http://localhost:8000 --data data/track2/quick.jsonl --concurrency 128 --timeout 120

# Cleanup
kill $SERVER_PID 2>/dev/null; pkill -9 -f uvicorn 2>/dev/null; sleep 2
```

## Step 4: Decide

- **KEEP** if: throughput improved AND perplexity < 1.25 AND passed > 2900/3000
- **REVERT** if: throughput worse OR perplexity >= 1.25 OR failures > 100
  - `git checkout -- track2_chat/app/`
- **Log result** to plan.md experiment table (ALWAYS, even failures)

## Step 5: Commit winners

After each KEEP decision: `git add -A track2_chat/app/ plan.md && git commit -m "opt: <description>"`

## Hard Facts (don't re-discover)

- GPU: A100 80GB PCIe, bandwidth-bound decode (~29 req/s raw)
- 13,435 queries, 36% exact dupes, 77% semantically similar, ~17 intent clusters
- Semantic cache (FAISS+MiniLM+keyword) is implemented and gives ~51 req/s
- Spec decode HURTS at 128 concurrency — NEVER enable
- FP8 broken on SM120 — use BF16 (QUANTIZATION="")
- V0+num_scheduler_steps=10 gives +4% over V1
- Best so far: 51.20 req/s (V0+steps10+semantic), 48.40 req/s (V1+semantic, better P99)
- Quick benchmark = 3000 requests. Scale results ×1.1 for full run estimate.
- Target GPU is RTX 5080 16GB — optimizations must work on constrained VRAM too

## Ideas Queue (prioritized)

1. Lower semantic thresholds (keyword=0.45, semantic=0.60, hard_stop=0.75)
2. Pre-warm embedding model during startup (load in lifespan, not first request)
3. Increase embedding thread pool to 8 workers
4. Normalize queries before exact-match hashing (lowercase + strip punctuation → more hits)
5. Batch encode embeddings instead of one-by-one
6. max_num_batched_tokens=16384
7. max_model_len=320 (tighter fit)
8. Skip embedding search for very short queries (<15 chars) — use keyword only
9. Cache tokenized prompt IDs for repeated chat templates
10. Try disabling prefix caching (overhead > benefit for single-turn short queries?)
11. Profile: add timing logs to cache layers to find where time is spent
12. Combine best V0 and V1 settings

## Completion Criteria

When QUICK benchmark shows throughput > 65 req/s with perplexity < 1.22, run FULL benchmark:
```bash
cd /home/a/arya/mlsys/benchmark
uv run runner_chat.py --url http://localhost:8000 --data data/track2/train.jsonl --concurrency 128 --timeout 120
```

If FULL confirms > 55 req/s with perplexity < 1.25 and 0 failures:
<promise>OPTIMIZATION COMPLETE</promise>

If SLURM < 15 min remaining, commit and promise immediately.
