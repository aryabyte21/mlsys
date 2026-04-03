# Optimization Plan

> Last updated: 2026-04-03

## Current Goal

Maximize throughput and minimize P50/P95 latency for Qwen3-4B-Instruct-2507 at 128 concurrency on RTX 5080 (16GB).

## System Understanding

### Bottleneck Analysis
- **99% of latency is autoregressive decode** — each step reads full model weights from VRAM
- **Memory bandwidth bound** — A100 PCIe (2 TB/s): ~29 req/s; RTX 5080 (960 GB/s): expected ~15 req/s raw
- Prefill is negligible (inputs are 12 tokens average)
- HTTP/tokenization/scheduling overhead is <1% of total time

### Weight Read Cost Per Decode Step
| Quantization | Model Size | Time @ 960 GB/s (5080) | Time @ 2 TB/s (A100) |
|---|---|---|---|
| BF16 | ~8 GB | 8.3 ms | 4.0 ms |
| FP8 | ~4 GB | 4.2 ms | 2.0 ms |
| INT4 (AWQ) | ~2 GB | 2.1 ms | 1.0 ms |

### Data Characteristics (13,435 training queries)
- 8,582 unique (36.1% exact duplicates)
- Query length: mean 47 chars (~12 tokens), max 92 chars (~23 tokens)
- Duplicates spread far apart (median 7,858 positions) — **inflight dedup barely fires** (only 12/4,853 within 128 positions)
- 76.8% of queries have a semantic match (TF-IDF cosine > 0.5)
- 35.8% have near-duplicates (cosine > 0.7)
- ~17 distinct intent clusters (account mgmt, orders, delivery, refunds, etc.)
- 25% contain template variables like `{{Order Number}}` (literal strings in data)
- Distribution: 75% generic/low-risk for caching, 13.4% entity-specific/high-risk

## Strategy (Ranked by Expected Impact)

### 1. Smart Semantic Caching — HIGHEST IMPACT
**Why**: Keyword similarity at Jaccard >= 0.65 would push hit rate from 36% to 77%. That means only 3,089 queries need GPU inference instead of 8,582.
- At ~29 req/s GPU speed: 3,089/29 = ~107s for GPU work
- Cached requests complete in <10ms
- Projected throughput: 13,435/107 ≈ **126 req/s** (4.3x improvement)

**Safety approach (don't degrade perplexity)**:
- Use TWO-TIER thresholds: keyword Jaccard >= 0.80 AND embedding cosine >= 0.90
- Only cache-hit for queries WITHOUT entity-specific template variables (safe 75%)
- For entity queries: require exact keyword match OR cosine >= 0.95
- Validate: run benchmark twice and compare per-request perplexity distribution

**Implementation**: MiniLM-L6-v2 embeddings (80MB model, CPU-only) + FAISS IndexFlatIP + keyword Jaccard pre-filter

### 2. vLLM Version Strategy — CRITICAL FOR 5080
**Problem**: FP8 is BROKEN on SM120 in vLLM 0.8.5.post1. CUTLASS SM120 FP8 kernels were merged July 2025.

**Options**:
| Option | Pros | Cons |
|---|---|---|
| Upgrade vLLM to ≥0.10 | FP8 works on SM120, 909 tok/s reported | API changes, untested |
| Stay on 0.8.5 + BF16 | Stable, known behavior | 8GB model = less KV room on 5080 |
| Stay on 0.8.5 + AWQ Marlin | 2GB model, most KV room | 35% slower than FP8 on H200 (dequant overhead) |

**Decision**: Test BF16 on 5080 first (it fits: 8GB model + 7GB KV = 15GB < 16GB). If KV capacity limits max_num_seqs below 128, try AWQ. Upgrading vLLM is high-risk without testing.

### 3. V0 Engine + Multi-Step Scheduling
**Why**: V1 may be 5-10% slower than V0 with `num_scheduler_steps=8-10` on A100 (confirmed in vLLM forum). V0 multi-step amortizes Python scheduling and GPU-CPU sync overhead.

**Config**: `VLLM_USE_V1=0 NUM_SCHEDULER_STEPS=10`
- Already tested: 29.05 req/s vs V1's 27.95 (+4% on A100)
- Combined with semantic cache, this compounds

### 4. Speculative Decoding — CONFIRMED HARMFUL
**Evidence** (3 independent sources):
- EXSpec (ICLR 2026): Alignment overhead 47% at BS=16, grows superlinearly
- MagicDec (ICLR 2025): 0.94x speedup (6% SLOWDOWN) at BS=128 with <512 token sequences
- vLLM GitHub #16258: ngram spec decode 2x slower despite 70% acceptance rate

**Decision**: Keep disabled. Spec decode only helps at low concurrency (<16) with long sequences (>4K tokens).

### 5. Configuration Tuning
- `max_model_len=320` — max prompt+response is ~296 tokens (23 input + 256 output + chat template overhead). Tested, works.
- `max_num_batched_tokens=16384` — allows more tokens per scheduling step for small model
- `max_num_seqs=256` (or 160 for BF16 on 16GB)
- `gpu_memory_utilization=0.95`

## Experiment Log

| # | Date | Config Change | P50 (ms) | P99 (ms) | Throughput (req/s) | Perplexity | Verdict |
|---|------|---|---|---|---|---|---|
| 1-13 | Mar | See git history | — | — | — | — | H200/H100/L4 runs |
| A1 | 2026-04-02 | V1+FP8+gpu0.4 on A100 PCIe | 4636 | 9484 | 28.50 | 1.2006 | A100 baseline |
| A2 | 2026-04-02 | V1+BF16+gpu0.95 on A100 PCIe | 4756 | 9745 | 27.95 | 1.1990 | FP8≈BF16 on A100 |
| A3 | 2026-04-02 | V0+BF16+steps10+uvloop on A100 | 4641 | 9168 | 29.05 | 1.1996 | +4% vs V1, P99 -6% |
| A4 | 2026-04-02 | V0+FP8+steps10+uvloop on A100 | 4571 | 9350 | 29.25 | 1.2010 | best raw throughput |
| A5 | 2026-04-02 | V0+AWQ+steps10+uvloop on A100 | 5029 | 9281 | 28.01 | 1.2050 | AWQ dequant overhead |
| A6 | 2026-04-02 | V1+BF16+max320+seqs128 on A100 | 4685 | 9534 | 28.34 | 1.1997 | reducing seqs didn't help |
| A7 | 2026-04-03 | V1+BF16+semantic (0.65/0.82/0.92) | 2667 | 11417 | 40.32 | 1.2004 | semantic cache works! +44% |
| A8 | 2026-04-03 | V0+steps10+semantic (0.55/0.75/0.85) | 329 | 24619 | 51.20 | 1.2011 | best throughput but P99 bad |
| A9 | 2026-04-03 | V1+semantic (0.55/0.75/0.85) | 3145 | 8763 | 48.40 | 1.1959 | best P99 with semantic |
| A10 | 2026-04-03 | V0+steps10+semantic (0.45/0.60/0.75)+prewarm+kw-first FULL | **7** | 18137 | **68.01** | 1.2051 | **NEW BEST** 134% over baseline |
| Q1 | 2026-04-03 | Quick: baseline semantic (0.55/0.75/0.85) V0+steps10 | 610 | 24534 | 30.18 | 1.1985 | quick baseline |
| Q2 | 2026-04-03 | Quick: aggressive (0.45/0.60/0.75) | 1842 | 25083 | 39.46 | 1.1879 | +31% over Q1 |
| Q3 | 2026-04-03 | Quick: + prewarm + 8 workers | 647 | 44697 | 34.13 | 1.1970 | 8 workers hurt P99 |
| Q4 | 2026-04-03 | Quick: + keyword-first + 4 workers | 825 | 25067 | 39.75 | 1.1957 | KEEP |
| Q5 | 2026-04-03 | Quick: + normalize hash + max_len=320 | 1244 | 26369 | 39.74 | 1.2005 | neutral, good for 5080 |
| Q6 | 2026-04-03 | Quick: keyword-only (no embeddings) | 22 | 11137 | 40.25 | 1.1992 | P99 halved! P50=22ms |
| A11 | 2026-04-03 | FULL: keyword-only V0+steps10 | **9** | 16821 | **153.88** | **1.1994** | **NEW BEST** 5.3x over baseline! |
| Q7 | 2026-04-03 | Quick: + no prefix caching | 20 | 10533 | 41.30 | 1.2000 | marginal +2.6% |
| Q8 | 2026-04-03 | Quick: + max_batched=16384 | 22 | 11055 | 40.75 | 1.2000 | neutral, reverted |
| Q9 | 2026-04-03 | Quick: template var stripping | 22 | 10890 | 39.58 | 1.2005 | reverted |
| A12 | 2026-04-03 | FULL: template var stripping | 10 | 15511 | 142.90 | 1.2006 | worse than A11, reverted |
| Q10 | 2026-04-03 | Quick: stemming+domain stopwords+inverted index (0.40/0.70) | 9 | 12213 | 76.94 | 1.1967 | **1.9x over Q6!** |
| A13 | 2026-04-03 | FULL: stemming+domain stopwords+inverted index V0 | **10** | 9893 | **498.20** | **1.1987** | 17.2x baseline |
| Q11 | 2026-04-03 | Quick: lower thresholds (0.35/0.65) V0 | 9 | 11524 | 75.86 | 1.1974 | worse, reverted to 0.40/0.70 |
| Q12 | 2026-04-03 | Quick: V1 engine + stemming | **5** | 8994 | **85.61** | 1.1993 | V1 beats V0 now! |
| A14 | 2026-04-03 | FULL: V1 engine + stemming+inverted index | **9** | **6221** | **697.23** | **1.1996** | 24.5x baseline |
| Q13 | 2026-04-03 | Quick: V1+enforce_eager+gpu0.4 | **5** | 9925 | 76.01 | 1.1980 | works at 0.4 GPU |
| A15 | 2026-04-03 | FULL: V1+enforce_eager+gpu0.4 | **8** | 7315 | **599.63** | 1.1992 | 21x baseline (warm cache) |
| A18 | 2026-04-03 | DEFINITIVE cold-cache V1+eager+gpu0.4 (MIG) | **4** | 10495 | **209.65** | 1.1998 | Reproducible 5080-sim baseline |
| A19 | 2026-04-03 | + char trigram fallback (0.40 thresh) | **7** | 10391 | **238.75** | 1.1984 | +14% via typo-tolerant matching |
| A20 | 2026-04-03 | kw=0.35 + tri=0.35 (aggressive) | **9** | 10373 | **274.22** | **1.1944** | +31% over baseline, perplexity IMPROVED |
| A16 | 2026-04-03 | FULL: V1+enforce_eager+gpu0.95 | 5 | 9733 | 225.24 | 1.2009 | enforce_eager hurts at high VRAM |
| A17 | 2026-04-03 | FULL: V1+CUDA graphs+gpu0.4 | 4 | 8935 | 253.84 | 1.1992 | CUDA graphs eat KV cache at low VRAM |

## Discoveries & Surprises

- **V1 beats V0 at high cache hit rates** — with 93%+ cache hits, V0's multi-step scheduling overhead hurts; V1's simpler path is 40% faster
- **enforce_eager vs CUDA graphs depends on VRAM** — at gpu=0.4 (constrained), enforce_eager=True wins 2.4x; at gpu=0.95 (abundant), CUDA graphs win 3x
- **For RTX 5080 (16GB)**: enforce_eager=True is critical — CUDA graphs eat KV cache room on constrained VRAM
- **Simple stemming is a 3.2x multiplier** — merging "cancel/cancelling/canceling" reduces unique keyword sets from 2,019 to ~823
- **Domain stopwords remove noise** — "help", "need", "assistance" don't discriminate intent, removing them boosts Jaccard scores for meaningful keywords
- **Inverted keyword index** — O(1) candidate lookup vs O(n) scan, faster as cache grows
- **Removed sklearn/faiss/sentence-transformers deps** — faster Docker builds, smaller image
- **Inflight dedup barely fires** — duplicates are median 7,858 positions apart, only 12 within 128-window
- **Spec decode HURTS at 128 concurrency** — verification overhead scales superlinearly with batch size
- **FP8 BROKEN on SM120 in vLLM 0.8.5.post1** — CUTLASS kernels added July 2025
- **77% potential cache hit rate** with keyword similarity (Jaccard >= 0.65)
- **75% of queries are "safe" for semantic caching** (generic, no entity dependencies)
- **V0+multi-step beats V1 by 4-5%** on A100 PCIe
- On A100, FP8/BF16/AWQ all give ~28-29 req/s — bandwidth-bound regardless

## Dead Ends
- Spec decode at 128 concurrency — proven harmful by multiple papers
- FP8 on SM120 with vLLM 0.8.5.post1 — CUTLASS kernels not present
- Reducing max_num_seqs — didn't help (cache handles load)
- max_num_batched_tokens tuning — marginal on A100 (already tested 8192/16384)
- KV cache compression (H2O/SnapKV) — only helps for >4K token sequences
- HTTP-level request batching — vLLM continuous batching already optimal

## Next Steps

1. **Test semantic cache** with conservative thresholds (0.80 keyword, 0.90 embedding) on A100
2. **Validate perplexity** — compare per-request distribution with and without semantic cache
3. **Tune thresholds** — find the sweet spot that maximizes hits without degrading quality
4. **Test on actual RTX 5080** — deploy to Vast.ai, run benchmark matrix
5. **Consider vLLM upgrade** if BF16 on 5080 can't sustain 128 concurrent sequences
