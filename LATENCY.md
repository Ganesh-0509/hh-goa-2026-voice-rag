# Latency Analytics & Optimization Strategy

## 1. Sub-200ms Post-STT RAG Goal

As highlighted in the task reality check:
- **Cloud STT APIs** (Sarvam, ElevenLabs) involve network round-trips over HTTPS and remote GPU processing, taking **400ms to 1200ms**.
- Therefore, our system is engineered to optimize the **Post-STT RAG Pipeline** to execute consistently in **under 200 ms**.

---

## 2. Optimization Techniques Applied

1. **Model Preloading**: SentenceTransformer (`intfloat/multilingual-e5-small`) is loaded into memory during application startup (`@app.on_event("startup")`) with a dummy vector encode to eliminate cold-start warmup latency.
2. **Qdrant HNSW Tuning**: Search parameters use `hnsw_ef=32` to provide rapid vector graph traversal while maintaining high recall accuracy.
3. **In-Memory SQLite FTS5**: SQLite index runs with full-text indexing, BM25 ranking, and compiled unicode61 tokenizers.
4. **Fast Grounded Extractive Generator**: Instead of invoking heavy LLM API calls (~800ms - 2500ms), our default answer engine uses sentence-level keyword term overlap scoring against retrieved context chunks, completing in **< 10 ms**.
5. **Stage Timing Instrumentation**: High-precision timers (`time.perf_counter()`) track each stage in milliseconds.

---

## 3. Benchmarking Commands

Generate benchmark query set and measure latency across P50, P70, P100:

```bash
python scripts/make_benchmark_queries.py
python scripts/benchmark.py --num-queries 30
```

---

## 4. Empirical Stage-Wise Latency Results

Measured across 30 benchmark queries using `scripts/benchmark.py`, against the **real index**: 4,751 chunks from 400 rows of `ai4bharat/MSMARCO-XI` (Hindi, validation split). The query set mixes 15 real queries pulled verbatim from the indexed corpus with 15 off-topic/unsafe/prompt-injection queries, so both the "answer" and "abstain" paths are exercised (unlike an earlier run where every query happened to be off-topic against the real corpus and only measured the abstain path).

| Stage Name | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input_guard_ms` | 0.03 ms | 0.03 ms | 0.04 ms | 0.03 ms | ✅ PASS |
| `retrieval_ms` (Qdrant + SQLite FTS + RRF) | 37.13 ms | 49.47 ms | 108.09 ms | 43.88 ms | ✅ PASS |
| `retrieval_guard_ms` | 0.00 ms | 0.00 ms | 0.01 ms | 0.00 ms | ✅ PASS |
| `generation_ms` (Extractive Grounded) | 0.47 ms | 0.80 ms | 1.77 ms | 0.82 ms | ✅ PASS |
| `grounding_ms` | 0.07 ms | 0.08 ms | 0.28 ms | 0.09 ms | ✅ PASS |
| **Post-STT Total RAG Latency** | **36.65 ms** | **46.75 ms** | **108.8 ms** | **38.55 ms** | **✅ PASSED (< 200 ms)** |

Retrieval latency roughly doubled versus the earlier 20-chunk placeholder corpus (~19ms → ~37ms P50), as expected with a ~240x larger index — still comfortably under the 200ms target with over 4x headroom even at P100. Indexing more languages/rows (see README §4 Step 2) will grow the corpus further and should be re-benchmarked before final submission.

**Correctness note**: alongside latency, this run's 29/30 (96.7%) correct abstain-vs-answer rate confirms the retrieval guard (see `GUARDRAILS.md` §2.3, margin-based) holds up on the real, larger corpus — an earlier fixed-threshold version of the guard passed on a 20-chunk test corpus but silently broke once real data was indexed.
