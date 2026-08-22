# Guardrails & Safety System Specification

Our system enforces multi-layered guardrails at input ingestion, post-retrieval context analysis, and post-generation answer verification.

---

## 1. Multi-Layer Guardrail Pipeline

```txt
Query Input
    │
    ▼
┌─────────────────────────┐
│     1. Input Guard      │ ──► [Refuse] Unsafe queries (weapons, self-harm, etc.)
└───────────┬─────────────┘ ──► [Refuse] Prompt injection attacks & override prompts
            │
            ▼
┌─────────────────────────┐
│  2. Hybrid Retrieval    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   3. Retrieval Guard    │ ──► [Abstain] Top-hit margin < MIN_DENSE_MARGIN (Off-topic)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   4. Answer Engine      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   5. Grounding Guard    │ ──► [Abstain] Answer token context support < 40%
└───────────┬─────────────┘
            │
            ▼
Verified Grounded Response
```

---

## 2. Rule Definitions & Abstain Logic

1. **Safety Refusal**: Block queries matching safety violations (bomb making, self harm, violence, credit card dumps).
2. **Prompt Injection Resistance**: Reject queries attempting system prompt extraction (`"reveal your system prompt"`), rule overrides (`"ignore previous instructions"`), or jailbreak attempts.
3. **Retrieval Confidence Guard (margin-based)**: Abstain with a friendly explanation (`"I could not find enough relevant context..."`) unless the dense retrieval hit has both (a) a top score above a loose absolute backstop (`MIN_DENSE_SCORE=0.50`) and (b) a meaningful **margin** over the mean of its tail candidates (`MIN_DENSE_MARGIN=0.055`) — see `HybridRetriever.confidence_from_dense_hits` in `app/retriever.py`. This went through two iterations:
   - v1 used a fixed absolute cosine threshold with a lexical-match bypass. Verified live that this failed: any query got *some* BM25 hit via OR-joined terms, which bypassed a low dense score entirely — "Who won the World Cup in 2022?" returned a confident, ungrounded answer.
   - v2 dropped the bypass and used dense score alone against a fixed threshold (0.80), tuned against a 20-chunk placeholder corpus. This broke once the real ~4,751-chunk MSMARCO-XI index was built: e5-small's "noise floor" score for unrelated queries climbs as the corpus grows (0.70-0.74 on 20 chunks vs 0.75-0.84 on 4,751 chunks), so a threshold tuned on one corpus size silently failed on another — verified live with garbage matches like "How to prevent hallucination in RAG systems?" returning an unrelated passage about Air Force safety equipment.
   - **Current (margin-based)**: instead of an absolute score, requires the top hit to stand out meaningfully above the general noise floor for that query (top score minus the mean of rank 10-40 candidates). This generalizes across corpus size since it's relative, not absolute. Calibrated against 2 known-relevant queries pulled from the indexed corpus (margin 0.076, 0.112) vs. 4 known off-topic queries (margin 0.018-0.040) — threshold set at 0.055, roughly the midpoint. Re-validated on the real corpus: 29/30 benchmark queries (15 relevant + 15 off-topic/unsafe/injection) got the correct abstain/answer decision.
4. **Grounding & Hallucination Check**: Validate that at least 40% of non-stopword tokens in the synthesized answer exist in the retrieved evidence chunks. If validation fails, trigger abstention. Note: token matching uses Unicode-property-aware tokenization (`\p{L}\p{M}\p{N}` via the `regex` package), not stdlib `re`'s `\w+` — see item 5.
5. **Indic-script tokenization fix**: stdlib `re`'s Unicode `\w` excludes combining marks (Mn/Mc Unicode categories), so it shredded Devanagari conjuncts into single-character fragments (e.g. "कॉर्पोरेशन" → `['क','र','प','र','शन',...]`) wherever term-overlap scoring or lexical query construction ran on Hindi text — i.e. the entire non-English MSMARCO-XI corpus. This corrupted both the extractive generator's sentence-selection scoring and the SQLite FTS5 lexical query (SQLite's own `unicode61` tokenizer was unaffected, so the index itself was fine — only query-side tokenization in `app/generator.py` and `app/retriever.py` was broken). Fixed by switching to the `regex` package's `\p{L}\p{M}\p{N}` pattern, which keeps letter+mark+digit runs together as one token.
