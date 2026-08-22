# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing codebase: static HTML/CSS/JS single-page UI (`web/index.html`) served by a FastAPI backend (`app/main.py`). No frontend framework/build step.

## Users

Primary: hackathon judges/evaluators for HH Goa 2026 (Shortlisting Task 2), testing the live deployed demo for the first time during judging — they need to quickly understand what the system does, try it themselves (voice or text), and see evidence it's real (not a scripted demo).

Secondary: the team itself, using it to demo the project and in the required demo video.

## Product Purpose

A voice-enabled Retrieval-Augmented Generation (RAG) system: a user speaks (or types) a question, the system transcribes it (Sarvam STT), retrieves relevant passages from a real indexed multilingual dataset, and returns a grounded answer with citations — or explicitly declines to answer when the question is off-topic, unsafe, or not supported by the retrieved context. Built to satisfy an explicit hackathon brief requiring: real STT, vast/multi-strategy chunking, sub-200ms post-transcription latency, measured P50/P70/P100 latency analytics, a structured harness (not a raw prompt-in/text-out call), and guardrails that visibly demonstrate the system "knows when not to answer."

## Positioning

Unlike a typical hackathon RAG demo, this one is built and verified against the real dataset (`ai4bharat/MSMARCO-XI`, not placeholder/fake data), with guardrail correctness empirically measured (29/30, 96.7% on a mixed real-query/off-topic benchmark) and real, honestly-reported latency numbers (not a single best-case run) — including documenting failures found and fixed along the way (an off-topic query that used to hallucinate, a Devanagari tokenization bug, a corpus-size-vs-latency trade-off). The UI's job is to make that rigor visible and legible to a judge in the first few seconds, not just claim it.

## Operating Context

- A judge opens the live Cloud Run link, likely on a laptop, and has a few minutes.
- They may click "record" and speak a question, or type one, in English or an Indic language (Hindi, Bengali, Tamil, Urdu, Marathi are indexed).
- The response should visibly show: the transcript, the answer (or an honest abstention with a reason), citations/evidence chunks when grounded, and a stage-by-stage latency breakdown — because latency transparency and grounding transparency are graded requirements, not incidental features.
- Some judges will deliberately try to break it (off-topic questions, unsafe prompts, prompt injection) to test the guardrails — the UI should make a refusal/abstention read as a *feature working correctly*, not an error state.

## Capabilities and Constraints

- Voice input via browser MediaRecorder, or a text input fallback.
- Backend: FastAPI, `/api/ask-audio` and `/api/ask-text`.
- Real indexed data: 5,573 chunks across Hindi, Bengali, Tamil, Urdu, Marathi (ai4bharat/MSMARCO-XI validation split).
- Response payload includes: transcript, answer, citations (chunk_id, score, strategy, language, quote), grounded/abstained booleans + abstain_reason, and a `timings_ms` breakdown (input_guard_ms, retrieval_ms, retrieval_guard_ms, generation_ms, grounding_ms, post_stt_total_ms, total_ms, stt_ms when audio).
- Deployed on Google Cloud Run (scale-to-zero); cold start after idle can take a few seconds.
- No user accounts, no persistence of user queries beyond the single request/response.

## Brand Commitments

- Name: "HH Goa 2026 — Voice-Enabled Grounded RAG" (from the hackathon task). No pre-existing logo or brand assets.
- Visual world: a bold flat-illustration "Goa beach" travel-poster aesthetic (deep emerald green, sun-gold yellow, white/sand, hot-pink accent), per a reference image supplied by the user — full visual identity, not a subtle accent (confirmed).

## Evidence on Hand

- Real benchmark results: `storage/benchmark_results.json`, `storage/benchmark_queries.json`.
- Real latency numbers documented in `LATENCY.md` / `README.md` (P50 114.29ms / P70 122.69ms / P100 174.54ms, post-STT).
- Real correctness rate: 29/30 (96.7%).
- Public GitHub repo: https://github.com/Ganesh-0509/hh-goa-2026-voice-rag
- Live deployed demo: https://hh-goa-voice-rag-686797972138.us-central1.run.app
- Full build/debug narrative in `PROJECT_LOG.md`.
- No customer testimonials, press, or third-party proof exist or should be implied — this is a hackathon submission, not a commercial product.

## Product Principles

1. Show the work: latency numbers, citations, and abstain reasons are the credibility mechanism — never hide or minimize them for the sake of a cleaner-looking screen.
2. A refusal is a success state. Off-topic/unsafe/ungrounded abstentions should read visually as "guardrail working," not as an error or dead end.
3. Real data, real numbers, no fabrication — the UI must never imply capability, coverage, or results the system doesn't actually have.
4. Fast to understand in the first 10 seconds: a judge should immediately grasp what this is and how to try it, without reading documentation.
5. Bold, memorable, on-brand — this is a hackathon; safe/generic SaaS-dashboard styling undersells genuinely rigorous work.

## Accessibility & Inclusion

No formally confirmed standard. Given multilingual (Indic-script) content is core to the product, text rendering must support Devanagari/Bengali/Tamil/Urdu/Marathi scripts correctly, and color contrast should hold up against the deep-green/gold palette (verify at build time).
