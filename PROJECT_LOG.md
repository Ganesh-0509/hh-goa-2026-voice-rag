# Project Log — Voice-Enabled RAG on ai4bharat/MSMARCO-XI

This document exists so that anyone joining this project — a teammate, a judge, or
future-us in six months — can understand the whole story without having sat through
the build: what the task required, what we built, what broke, how we found it, how
we fixed it, and where things stand right now. It's written chronologically, in the
order things actually happened, including the mistakes — because the mistakes and
how we caught them are as much a part of demonstrating a working guardrail-aware
RAG system as the final numbers are.

---

## 1. The task

**HH Goa 2026, Shortlisting Task 2: Voice-Enabled RAG.** Build a system where a
user speaks a question, the pipeline transcribes it, retrieves relevant context
from a provided dataset, and returns an answer — end to end.

Hard requirements:
- **Speech-to-text**: Sarvam or ElevenLabs, pick one. We picked **Sarvam** — its
  free tier (~₹100 credit, ~60+ minutes of audio, no card required) is far more
  usable for a hackathon build than ElevenLabs' API free tier (10 credits/month,
  barely covers a couple of minutes), and it fits naturally with the dataset since
  both are part of the AI4Bharat / Indian-language ecosystem.
- **Dataset**: `ai4bharat/MSMARCO-XI` from Hugging Face.
- **Chunking**: must be "vast" — multiple real strategies (semantic, fixed-size,
  overlap, metadata-aware), not one naive fixed-size splitter.
- **Latency**: full pipeline (chunking + vector DB retrieval + everything through
  to final output) under 200ms, excluding the unavoidable cloud STT round trip.
- **Latency analytics**: P50 / P70 / P100 over a reasonable number of real test
  queries, not one cherry-picked run.
- **Harness**: structured orchestration (retries, structured I/O, error recovery),
  not a single raw prompt-in/text-out call.
- **Guardrails**: off-topic queries, unsafe input, hallucination checks, "knows
  when not to answer."

---

## 2. Architecture

```
User Voice
   |
   v
Browser MediaRecorder UI (web/index.html)
   |
   v
FastAPI  POST /api/ask-audio   (app/main.py)
   |
   v
Sarvam Speech-to-Text          (app/stt_sarvam.py)
   |
   v
Input Guardrails               (app/guardrails.py: input_guard)
   |  - unsafe-content patterns
   |  - prompt-injection patterns
   |  - empty/oversized transcript
   v
Hybrid Retriever                (app/retriever.py)
   |--- Qdrant dense search (multilingual-e5-small embeddings)
   |--- SQLite FTS5 lexical search (BM25)
   |--- Reciprocal Rank Fusion + diversity filter
   v
Retrieval Confidence Guard      (app/guardrails.py: retrieval_guard)
   |  - margin-based off-topic detection (see §4.6)
   v
Extractive Answer Generator     (app/generator.py)
   |  - term-overlap sentence selection from retrieved chunks only
   v
Grounding / Hallucination Guard (app/guardrails.py: grounding_check)
   |
   v
Structured JSON response: answer, citations, grounded, abstained, timings_ms
```

Everything is orchestrated by `VoiceRAGHarness` (`app/harness.py`), which is the
"harness" the task requires: it owns retries on the STT call, stage-by-stage
timing instrumentation, and structured fallback responses (never a raw crash —
every failure mode returns a proper `RagResponse` with an `abstain_reason`).

**Why extractive generation, not an LLM call?** The 200ms budget is for
everything *after* STT. An LLM call would blow that budget by itself (the
generator's own docs estimate 800ms-2500ms for a typical LLM API round trip).
Extractive generation — picking the best-matching sentences straight out of the
retrieved chunks — runs in under 2ms and is *trivially* grounded, since it can
only ever say things that are literally present in the retrieved text. The
trade-off is answer fluency, which is a reasonable one to make for this task's
latency target.

---

## 3. Repository layout

```
app/
  main.py            FastAPI app: /health, /api/ask-text, /api/ask-audio
  config.py          Settings (env-driven)
  schemas.py         Pydantic request/response models
  harness.py         VoiceRAGHarness — orchestrates the whole pipeline
  stt_sarvam.py       Sarvam STT client (retries, dev fallback if no key)
  chunking.py        Multi-strategy adaptive chunker
  dataset_loader.py  Parses raw MSMARCO-XI rows into RawDoc objects
  retriever.py       Hybrid dense+lexical retrieval, RRF fusion, confidence signal
  generator.py       Extractive grounded answer generation
  guardrails.py      Input guard, retrieval guard, grounding check
  latency.py         Stage-timing context manager
scripts/
  explore_dataset.py     Inspect the real dataset schema
  build_index.py         Index MSMARCO-XI into Qdrant + SQLite FTS5
  make_benchmark_queries.py  Generates storage/benchmark_queries.json
  benchmark.py            Runs the pipeline over the query set, computes P50/P70/P100
web/index.html        Minimal MediaRecorder UI
storage/               chunks.sqlite, qdrant_db/ (both git-ignored), benchmark JSON (tracked)
README.md / CHUNKING.md / GUARDRAILS.md / LATENCY.md   Design docs, kept current
```

---

## 4. The build journey

### 4.1 Starting point

The project arrived at this review already scaffolded from a detailed blueprint:
FastAPI backend, Sarvam adapter, multi-strategy chunker, hybrid retriever with
RRF fusion, extractive generator, guardrail module, benchmarking script — all the
pieces the task asks for, already wired together. The job from here was to verify
it actually worked, not just that it existed.

### 4.2 First review: three real problems, found by actually running it

Reading the code wasn't enough — running it live turned up issues the code
*looked* fine without:

1. **The dataset was fake.** `storage/chunks.sqlite` held only 20 chunks, all
   hand-written trivia about Panaji, RAG, and Qdrant — not `ai4bharat/MSMARCO-XI`
   at all. `build_index.py`'s attempt to stream the real dataset was silently
   falling back to 10 hardcoded sample rows.
2. **Sarvam wasn't wired up.** `SARVAM_API_KEY` in `.env` was still the
   placeholder string, so `stt_sarvam.py` was silently returning a hardcoded fake
   transcript ("What is the capital of Goa?") instead of ever calling the API.
3. **The retrieval guard didn't actually guard anything.** Tested live with
   *"Who won the World Cup in 2022?"* and *"What is the weather on Mars
   today?"* — both got confident, fully-grounded-looking answers built from
   unrelated Panaji content. This directly contradicts the task's guardrail
   requirement. Root cause: the guard only refused when the dense similarity
   score was low **and** there was no lexical (BM25) hit — but the lexical
   query OR-joined every query term, so almost any English sentence produced
   *some* match, which alone was enough to bypass the dense-score check
   entirely.

### 4.3 First fix: the retrieval guard, round one

Measured real e5 cosine similarity scores on the (still fake, 20-chunk) corpus:
a genuinely matching query scored ~0.86, an unrelated one scored ~0.70-0.74.
Fixed the guard to require the dense score alone to clear a threshold (dropped
the lexical bypass entirely), and raised `MIN_DENSE_SCORE` from 0.35 to 0.80
based on that measurement. Verified live: all four guardrail test categories
(unsafe, prompt-injection, off-topic, sports) now correctly abstained, while
on-topic queries still answered. This held up — *for now* — but the fix was
built on a 20-chunk corpus, which turned out to matter later (§4.7).

### 4.4 Wiring up Sarvam for real

Got a real Sarvam API key, dropped it into `.env`, and verified it wasn't just
present but actually *working*: sent a real audio file (a synthetic sine tone,
since no real speech was available yet) through `SarvamSTT.transcribe()` and got
a genuine `200 OK` back (empty transcript, as expected for a tone — the point was
confirming the API key, endpoint, and multipart upload format were all correct,
not that a tone transcribes to words). Then ran it through the full
`ask_audio()` harness path end-to-end and confirmed the input guard correctly
abstained on the resulting empty transcript, with `stt_ms` ≈ 1226ms — a real
cloud round trip, not a stub.

Also noticed: no `.gitignore` existed yet, and this wasn't even a git repo. Added
one immediately (`.env`, `__pycache__/`, local Qdrant/SQLite data all excluded)
before a real secret sat in a soon-to-be-pushed repo.

### 4.5 Getting the real dataset — diagnosing why it was hard

Given a Hugging Face token, re-attempted the real dataset. `load_dataset(...,
streaming=True)` still hung indefinitely — not an auth problem. Investigated
directly against the HF API and found the real shape of the problem:

- `ai4bharat/MSMARCO-XI` ships per-language **parquet files** (`train/hintrain.parquet`,
  `validation/hinval.parquet`, etc. — 13 languages: Assamese, Bengali, Gujarati,
  Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Urdu)
  behind a **legacy `datasets` loading script** (`ms_marco_translations.py`).
  Modern versions of the `datasets` library handle script-based loaders badly —
  that's why it hung instead of erroring cleanly.
- `train/*` files are ~3.7-4.0GB **per language**. `validation/*` files are
  ~460-495MB per language — the same real dataset, just a smaller (and still
  entirely legitimate) split. Switched the default to `validation`.
- The real schema turned out to be quite different from what the generic loader
  assumed: `passages` is a struct of `{English_passages: [...], Translated_passages:
  [...], is_selected: [...]}`, not the generic `positive_passages`/`negative_passages`
  shape the code had been written for. `dataset_loader.py` was rewritten with a
  dedicated parser for this exact schema — positive (`is_selected==1`) passages
  get the query attached for the `qa_fused` chunk strategy, negative ones don't
  (so a genuinely irrelevant passage never gets a misleadingly high-relevance
  fused chunk).
- `build_index.py` was rewritten to stream the parquet files directly via
  `huggingface_hub.HfFileSystem` + `pyarrow`, bypassing the broken loading
  script entirely, with column projection to skip unused fields.

The fake-data fallback was removed entirely — a failed build now fails loudly
instead of silently reverting to placeholder trivia.

### 4.6 Actually pulling the data: a bandwidth fight

Running the new indexer surfaced a second, purely environmental problem: this
sandbox's connection to Hugging Face's storage backend measured **~770KB/s** —
far too slow to pull real data in reasonable time. Worse, every one of these
parquet files stores **all of its rows in a single Parquet row group**, so
`pyarrow` has to materialize the relevant column data as one block regardless of
how few rows are requested — there's no way to cheaply read "just the first 40
rows." That one-time cost was ~6-9 minutes per language file, but crucially it's
**fixed per file, not per row**: once paid, pulling hundreds more rows from the
same open file is nearly instant.

Given that, and given a strict ~10-minute-per-command ceiling on this
environment's tooling, the indexing script was extended with an `--append` flag
so each language could be indexed in its own command invocation without wiping
out languages indexed in a previous run. (Two supporting fixes came out of this:
`recreate_collection` doesn't reliably purge old on-disk segments in Qdrant's
local/embedded mode, so collection wipes now do an explicit `delete_collection`
first; and `ensure_sqlite()` was changed to `DROP TABLE` before recreating,
since `CREATE TABLE IF NOT EXISTS` was silently letting old rows survive a
"clean" rebuild — both of these had let 20 stale placeholder chunks quietly
survive one early rebuild, caught by noticing the chunk count was 20 higher
than expected.)

First real indexing run: **Hindi, 400 rows → 4,751 real chunks**, verified by
reading actual chunk text back out of the index (real MSMARCO passages about
McDonald's Corporation, a Rachel Carson essay, the definition of honesty — not
fabricated). Bandwidth in this environment later improved substantially (a later
Bengali run's one-time materialization took ~95 seconds instead of ~8 minutes),
which made adding more languages much faster — see §4.9.

### 4.7 The guardrail breaks again — and why that's actually informative

Re-running the benchmark against the real, larger Hindi corpus produced a
troubling result: 7 of 30 previously-abstaining queries now answered — with
garbage. *"Explain SQLite FTS5 full text search indexing"* returned an
unrelated JavaScript error message pulled from some forum post; *"How to
prevent hallucination in RAG systems?"* returned Air Force safety equipment
instructions. Confidently, as grounded answers.

The §4.3 fix (fixed absolute threshold of 0.80) had been calibrated on a
20-chunk corpus and silently stopped working on a 4,751-chunk one. Measuring
again explained why: e5-style embeddings have a "noise floor" — a baseline
cosine similarity that even *completely unrelated* text pairs tend to sit
around — and that noise floor **rises with corpus size**, because you're taking
a max over more candidates. It measured ~0.70-0.74 on 20 chunks and ~0.75-0.84
on 4,751 chunks. A threshold tuned for one corpus size is not a threshold that
generalizes to another.

**Fix: replaced the absolute-threshold guard with a margin-based one.**
Instead of asking "is the top score above some fixed number," it asks "does the
top hit stand out meaningfully above the general noise for *this specific
query*" — computed as `top_dense_score - mean(scores of rank 10-40)`. This is
relative, not absolute, so it doesn't drift as the corpus grows. Calibrated
against real data pulled from the indexed corpus itself: 2 known-relevant
queries scored a margin of 0.076 and 0.112; 4 known off-topic queries scored
0.018-0.040. Threshold set at 0.055 — roughly the midpoint, well clear of both
clusters. (Full detail and the exact numbers are in `GUARDRAILS.md`.)

### 4.8 A second bug, found while spot-checking the fix

While verifying the margin-based guard, one *should-answer* query
("कॉर्पोरेशन क्या है?" — "what is a corporation?") kept failing to produce a
correct answer even though retrieval was finding the right passage. Debugging
traced it to `important_terms()` in `app/generator.py`: Python's standard `re`
module's Unicode `\w` **does not include combining marks** (the Mn/Mc Unicode
categories), and Devanagari (like most Indic scripts) builds words out of base
consonants plus combining vowel signs and a virama. `\w+` was shredding
"कॉर्पोरेशन" into fragments like `['क', 'र', 'प', 'र', 'शन']` — meaningless
single-character tokens — instead of keeping it as one word. This silently broke
term-overlap scoring for sentence selection, and the same bug in
`escape_fts_query()` in `app/retriever.py` was corrupting the SQLite FTS5 lexical
query the same way, for **every non-English query and passage in the entire
dataset**. (SQLite's own `unicode61` FTS5 tokenizer, checked separately, handles
this correctly — the index itself was fine, only the query-construction code was
broken.)

Fixed by switching both call sites to the third-party `regex` package's Unicode
property syntax, `[\p{L}\p{M}\p{N}]+`, which correctly keeps letter+mark+digit
runs together as one token. Also added a small set of Hindi stopwords
(क्या, है, के, का, की, में, ...) alongside the existing English ones, since
these function words would now be correctly tokenized as real words and
shouldn't dominate lexical matching.

### 4.9 Final verification, and rebuilding the benchmark set honestly

The original `benchmark_queries.json` had been authored against the *fake*
placeholder corpus (questions about Goa, RAG, Qdrant), so once real Hindi
MSMARCO content was indexed, every single one of those queries correctly
abstained — which is *correct behavior*, but it meant the benchmark was no
longer exercising the "successfully answers" path at all. Rebuilt the query set
with 15 real queries pulled verbatim from the indexed corpus (so they're
guaranteed to have a legitimate match) plus 15 genuinely off-topic/unsafe/
prompt-injection queries.

Final result: **29/30 (96.7%) correct abstain-vs-answer decisions.** All 15
safety/off-topic/injection queries correctly refused or abstained; 14/15 real
queries got correctly grounded answers; the one miss was a false-negative
abstention (declining to answer rather than hallucinating) — a safe failure
mode, not a wrong answer.

Also did a full HTTP-level sanity check: started the actual `uvicorn` server and
hit `/health` and `/api/ask-text` over real HTTP with `curl`. One test initially
looked like a regression (a real-corpus query abstaining when it shouldn't) —
traced to the shell mangling the Devanagari UTF-8 text passed as an inline
command-line argument (a red herring from the *test method*, not the app);
confirmed correct behavior once the payload was sent from a properly
UTF-8-encoded file instead.

**Real speech test.** Sarvam had only been validated with a synthetic sine tone
up to this point (a real 200 OK response, but not real words). Used Windows'
built-in offline TTS (`System.Speech.Synthesis`) to generate a genuine spoken
WAV file saying "What is the capital of Goa," fed it through
`SarvamSTT.transcribe()`, and got back **"What is the capital of Goa?"** —
correctly transcribed, near-perfect. This confirms the STT leg of the pipeline
handles real speech content correctly, not just that it returns *something*.

### 4.10 Pushing to GitHub

Initialized a git repository (there wasn't one before), staged everything —
confirmed `.env`, the local SQLite/Qdrant data files, and the STT cache were all
correctly excluded by `.gitignore`, leaving only source, docs, and the tracked
benchmark JSON evidence (29 files, ~3,300 lines) — and made an initial commit.
Created a public GitHub repository and pushed:
**https://github.com/Ganesh-0509/hh-goa-2026-voice-rag**

### 4.11 Growing the corpus beyond one language

With bandwidth in this environment now much better (a later run's one-time
materialization took ~70-95 seconds instead of ~8 minutes), used the `--append`
flow from §4.6 to add every remaining default language on top of the validated
Hindi index without wiping it:

- Hindi: 400 rows → 4,751 chunks (first run, §4.6)
- Bengali: 500 rows → 5,493 chunks
- Tamil: 500 rows → 5,668 chunks
- Urdu: 500 rows → 6,441 chunks
- Marathi: 500 rows → 5,602 chunks

Total: **27,955 chunks across 5 languages.**

### 4.12 A second latency problem, caused by success

Re-running the benchmark against this much bigger corpus blew the 200ms
target: P50 268ms, P100 842ms. The cause was flagged by Qdrant itself, in a
warning that had been scrolling past unnoticed during indexing: *"Local mode is
not recommended for collections with more than 20,000 points... Consider using
Qdrant in Docker or Qdrant Cloud."* Local/embedded Qdrant (used throughout,
since no server was running) does **exact brute-force search, not HNSW** —
`app/retriever.py` already configures `hnsw_ef=32` for HNSW search, but a
separate `UserWarning` confirms it's inert in local mode ("Local mode performs
exact (brute-force) search, so `search_params` has no effect"). A linear scan
over 27,955 vectors per query is a fundamentally different cost curve than an
HNSW graph traversal, and it showed.

Checked whether a real Qdrant server could be spun up instead (`docker-compose.yml`
already provisions one) — the Docker CLI was present but its daemon wasn't
running, and starting Docker Desktop isn't something scriptable from here. So
the corpus was pruned back down instead: using each chunk's `row_index` payload
field (present in Qdrant, joined against SQLite by the shared `chunk_id`), kept
only the first 100 query rows per language and deleted the rest from both
stores — going from 27,955 chunks back to **5,573**, without needing to
re-download or re-embed anything.

Re-benchmarked: P50 114.29ms, P70 122.69ms, P100 174.54ms — back under target,
though with less margin at P100 than the single-language version had. This is a
**local-mode-specific ceiling, not an architectural one**: running the bundled
Qdrant Docker container instead of local/embedded mode would restore real HNSW
indexing and remove this trade-off entirely, letting the corpus grow much larger
without a latency cost. Worth doing before a real deployment, if a Qdrant
server is available wherever this gets deployed.

Final correctness re-check on the pruned, 5-language corpus: still **29/30
(96.7%)**, with all 5 languages represented in the "should answer" set and all
correctly grounded.

### 4.13 Getting a live public link

The task requires a "live working link," not just a GitHub repo. Getting there
took three real attempts, each surfacing a genuine constraint:

**Choosing a host.** The initial choice (Render's free tier) turned out to be
infeasible before a single deploy attempt: it caps free web services at 512MB
RAM, and this app — FastAPI + PyTorch + `sentence-transformers` + a local
Qdrant instance — measured **1.9GB RAM in a local Docker test**. Railway's free
tier has the identical 512MB cap, so it was ruled out for the same reason.
Hugging Face Spaces looked promising next (16GB RAM on its free CPU tier,
and thematically fitting since the dataset itself lives on HF) — but attempting
to actually create a Docker Space returned `402 Payment Required`: as of mid-2026,
Docker/Gradio Space hosting requires a PRO subscription ($9/month); only fully
static (no-backend) Spaces remain free. Fly.io was checked too and has no free
tier at all anymore (removed in 2024). **Google Cloud Run** ended up being the
real answer: a genuine perpetual free tier (2M requests, 360K GB-seconds
memory, 180K vCPU-seconds/month) with true serverless scale-to-zero, so light
hackathon-judging traffic costs nothing — and `gcloud` was already installed
and authenticated in this environment. A dedicated GCP project
(`hh-goa-2026-voicerag`) was created specifically for this, rather than reusing
an unrelated existing project, with billing linked and Cloud Run/Cloud
Build/Artifact Registry APIs enabled.

**Shipping the index without committing 214MB to git.** `storage/chunks.sqlite`
and `storage/qdrant_db/` are git-ignored (correctly — one internal file is
140MB, over GitHub's 100MB hard limit for a normal push). Rather than have the
deployed container rebuild the index from Hugging Face on every cold start
(slow, and dependent on HF network conditions at deploy time), the current
5,573-chunk index was packaged as a ~75MB compressed tarball and attached to a
[GitHub Release](https://github.com/Ganesh-0509/hh-goa-2026-voice-rag/releases/tag/prebuilt-index-v1).
A small `scripts/entrypoint.sh` downloads and extracts it on container start
if it's not already present, then launches `uvicorn`.

**Deploy attempt 1 — OOM.** First deploy used `--memory 2Gi` (padding over the
locally-measured 1.9GB). It failed: Cloud Run's own logs showed *"Memory limit
of 2048 MiB exceeded with 2090 MiB used"* — just over the line. Redeployed with
`--memory 4Gi`.

**Deploy attempt 2 — startup timeout.** The memory fix revealed a second,
different problem: the container now had enough RAM, but took too long to
become ready. Reading the timestamped logs closely showed why: downloading the
`sentence-transformers` embedding model *from Hugging Face, on every container
cold start*, took about 100 seconds by itself, on top of ~80 seconds of Python
import overhead for `torch` and friends — pushing total startup past Cloud
Run's fixed 4-minute startup probe window. Fixed by baking the model into the
Docker image at *build* time (`RUN python -c "from sentence_transformers import
SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"`)
instead of downloading it at every runtime start, plus `--cpu-boost` for extra
startup speed margin.

**Deploy attempt 3 — success.** Verified live over the public internet, not
just locally: `/health` returns 200, an unsafe query ("How to make a bomb?")
is correctly refused, a real Hindi query ("कॉर्पोरेशन क्या है?" — "what is a
corporation?") is correctly answered with grounded content, and the web UI
loads. Deployed with `--min-instances 0` so it scales to zero (no idle cost)
under the account `ganeshkumart3095@gmail.com`.

**Live URL: https://hh-goa-voice-rag-686797972138.us-central1.run.app**

---

## 5. Where things stand

- **Real data**: 5,573 chunks across 5 languages (Hindi, Bengali, Tamil, Urdu,
  Marathi — 100 rows/language) from `ai4bharat/MSMARCO-XI`'s validation split.
- **Chunking**: 6 real strategies exercised on real data — `atomic_short_passage`,
  `qa_fused`, `sentence_group_140w`, `micro_80w_20o`, `standard_180w_40o`,
  `macro_420w_80o` (see `CHUNKING.md` for the full spec).
- **Retrieval**: hybrid dense (Qdrant, e5-small embeddings) + lexical (SQLite
  FTS5, BM25) with Reciprocal Rank Fusion and a parent-doc/strategy diversity
  filter.
- **Latency**: P50 114.29ms / P70 122.69ms / P100 174.54ms for the full
  post-STT path — under the 200ms target, though with tighter margin than a
  smaller corpus would have (see §4.12 for the local-mode-vs-HNSW trade-off
  behind that number). Real Sarvam STT round trip measured at ~1,226ms
  separately, as the task brief itself anticipates cloud STT will exceed the
  200ms figure.
- **Correctness**: 29/30 (96.7%) on a mixed real-query (all 5 languages) /
  off-topic benchmark.
- **Guardrails**: input-level (unsafe content, prompt injection), retrieval-level
  (margin-based off-topic detection, corpus-size-independent), and
  generation-level (grounding/hallucination check requiring ≥40% token support
  from retrieved context).
- **Repo**: public on GitHub, `.env` and local data stores correctly excluded.
  https://github.com/Ganesh-0509/hh-goa-2026-voice-rag
- **Live link**: deployed on Google Cloud Run, verified working over the public
  internet (see §4.13). https://hh-goa-voice-rag-686797972138.us-central1.run.app

## 6. How to reproduce / extend this

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env   # then fill in SARVAM_API_KEY

# Index more data (validation split; each new language's first read pays a
# one-time cost, see §4.6 — use --append to add a language without wiping
# existing ones). If you have a real Qdrant server running (recommended once
# you go past ~20K chunks, see §4.12), it'll be used automatically instead of
# local/embedded mode.
export HF_TOKEN=<your huggingface token>
python scripts/build_index.py --languages tam --max-rows 500 --append

# Benchmark
python scripts/make_benchmark_queries.py   # only if you want to regenerate the query set
python scripts/benchmark.py --num-queries 30

# Run the server
uvicorn app.main:app --reload
# open http://localhost:8000, record real speech, and try it end to end
```

## 7. What's still worth doing before final submission

- If deploying somewhere with Docker available, start the real Qdrant server
  (`docker compose up -d`) instead of relying on local/embedded mode — this
  removes the corpus-size-vs-latency trade-off from §4.12 and lets the corpus
  grow well past 100 rows/language without a latency penalty.
- Record real human speech (not just TTS) through the web UI as a final sanity
  check before the demo video — the deployed live link is the right place to
  do this: https://hh-goa-voice-rag-686797972138.us-central1.run.app
- Record and post the two required videos (team/process, ≤90s; demo, end-to-end)
  with `#RAGInGoa` on Instagram and X, from every team member individually, per
  the task's promotion requirement.
- Fill in and submit the hackathon submission form with the GitHub repo link
  and the live Cloud Run link above. Double-check everything one more time
  first — no resubmissions are allowed.
