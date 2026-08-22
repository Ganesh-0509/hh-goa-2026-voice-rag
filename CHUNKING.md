# Multi-Strategy Adaptive Chunking Specification

Naive fixed-size chunking (e.g. splitting every 500 characters) leads to sentence fragmentation, context truncation, lost metadata, and suboptimal vector representations.

Our system implements a **vast, multi-strategy adaptive chunker** specifically tuned for multilingual datasets (`ai4bharat/MSMARCO-XI`).

---

## 1. Chunking Strategies Overview

| Strategy Name | Target Window Size | Overlap | Use Case & Rationale |
| :--- | :--- | :--- | :--- |
| `atomic_short_passage` | ≤ 140 words | Full Passage | Short passages are preserved as self-contained atomic chunks to maintain complete semantic integrity. |
| `metadata_title_intro` | First 180 words | N/A | Title, URL, and document introduction are prepended to provide strong high-level vector metadata anchor. |
| `qa_fused` | Question + Passage | N/A | If a query is present in dataset context, builds `"Question: {q}\nRelevant evidence: {p}"` for query-passage direct matching. |
| `sentence_group_140w` | ~140 words | 1 Sentence | Splits text at multilingual sentence boundaries (`.`, `?`, `!`, `।`, `॥`) preventing mid-sentence splits. |
| `micro_80w_20o` | 80 words | 20 words | Micro window for capturing hyper-focused facts, entities, and direct answer spans. |
| `standard_180w_40o` | 180 words | 40 words | Standard window for general passage representation. |
| `macro_420w_80o` | 420 words | 80 words | Macro window enabled exclusively for longer documents (>450 words) to capture multi-paragraph context. |

---

## 2. Deduplication & Unique Identification

1. **Normalized Hashing**: Every chunk text is normalized (lowercased, whitespace collapsed) and hashed with SHA256 to prevent duplicate chunk storage across Qdrant and SQLite FTS.
2. **Stable UUIDs**: Deterministic UUID version-5 mapping using namespace domain on `parent_doc_id|strategy|text`.

---

## 3. Metadata Payload Schema

Each indexed chunk retains comprehensive payload metadata:

```json
{
  "parent_doc_id": "config_field_doc123",
  "title": "Document Title or Heading",
  "language": "hi",
  "source_type": "positive_passages",
  "chunk_strategy": "sentence_group_140w",
  "dataset_config": "hindi",
  "split": "train",
  "row_index": 42
}
```
