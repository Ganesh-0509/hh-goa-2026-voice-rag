import hashlib
from typing import Any, Dict, List

from app.chunking import RawDoc, normalize_text


TEXT_KEYS = [
    "text",
    "passage_text",
    "body",
    "content",
    "document",
    "context",
    "answer",
    "is_selected",
]

TITLE_KEYS = [
    "title",
    "heading",
    "url",
]

PASSAGE_FIELDS = [
    "positive_passages",
    "negative_passages",
    "passages",
    "contexts",
    "documents",
    "positive",
    "negative",
]


def stable_doc_id(text: str, prefix: str = "doc") -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{h}"


def get_first(obj: Dict, keys: List[str], default: str = "") -> str:
    for k in keys:
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def dict_of_lists_to_items(obj: Dict) -> List[Dict]:
    lengths = []
    for v in obj.values():
        if isinstance(v, list):
            lengths.append(len(v))

    if not lengths:
        return [obj]

    n = max(lengths)
    items = []

    for i in range(n):
        item = {}
        for k, v in obj.items():
            if isinstance(v, list):
                item[k] = v[i] if i < len(v) else None
            else:
                item[k] = v
        items.append(item)

    return items


def passage_items(value: Any) -> List[Dict]:
    if value is None:
        return []

    if isinstance(value, str):
        return [{"text": value}]

    if isinstance(value, dict):
        return dict_of_lists_to_items(value)

    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append({"text": item})
            elif isinstance(item, dict):
                out.append(item)
        return out

    return []


def row_to_docs_msmarco_xi(row: Dict, config_name: str, split: str, row_index: int) -> List[RawDoc]:
    """
    Parser for the real ai4bharat/MSMARCO-XI parquet schema:
    source_lang, target_lang, meta, Answer, query_id, query_type,
    passages: {English_passages: [str], Translated_passages: [str], is_selected: [int]},
    Eng_Query, Eng_Answer, query.

    `is_selected[i] == 1` marks the passage MS MARCO judged relevant to the query;
    we only attach the query to the qa_fused chunk for those, so irrelevant
    (negative) passages don't get a misleadingly high-relevance fused chunk.
    """
    docs: List[RawDoc] = []

    query = row.get("query") or row.get("Eng_Query") or ""
    query_id = row.get("query_id")
    query_type = row.get("query_type") or ""
    language = row.get("target_lang") or config_name or ""

    passages = row.get("passages") or {}
    translated = passages.get("Translated_passages") or []
    is_selected = passages.get("is_selected") or []

    for idx, text in enumerate(translated):
        text = normalize_text(text or "")
        if len(text) < 40:
            continue

        selected = idx < len(is_selected) and is_selected[idx] == 1
        source_type = "positive_passages" if selected else "negative_passages"

        docs.append(
            RawDoc(
                doc_id=f"{config_name}_{query_id}_{idx}",
                text=text,
                title="",
                language=language,
                source_type=source_type,
                query=query if selected else "",
                metadata={
                    "dataset_config": config_name,
                    "split": split,
                    "row_index": row_index,
                    "query_id": query_id,
                    "query_type": query_type,
                    "passage_index": idx,
                    "is_selected": selected,
                },
            )
        )

    return docs


def row_to_docs(row: Dict, config_name: str, split: str, row_index: int) -> List[RawDoc]:
    if isinstance(row.get("passages"), dict) and "Translated_passages" in row["passages"]:
        return row_to_docs_msmarco_xi(row, config_name, split, row_index)

    docs: List[RawDoc] = []

    query = (
        row.get("query")
        or row.get("question")
        or row.get("query_text")
        or ""
    )

    language = (
        row.get("language")
        or row.get("lang")
        or config_name
        or ""
    )

    for field in PASSAGE_FIELDS:
        if field not in row:
            continue

        for j, item in enumerate(passage_items(row[field])):
            text = get_first(item, TEXT_KEYS)
            text = normalize_text(text)

            if len(text) < 40:
                continue

            title = get_first(item, TITLE_KEYS)
            pid = (
                item.get("docid")
                or item.get("doc_id")
                or item.get("pid")
                or stable_doc_id(text, prefix=f"{config_name}_{field}")
            )

            docs.append(
                RawDoc(
                    doc_id=str(pid),
                    text=text,
                    title=title,
                    language=language,
                    source_type=field,
                    query=query,
                    metadata={
                        "dataset_config": config_name,
                        "split": split,
                        "row_index": row_index,
                        "passage_index": j,
                    },
                )
            )

    # fallback for unusual/flat schemas
    if not docs:
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 120 and key not in {"query", "question", "query_id", "id"}:
                docs.append(
                    RawDoc(
                        doc_id=stable_doc_id(value, prefix=f"{config_name}_{key}"),
                        text=value,
                        title="",
                        language=language,
                        source_type=key,
                        query=query,
                        metadata={
                            "dataset_config": config_name,
                            "split": split,
                            "row_index": row_index,
                        },
                    )
                )

    return docs
