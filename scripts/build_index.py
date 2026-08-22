import argparse
import os
import sqlite3
import sys
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent dir to sys.path to allow app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import make_chunks
from app.config import settings
from app.dataset_loader import row_to_docs

DATASET = "ai4bharat/MSMARCO-XI"

# ai4bharat/MSMARCO-XI ships per-language parquet files under train/ and validation/
# (e.g. train/hintrain.parquet, validation/hinval.parquet) instead of a working
# HF `datasets` loading script/config split. We resolve and stream those parquet
# files directly with datasets' generic "parquet" builder, which bypasses the
# broken legacy script entirely.
DEFAULT_LANGUAGES = ["hin", "ben", "tam", "urd", "mar"]

# Only pull the columns row_to_docs_msmarco_xi actually uses. These files store
# every column of one split's ~98K-778K rows in a SINGLE parquet row group, so
# pyarrow must materialize the full column chunk for any requested column before
# yielding even one row - there's no way to cheaply read "just the first N rows".
# Column projection still helps by skipping 'meta'/'Answer'/'Eng_Answer', which are
# unused. Once that one-time read completes, pulling more rows from the same open
# file is fast, so --max-rows only controls how many are kept, not how long the
# initial read takes.
NEEDED_COLUMNS = ["query", "Eng_Query", "query_id", "query_type", "target_lang", "source_lang", "passages"]


def stream_parquet_rows(hf_path: str, max_rows: int, hf_token: str = None):
    """Yield up to max_rows dict rows from a hf:// parquet path, columns-projected."""
    fs = HfFileSystem(token=hf_token)
    rel_path = hf_path.replace("hf://", "")

    count = 0
    with fs.open(rel_path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=NEEDED_COLUMNS):
            for row in batch.to_pylist():
                yield row
                count += 1
                if count >= max_rows:
                    return


def resolve_parquet_urls(split: str, languages):
    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET, repo_type="dataset")
    split_files = [f for f in files if f.startswith(f"{split}/") and f.endswith(".parquet")]

    if languages:
        wanted = set(languages)
        split_files = [f for f in split_files if Path(f).name[:3] in wanted]

    if not split_files:
        raise RuntimeError(
            f"No parquet files found for split='{split}' languages={languages}. "
            f"Available files: {files}"
        )

    return [f"hf://datasets/{DATASET}/{f}" for f in sorted(split_files)]


def ensure_sqlite(path: str, wipe: bool = True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    if wipe:
        # Drop any previous build's rows (e.g. earlier placeholder/demo data) so a
        # fresh run produces a clean index. With --append, skip this so multiple
        # per-language runs (each bounded by this tool's ~10min timeout) accumulate
        # into one index instead of each wiping the last.
        cur.execute("DROP TABLE IF EXISTS chunks_fts")
        cur.execute("DROP TABLE IF EXISTS chunks_meta")

    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(
        chunk_id UNINDEXED,
        text,
        title,
        language,
        strategy,
        tokenize='unicode61'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks_meta (
        chunk_id TEXT PRIMARY KEY,
        text TEXT,
        title TEXT,
        language TEXT,
        strategy TEXT,
        parent_doc_id TEXT
    )
    """)

    conn.commit()
    return conn


def insert_sqlite(conn, chunk):
    cur = conn.cursor()
    p = chunk.payload

    cur.execute(
        "INSERT OR REPLACE INTO chunks_meta VALUES (?, ?, ?, ?, ?, ?)",
        (
            chunk.chunk_id,
            chunk.text,
            p.get("title", ""),
            p.get("language", ""),
            p.get("chunk_strategy", ""),
            p.get("parent_doc_id", ""),
        ),
    )

    cur.execute(
        "INSERT INTO chunks_fts(chunk_id, text, title, language, strategy) VALUES (?, ?, ?, ?, ?)",
        (
            chunk.chunk_id,
            chunk.text,
            p.get("title", ""),
            p.get("language", ""),
            p.get("chunk_strategy", ""),
        ),
    )


def batched(items, batch_size):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def flush_chunks(chunk_buffer, model, client, conn, batch_size):
    total = 0
    for batch in batched(chunk_buffer, batch_size):
        texts = [f"passage: {c.text}" for c in batch]
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=batch_size)

        points = []
        for c, v in zip(batch, vectors):
            payload = dict(c.payload)
            payload["text"] = c.text

            points.append(
                PointStruct(
                    id=c.chunk_id,
                    vector=v.tolist(),
                    payload=payload,
                )
            )
            insert_sqlite(conn, c)

        client.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)

    conn.commit()
    return total


def main():
    parser = argparse.ArgumentParser(description="Index MSMARCO-XI into Qdrant & SQLite FTS5")
    parser.add_argument(
        "--languages", nargs="*", default=DEFAULT_LANGUAGES,
        help="3-letter language codes (e.g. hin ben tam). Pass 'all' for every available language.",
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation"],
                         help="Dataset split. 'validation' files (~460MB/lang) are far smaller than "
                              "'train' files (~3.7GB/lang) and are still real MSMARCO-XI data.")
    parser.add_argument("--max-rows", type=int, default=500, help="Max query rows per language to ingest")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--append", action="store_true",
                         help="Add to the existing index instead of wiping it first. Use this when "
                              "indexing languages one at a time across multiple runs (each language's "
                              "first read is slow - see README) so earlier languages aren't lost.")
    args = parser.parse_args()

    os.makedirs("storage", exist_ok=True)

    languages = None if args.languages == ["all"] else args.languages
    print(f"Resolving parquet files for split='{args.split}' languages={languages or 'all'}...")
    parquet_urls = resolve_parquet_urls(args.split, languages)
    print(f"Found {len(parquet_urls)} file(s):")
    for u in parquet_urls:
        print(" -", u)

    print(f"Loading embedding model: {settings.embed_model}...")
    model = SentenceTransformer(settings.embed_model)
    test_vec = model.encode(["passage: test"], normalize_embeddings=True)[0]
    dim = len(test_vec)
    print(f"Embedding dimension: {dim}")

    print("Connecting to Qdrant...")
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=2.0)
        client.get_collections()
        print(f"Connected to Qdrant server at {settings.qdrant_url}")
    except Exception:
        print(f"Qdrant server unreachable at {settings.qdrant_url}. Using local embedded database at '{settings.qdrant_path}'")
        client = QdrantClient(path=settings.qdrant_path)

    collection_exists = client.collection_exists(settings.qdrant_collection)

    if args.append and collection_exists:
        print(f"--append: keeping existing Qdrant collection '{settings.qdrant_collection}'")
    else:
        # recreate_collection alone doesn't reliably purge old on-disk segments in
        # Qdrant's local/embedded mode - explicitly delete first so stale points
        # from a previous build (e.g. earlier placeholder/demo data) can't survive.
        if collection_exists:
            client.delete_collection(settings.qdrant_collection)

        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created clean Qdrant collection: '{settings.qdrant_collection}'")

    conn = ensure_sqlite(settings.sqlite_fts_path, wipe=not args.append)
    total_chunks = 0

    hf_token = os.environ.get("HF_TOKEN")

    for url in parquet_urls:
        lang_code = Path(url).stem.replace(args.split[:3], "").replace("val", "").replace("train", "") or Path(url).stem
        print(f"\nStreaming '{url}' (max {args.max_rows} rows)... this file's column data is read in one "
              f"shot regardless of --max-rows, so this may take several minutes before the first row appears.")

        chunk_buffer = []
        row_count = 0

        for row_index, row in enumerate(tqdm(stream_parquet_rows(url, args.max_rows, hf_token), total=args.max_rows)):
            docs = row_to_docs(row, lang_code, args.split, row_index)
            for doc in docs:
                chunk_buffer.extend(make_chunks(doc))

            row_count += 1
            if row_count >= args.max_rows:
                break

        if chunk_buffer:
            n = flush_chunks(chunk_buffer, model, client, conn, args.batch_size)
            total_chunks += n
            print(f"Indexed {n} chunks from {row_count} rows for language '{lang_code}'.")
        else:
            print(f"WARNING: no chunks extracted for language '{lang_code}' from {row_count} rows.")

    conn.close()
    print(f"\nIndexing complete! Total chunks indexed across Qdrant & SQLite FTS5: {total_chunks}")

    if total_chunks == 0:
        print("ERROR: zero chunks were indexed. Check dataset schema / network access before using this index.")
        sys.exit(1)


if __name__ == "__main__":
    main()
