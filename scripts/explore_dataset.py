import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from scripts.build_index import resolve_parquet_urls

DATASET = "ai4bharat/MSMARCO-XI"


def main():
    print(f"Exploring '{DATASET}' (validation split, one language)...\n")
    print(
        "NOTE: this dataset ships per-language parquet files (train/*.parquet, "
        "validation/*.parquet) rather than a working `datasets` loading script/config "
        "split, so we resolve and stream the parquet files directly instead of calling "
        "load_dataset(DATASET, config, split=...)."
    )

    urls = resolve_parquet_urls("validation", ["hin"])
    print("\nFile:", urls[0])

    ds = load_dataset("parquet", data_files={"validation": urls[0]}, split="validation", streaming=True)
    row = next(iter(ds))

    print("\nRow keys:", list(row.keys()))
    for k, v in row.items():
        v_repr = str(v)[:200] + ("..." if len(str(v)) > 200 else "")
        print(f"  {k} ({type(v).__name__}): {v_repr}")


if __name__ == "__main__":
    main()
