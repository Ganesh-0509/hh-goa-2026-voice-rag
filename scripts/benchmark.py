import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.harness import VoiceRAGHarness


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(values, p)), 2)


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for Voice RAG pipeline")
    parser.add_argument("--queries-file", default="storage/benchmark_queries.json", help="JSON file with benchmark queries")
    parser.add_argument("--num-queries", type=int, default=30, help="Number of queries to run")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup queries")
    args = parser.parse_args()

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        print(f"Benchmark queries file not found at {queries_path}. Generating default set...")
        from scripts.make_benchmark_queries import main as make_queries
        make_queries()

    queries = json.loads(queries_path.read_text(encoding="utf-8"))[:args.num_queries]

    print("Initializing VoiceRAGHarness...")
    harness = VoiceRAGHarness()

    print(f"\nRunning {args.warmup} warmup queries...")
    for q in queries[:args.warmup]:
        harness.ask_text(q["query"])

    print(f"Running benchmark on {len(queries)} queries...\n")
    results = []

    for i, item in enumerate(queries, start=1):
        q_text = item["query"]
        res = harness.ask_text(q_text)

        t = res.timings_ms
        results.append({
            "id": item.get("id", f"q_{i}"),
            "query": q_text,
            "category": item.get("category", "general"),
            "abstained": res.abstained,
            "grounded": res.grounded,
            "timings": t,
        })
        q_printable = q_text[:35].encode("ascii", errors="ignore").decode("ascii") or "Query"
        print(f"[{i}/{len(queries)}] '{q_printable}...' -> Post-STT: {t.get('post_stt_total_ms', 0)}ms (Ret: {t.get('retrieval_ms', 0)}ms, Gen: {t.get('generation_ms', 0)}ms)")

    stages = [
        "input_guard_ms",
        "retrieval_ms",
        "retrieval_guard_ms",
        "generation_ms",
        "grounding_ms",
        "post_stt_total_ms",
    ]

    metrics: Dict[str, Dict[str, float]] = {}

    for stage in stages:
        vals = [r["timings"].get(stage, 0.0) for r in results if stage in r["timings"]]
        if vals:
            metrics[stage] = {
                "P50": percentile(vals, 50),
                "P70": percentile(vals, 70),
                "P100": percentile(vals, 100),
                "mean": round(float(np.mean(vals)), 2),
            }

    print("\n" + "=" * 65)
    print("           LATENCY BENCHMARK RESULTS (in milliseconds)          ")
    print("=" * 65)
    print(f"{'Stage Name':<22} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10} | {'Mean (ms)':<10}")
    print("-" * 65)

    for stage, val in metrics.items():
        print(f"{stage:<22} | {val['P50']:<10} | {val['P70']:<10} | {val['P100']:<10} | {val['mean']:<10}")

    print("=" * 65)

    post_stt_p50 = metrics.get("post_stt_total_ms", {}).get("P50", 0.0)
    print(f"\n>> Post-STT RAG Path P50 Latency: {post_stt_p50} ms Target (<200 ms): {'PASSED [OK]' if post_stt_p50 < 200 else 'NEEDS OPTIMIZATION'}")

    output_file = Path("storage/benchmark_results.json")
    output_data = {
        "num_queries": len(results),
        "metrics": metrics,
        "details": results,
    }
    output_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Full benchmark results saved to '{output_file}'")


if __name__ == "__main__":
    main()
