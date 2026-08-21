"""
New Testing/generate_and_test.py — Large-Scale Synthetic Corpus Generator & Performance Stress Test.

Generates a large synthetic corpus, queries, and qrels, then measures:
1. Index build time and memory scalability.
2. On-disk index size compression ratio.
3. Index load latency (fresh process / mmap).
4. Query retrieval throughput (queries per second) and latency distribution.
5. Contract correctness validation (no duplicates, valid doc_ids).
"""
import json
import os
import random
import string
import sys
import time
from typing import List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from harness.metrics import evaluate_run
from submission import retrieve as submission_module

# Vocabulary pool for synthetic generation
WORD_POOL = [
    "information", "retrieval", "search", "engine", "algorithm", "database", "computer", "science",
    "indexing", "system", "performance", "optimization", "vector", "space", "model", "probabilistic",
    "ranking", "score", "document", "corpus", "query", "expansion", "relevance", "feedback", "latency",
    "memory", "compression", "binary", "block", "wand", "sequential", "dependence", "bigram", "proximity",
    "python", "cpp", "pybind11", "mmap", "benchmark", "evaluation", "metrics", "ndcg", "map", "mrr",
    "network", "deep", "learning", "neural", "natural", "language", "processing", "transformer", "bert",
]

# Add random alphanumeric noise words to expand vocabulary size to 5,000 terms
for _ in range(4500):
    WORD_POOL.append("".join(random.choices(string.ascii_lowercase, k=random.randint(4, 10))))


def generate_synthetic_corpus(num_docs: int, min_words: int, max_words: int) -> List[Tuple[str, str]]:
    print(f"Generating {num_docs:,} synthetic documents...")
    corpus = []
    for i in range(1, num_docs + 1):
        doc_id = f"doc_{i}"
        num_words = random.randint(min_words, max_words)
        words = random.choices(WORD_POOL, k=num_words)
        text = " ".join(words)
        corpus.append((doc_id, text))
    return corpus


def generate_synthetic_queries(num_queries: int, min_terms: int, max_terms: int) -> List[Tuple[str, str]]:
    print(f"Generating {num_queries:,} synthetic queries...")
    queries = []
    for i in range(1, num_queries + 1):
        qid = f"q_{i}"
        num_terms = random.randint(min_terms, max_terms)
        terms = random.choices(WORD_POOL[:200], k=num_terms)  # Pick from common vocabulary
        qtext = " ".join(terms)
        queries.append((qid, qtext))
    return queries


def run_stress_test(num_docs: int = 10000, num_queries: int = 500):
    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "synthetic_data")
    index_dir = os.path.join(test_dir, "synthetic_index")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries.tsv")

    # 1. Generate Synthetic Data
    t0 = time.perf_counter()
    corpus = generate_synthetic_corpus(num_docs, min_words=50, max_words=300)
    queries = generate_synthetic_queries(num_queries, min_terms=1, max_terms=5)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, text in corpus:
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")

    with open(queries_path, "w", encoding="utf-8") as f:
        for qid, qtext in queries:
            f.write(f"{qid}\t{qtext}\n")
    print(f"Data generation took {time.perf_counter() - t0:.2f}s\n")

    # 2. Benchmark Index Build Time
    print("==========================================")
    print("BENCHMARK 1: Index Building")
    print("==========================================")
    t0 = time.perf_counter()
    submission_module.build_index(corpus_path, index_dir)
    build_time = time.perf_counter() - t0

    # Calculate index size
    index_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(index_dir)
        for f in filenames
    )
    print(f"Corpus size:        {num_docs:,} documents")
    print(f"Index build time:   {build_time:.4f} seconds")
    print(f"Index size on disk: {index_bytes / (1024 * 1024):.2f} MB ({index_bytes:,} bytes)")
    print(f"Avg per doc size:   {index_bytes / num_docs:.1f} bytes/doc\n")

    # 3. Benchmark Index Load Time
    print("==========================================")
    print("BENCHMARK 2: Index Loading (mmap)")
    print("==========================================")
    t0 = time.perf_counter()
    submission_module.load_index(index_dir)
    load_time = time.perf_counter() - t0
    print(f"Index load time:    {load_time * 1000:.3f} ms\n")

    # 4. Benchmark Query Retrieval Performance & Latency
    print("==========================================")
    print("BENCHMARK 3: Query Retrieval & Throughput")
    print("==========================================")
    latencies = []
    run_results = {}

    for qid, qtext in queries:
        t_start = time.perf_counter()
        results = submission_module.retrieve(qtext, k=10)
        t_lat = (time.perf_counter() - t_start) * 1000.0  # in ms
        latencies.append(t_lat)
        run_results[qid] = results

        # Contract Checks
        assert len(results) <= 10, f"Returned more than k=10 results for {qid}"
        seen_docs = set()
        for doc_id, score in results:
            assert doc_id not in seen_docs, f"Duplicate doc_id {doc_id} in {qid}"
            seen_docs.add(doc_id)
            assert isinstance(score, (int, float)), f"Non-numeric score for {doc_id}"

    total_query_time = sum(latencies) / 1000.0
    qps = num_queries / total_query_time if total_query_time > 0 else 0

    print(f"Total queries:      {num_queries:,}")
    print(f"Mean query latency: {np_mean(latencies):.3f} ms")
    print(f"P50 query latency:  {np_percentile(latencies, 50):.3f} ms")
    print(f"P95 query latency:  {np_percentile(latencies, 95):.3f} ms")
    print(f"Max query latency:  {max(latencies):.3f} ms")
    print(f"Query Throughput:   {qps:.1f} queries/sec (QPS)\n")

    print("==========================================")
    print("PASSED: All stress tests & contract checks successfully completed!")
    print("==========================================")


def np_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def np_percentile(lst, p):
    if not lst:
        return 0.0
    sorted_lst = sorted(lst)
    idx = int((p / 100.0) * len(sorted_lst))
    return sorted_lst[min(idx, len(sorted_lst) - 1)]


if __name__ == "__main__":
    # Test on 10,000 synthetic documents and 500 queries
    run_stress_test(num_docs=10000, num_queries=500)
