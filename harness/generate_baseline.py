"""
harness/generate_baseline.py — generate YOUR OWN local BM25 comparison
run, using the third-party `rank_bm25` library with generic textbook
parameters (k1=1.2, b=0.75 by default — the values commonly cited as
reasonable defaults, e.g. in Robertson & Zaragoza's survey).

Read this carefully: this is a tool for your own sanity-checking, not a
reproduction of the official grading baseline. Course staff score you
against a separately tuned BM25 reference on data you don't have, with
parameters that are not disclosed and are deliberately not assumed to
match the defaults below (assignment Section 7). Running this script and
confirming you beat *its* output tells you very little about whether
you'd beat the real one — use it to catch obviously broken rankings, not
as a finish line.

This is explicitly the "reference tooling" mentioned in the assignment
(Section 6): it is a standard library call, not a from-scratch
implementation, so running or reading this script does not give away a
Boolean/VSM/BM25 implementation you could pass off as your own. Using
`rank_bm25` (or any other existing search library) inside your own
submission/ code is not allowed — see the assignment's Academic
Integrity section.

Usage:
    python -m harness.generate_baseline \
        --corpus data/toy/corpus.jsonl \
        --queries data/toy/queries_dev.tsv \
        --out data/toy/reference_bm25_run_dev.trec \
        --k 10
"""
import argparse
import time

from harness.trec_io import read_corpus, read_queries, write_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit(
            "rank_bm25 is not installed. Run `pip install rank_bm25` "
            "(it is already listed in requirements.txt) and try again."
        )

    corpus = read_corpus(args.corpus)
    doc_ids = [doc_id for doc_id, _text in corpus]
    tokenized_docs = [text.lower().split() for _doc_id, text in corpus]

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized_docs, k1=args.k1, b=args.b)
    build_time = time.perf_counter() - t0

    queries = read_queries(args.queries)
    run = {}
    t0 = time.perf_counter()
    for qid, text in queries:
        scores = bm25.get_scores(text.lower().split())
        ranked = sorted(zip(doc_ids, scores), key=lambda pair: pair[1], reverse=True)[: args.k]
        run[qid] = ranked
    query_time = time.perf_counter() - t0

    write_run(args.out, run, run_tag=f"rank_bm25-reference-k1={args.k1}-b={args.b}")
    print(f"Wrote baseline run for {len(queries)} queries to {args.out}")
    print(f"Index build time: {build_time:.4f}s | total query time: {query_time:.4f}s "
          f"({query_time / max(len(queries), 1) * 1000:.2f} ms/query)")


if __name__ == "__main__":
    main()
