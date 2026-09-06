"""
scripts/error_analysis.py — Error Analysis Tool for 3-5 Poorly Performing Queries.

Runs evaluation per-query, identifies low nDCG queries, and dumps detailed diagnostic info for the report.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from harness.trec_io import read_queries, read_qrels
from harness.metrics import dcg_at_k
from submission import retrieve as submission_module


def analyze_errors(corpus_path: str, queries_path: str, qrels_path: str):
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)

    query_map = {qid: text for qid, text in queries}

    with tempfile.TemporaryDirectory() as index_dir:
        submission_module.build_index(corpus_path, index_dir)
        submission_module.load_index(index_dir)

        query_scores = []

        for qid, qtext in queries:
            results = submission_module.retrieve(qtext, k=10)
            retrieved_ids = [doc_id for doc_id, _score in results]
            relevant_dict = qrels.get(qid, {})

            # Compute ideal DCG
            rel_scores = sorted(relevant_dict.values(), reverse=True)
            idcg = dcg_at_k(rel_scores, k=10)

            # Compute actual DCG
            actual_rels = [relevant_dict.get(doc_id, 0) for doc_id in retrieved_ids]
            actual_dcg = dcg_at_k(actual_rels, k=10)

            ndcg = actual_dcg / idcg if idcg > 0 else 0.0
            query_scores.append((ndcg, qid, qtext, retrieved_ids, relevant_dict))

    # Sort queries by nDCG ascending (worst first)
    query_scores.sort(key=lambda x: x[0])

    print("==========================================")
    print("ERROR ANALYSIS: Lowest nDCG@10 Queries")
    print("==========================================")

    analysis_out = []
    for rank, (ndcg, qid, qtext, retrieved, relevant) in enumerate(query_scores[:5], 1):
        print(f"\n[{rank}] Query ID: {qid}")
        print(f"    Query Text:  {qtext!r}")
        print(f"    nDCG@10:     {ndcg:.4f}")
        print(f"    Top 5 Retr:  {retrieved[:5]}")
        print(f"    Relevant:    {dict(list(relevant.items())[:5])}")

        analysis_out.append({
            "qid": qid,
            "query": qtext,
            "ndcg": ndcg,
            "retrieved_top10": retrieved,
            "relevant_docs": relevant
        })

    os.makedirs(os.path.join(REPO_ROOT, "runs"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "runs", "error_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis_out, f, indent=2)
    print(f"\nDetailed error analysis dumped to {out_path}")


if __name__ == "__main__":
    toy_dir = os.path.join(REPO_ROOT, "data", "toy")
    corpus = os.path.join(toy_dir, "corpus.jsonl")
    queries = os.path.join(toy_dir, "queries_dev.tsv")
    qrels = os.path.join(toy_dir, "qrels_dev.txt")
    analyze_errors(corpus, queries, qrels)
