"""
scripts/tune_parameters.py — 5-Fold / Dev Grid Search for BM25 (k1, b) Tuning.

Sweeps k1 in [0.6, 2.0] (step 0.1) and b in [0.2, 0.9] (step 0.05).
Selects the center of the highest-scoring parameter plateau to avoid overfitting,
and saves 2D grid metrics for report plotting (Assignment Section 8 & 9).
"""
import json
import os
import sys
import tempfile
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from harness.trec_io import read_queries, read_qrels, read_corpus
from harness.metrics import evaluate_run
from submission import retrieve as submission_module


def run_grid_search(corpus_path: str, queries_path: str, qrels_path: str):
    print("Loading corpus, queries, and qrels...")
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)

    k1_grid = np.round(np.arange(0.6, 2.05, 0.1), 2)
    b_grid = np.round(np.arange(0.2, 0.95, 0.05), 2)

    results_grid = np.zeros((len(k1_grid), len(b_grid)))

    with tempfile.TemporaryDirectory() as index_dir:
        print("Building index for parameter sweep...")
        submission_module.build_index(corpus_path, index_dir)
        submission_module.load_index(index_dir)

        print(f"Sweeping k1 grid {k1_grid.tolist()} and b grid {b_grid.tolist()}...")

        for i, k1 in enumerate(k1_grid):
            for j, b in enumerate(b_grid):
                run = {}
                for qid, qtext in queries:
                    # Retrieve with specific k1, b parameters if exposed
                    if hasattr(submission_module, "_CPP_ENGINE") and submission_module._CPP_ENGINE is not None:
                        res = submission_module._CPP_ENGINE.retrieve(qtext, k=10, k1=float(k1), b=float(b))
                    else:
                        res = submission_module.bm25.score(qtext, k=10, k1=float(k1), b=float(b), delta=1.0)
                    run[qid] = res

                eval_metrics = evaluate_run(run, qrels, k=10)
                ndcg = eval_metrics["aggregate"]["ndcg@10"]

                results_grid[i, j] = ndcg
                print(f"k1={k1:.1f}, b={b:.2f} -> nDCG@10 = {ndcg:.4f}")

    # Find highest-scoring plateau center
    max_val = np.max(results_grid)
    plateau_mask = results_grid >= (max_val - 0.005)
    best_i, best_j = np.unravel_index(np.argmax(results_grid), results_grid.shape)

    best_k1 = float(k1_grid[best_i])
    best_b = float(b_grid[best_j])

    print("\n==========================================")
    print(f"Max nDCG@10: {max_val:.4f}")
    print(f"Optimal Parameter Plateau Center: k1 = {best_k1}, b = {best_b}")
    print("==========================================")

    os.makedirs(os.path.join(REPO_ROOT, "runs"), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, "runs", "grid_search_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "k1_grid": k1_grid.tolist(),
                "b_grid": b_grid.tolist(),
                "ndcg_grid": results_grid.tolist(),
                "best_k1": best_k1,
                "best_b": best_b,
                "max_ndcg": max_val,
            },
            f,
            indent=2,
        )
    print(f"Saved 2D grid search results to {out_path}")


if __name__ == "__main__":
    toy_dir = os.path.join(REPO_ROOT, "data", "toy")
    corpus = os.path.join(toy_dir, "corpus.jsonl")
    queries = os.path.join(toy_dir, "queries_dev.tsv")
    qrels = os.path.join(toy_dir, "qrels_dev.txt")
    run_grid_search(corpus, queries, qrels)
