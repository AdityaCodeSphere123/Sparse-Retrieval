#!/usr/bin/env python
"""Score the competition retrieve() on every local proxy/dev collection."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import retrieve as r

SETS = [
    ("cord", "data/full"),
    ("cranfield", "data/cranfield"),
    ("cacm", "data/cacm"),
    ("touche", "data/proxy_touche"),
    ("fiqa", "data/proxy_fiqa"),
    ("scifact", "data/proxy_scifact"),
    ("nfcorpus", "data/proxy_nfcorpus"),
    ("arguana", "data/proxy_arguana"),
    ("antique", "data/proxy_antique"),
]


def main() -> None:
    for name, folder in SETS:
        corpus = os.path.join(folder, "corpus.jsonl")
        queries_p = os.path.join(folder, "queries_dev.tsv")
        qrels_p = os.path.join(folder, "qrels_dev.txt")
        if not (os.path.isfile(corpus) and os.path.isfile(queries_p) and os.path.isfile(qrels_p)):
            print(f"{name:12} SKIP (missing files)")
            continue
        idx = os.path.join(folder, "_index")
        os.makedirs(idx, exist_ok=True)
        print(f"{name:12} indexing...", flush=True)
        r.build_index(corpus, idx)
        r.load_index(idx)
        queries = read_queries(queries_p)
        qrels = read_qrels(qrels_p)
        run = {qid: r.retrieve(qtext, k=10) for qid, qtext in queries}
        agg = evaluate_run(run, qrels, k=10)["aggregate"]
        n = r._CPP_ENGINE.N if r._CPP_ENGINE is not None else 0
        avg = r._CPP_ENGINE.avg_doc_len if r._CPP_ENGINE is not None else 0.0
        print(
            f"{name:12} N={n:<7} avgdl={avg:7.1f}  "
            f"k1={r._ACTIVE['k1']:.2f} b={r._ACTIVE['b']:.2f}  "
            f"nDCG@10={agg['ndcg@10']:.4f} MAP@10={agg['map@10']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
