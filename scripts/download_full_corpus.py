#!/usr/bin/env python
"""
Fetch the full corpus + released dev topics/qrels via ir_datasets
and write them into data/full/ in the same format used by data/toy/.

Default dataset: beir/trec-covid (~171K documents, public, no NIST forms
required). Course staff can point this at a different ir_datasets id by
passing --dataset, e.g. a Robust04 subset if your institution has a TREC
license.

Usage:
    python scripts/download_full_corpus.py
    python scripts/download_full_corpus.py --dataset beir/trec-covid --out data/full
"""
import argparse
import json
import os

DEFAULT_DATASET = "beir/trec-covid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="ir_datasets dataset id")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "full"))
    args = parser.parse_args()

    try:
        import ir_datasets
    except ImportError:
        raise SystemExit(
            "ir_datasets is not installed. Run `pip install ir_datasets` "
            "(it is already listed in requirements.txt) and try again."
        )

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading '{args.dataset}' via ir_datasets (this downloads and caches the dataset)...")
    dataset = ir_datasets.load(args.dataset)

    def _doc_text(obj) -> str:
        if hasattr(obj, "default_text"):
            try:
                t = obj.default_text()
                if t:
                    return t
            except Exception:
                pass
        for attr in ("text", "body", "title"):
            t = getattr(obj, attr, None)
            if t:
                return str(t)
        return ""

    corpus_path = os.path.join(args.out, "corpus.jsonl")
    n_docs = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in dataset.docs_iter():
            f.write(json.dumps({"doc_id": str(doc.doc_id), "text": _doc_text(doc)}) + "\n")
            n_docs += 1
    print(f"Wrote {n_docs} documents to {corpus_path}")

    queries_path = os.path.join(args.out, "queries_dev.tsv")
    n_queries = 0
    with open(queries_path, "w", encoding="utf-8") as f:
        for query in dataset.queries_iter():
            text = _doc_text(query).replace("\t", " ").replace("\n", " ")
            f.write(f"{query.query_id}\t{text}\n")
            n_queries += 1
    print(f"Wrote {n_queries} queries to {queries_path}")

    qrels_path = os.path.join(args.out, "qrels_dev.txt")
    n_qrels = 0
    with open(qrels_path, "w", encoding="utf-8") as f:
        for qrel in dataset.qrels_iter():
            f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
            n_qrels += 1
    print(f"Wrote {n_qrels} qrel lines to {qrels_path}")

    print(
        "\nDone. NOTE: the queries/qrels ir_datasets ships for this collection are "
        "released as your DEV set only. Course staff hold out a disjoint topic set, "
        "scored with the same corpus and the same harness, for the private leaderboard."
    )


if __name__ == "__main__":
    main()