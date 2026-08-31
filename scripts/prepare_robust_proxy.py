#!/usr/bin/env python
"""Build a Robust04-shaped JSONL proxy (long docs, short graded queries).

Default: MS MARCO Document v1 subsampled to --n-docs, keeping every document
that appears in TREC DL 2019/2020 qrels, then filling the rest with a stable
hash sample. Fallback: BEIR Touché-2020 (~380K, long, short queries).

Does not download TREC Disks 4/5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _doc_text(obj) -> str:
    if hasattr(obj, "default_text"):
        try:
            t = obj.default_text()
            if t:
                return t
        except Exception:
            pass
    parts = []
    for attr in ("title", "body", "text"):
        t = getattr(obj, attr, None)
        if t:
            parts.append(str(t))
    return "\n".join(parts)


def _stable_keep(doc_id: str, rate: float) -> bool:
    h = hashlib.md5(doc_id.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "little") / float(2**64)
    return v < rate


def _write_jsonl_queries_qrels(out_dir: str, queries, qrels, qrel_filter=None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    qpath = os.path.join(out_dir, "queries_dev.tsv")
    nq = 0
    with open(qpath, "w", encoding="utf-8") as f:
        for q in queries:
            text = _doc_text(q).replace("\t", " ").replace("\n", " ")
            f.write(f"{q.query_id}\t{text}\n")
            nq += 1
    rpath = os.path.join(out_dir, "qrels_dev.txt")
    nr = 0
    with open(rpath, "w", encoding="utf-8") as f:
        for r in qrels:
            if qrel_filter is not None and str(r.doc_id) not in qrel_filter:
                continue
            f.write(f"{r.query_id} 0 {r.doc_id} {r.relevance}\n")
            nr += 1
    print(f"Wrote {nq} queries, {nr} qrel lines")


def prepare_msmarco(out_dir: str, n_docs: int) -> None:
    import ir_datasets

    judged = set()
    query_rows = []
    qrel_rows = []
    for dsid in (
        "msmarco-document/trec-dl-2019/judged",
        "msmarco-document/trec-dl-2020/judged",
    ):
        try:
            ds = ir_datasets.load(dsid)
        except Exception:
            ds = ir_datasets.load(dsid.rsplit("/", 1)[0])
        for q in ds.queries_iter():
            query_rows.append(q)
        for r in ds.qrels_iter():
            qrel_rows.append(r)
            judged.add(str(r.doc_id))
    print(f"Judged documents in DL19/20 qrels: {len(judged)}")

    docs = ir_datasets.load("msmarco-document")
    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    kept = 0
    extra_budget = max(0, n_docs - len(judged))
    extra_rate = 0.20
    extras = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in docs.docs_iter():
            did = str(doc.doc_id)
            take = did in judged
            if not take:
                if kept >= n_docs or extras >= extra_budget:
                    continue
                if not _stable_keep(did, extra_rate):
                    continue
                extras += 1
            f.write(json.dumps({"doc_id": did, "text": _doc_text(doc)}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50000 == 0:
                print(f"  wrote {kept} docs...", flush=True)
    print(f"Wrote {kept} documents to {corpus_path} (extras={extras})")
    _write_jsonl_queries_qrels(out_dir, query_rows, qrel_rows, qrel_filter=None)


def prepare_ir_dataset(dsid: str, out_dir: str, n_docs: int) -> None:
    import ir_datasets

    ds = ir_datasets.load(dsid)
    os.makedirs(out_dir, exist_ok=True)
    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    kept = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in ds.docs_iter():
            f.write(json.dumps({"doc_id": str(doc.doc_id), "text": _doc_text(doc)}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= n_docs:
                break
    print(f"Wrote {kept} documents to {corpus_path}")
    qds = ds
    has_qrels = getattr(ds, "has_qrels", lambda: False)()
    if not has_qrels:
        try:
            qds = ir_datasets.load(dsid + "/test")
        except Exception:
            qds = ds
    if not getattr(qds, "has_qrels", lambda: False)():
        raise SystemExit(f"{dsid} has no qrels (tried {dsid}/test)")
    _write_jsonl_queries_qrels(out_dir, qds.queries_iter(), qds.qrels_iter())


BEIR_SOURCES = {
    "touche": "beir/webis-touche2020",
    "fiqa": "beir/fiqa",
    "scifact": "beir/scifact",
    "nfcorpus": "beir/nfcorpus",
    "arguana": "beir/arguana",
    "antique": "antique",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("msmarco",) + tuple(BEIR_SOURCES),
        default="touche",
    )
    parser.add_argument("--n-docs", type=int, default=500_000)
    parser.add_argument(
        "--out",
        default="",
        help="output directory (default data/proxy_<source>)",
    )
    args = parser.parse_args()
    try:
        import ir_datasets  # noqa: F401
    except ImportError:
        sys.exit("pip install ir_datasets")

    out = args.out or os.path.join(ROOT, "data", f"proxy_{args.source}")
    print(f"Preparing {args.source} proxy -> {out}")
    if args.source == "msmarco":
        prepare_msmarco(out, args.n_docs)
    else:
        prepare_ir_dataset(BEIR_SOURCES[args.source], out, args.n_docs)


if __name__ == "__main__":
    main()
