"""
harness/trec_io.py — reading/writing the plain-text formats used
throughout this assignment. Deliberately dependency-free (no pytrec_eval
requirement) so the harness runs anywhere.

Formats:
    corpus.jsonl    one JSON object per line: {"doc_id": ..., "text": ...}
    queries.tsv     "qid<TAB>query text" per line
    qrels.txt       TREC qrels format: "qid 0 doc_id relevance" per line
                    (whitespace-separated; the literal "0" is the TREC
                    qrels iteration column and is ignored)
    run file        TREC run format: "qid Q0 doc_id rank score run_tag"
                    (whitespace-separated), one line per (query, doc) pair

These are the same conventions trec_eval / pytrec_eval expect, so you can
cross-check this harness's numbers with the standard tools if you want.
"""
import json
from typing import Dict, List, Tuple


def read_corpus(path: str) -> List[Tuple[str, str]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((obj["doc_id"], obj["text"]))
    return docs


def read_queries(path: str) -> List[Tuple[str, str]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            qid, text = line.split("\t", 1)
            queries.append((qid, text))
    return queries


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Returns {qid: {doc_id: relevance}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Malformed qrels line (expected 4 fields): {line!r}")
            qid, _iter, doc_id, rel = parts
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def write_run(path: str, run: Dict[str, List[Tuple[str, float]]], run_tag: str = "run") -> None:
    """run: {qid: [(doc_id, score), ...]} already sorted best-first."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


def read_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Returns {qid: [(doc_id, score), ...]}, sorted by score descending
    (re-sorted here rather than trusted from the rank column, in case the
    file wasn't written by write_run)."""
    raw: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Malformed run line (expected 6 fields): {line!r}")
            qid, _q0, doc_id, _rank, score, _tag = parts
            raw.setdefault(qid, []).append((doc_id, float(score)))
    for qid in raw:
        raw[qid].sort(key=lambda pair: pair[1], reverse=True)
    return raw
