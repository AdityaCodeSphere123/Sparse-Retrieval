"""
submission/corpus_utils.py

Small, shared, non-graded utility for reading the corpus format used
throughout this assignment. You are welcome to use this as-is — reading a
JSONL file is plumbing, not the ranking logic the assignment grades — or
replace it if your index needs a different loading strategy (e.g.
streaming instead of loading everything into memory).
"""
import json
import os
from typing import Iterable, List, Tuple


def _stream_jsonl(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["doc_id"], obj["text"]


def stream_corpus(path: str) -> Iterable[Tuple[str, str]]:
    """Yield (doc_id, text) from a jsonl file, a directory, or ir_datasets.

    A directory may contain `corpus.jsonl`, one or more `*.jsonl` files, or
    one document per file (filename stem = doc_id). The assignment allows
    the held-out collection to arrive as a single directory.
    """
    if os.path.isdir(path):
        jsonl = os.path.join(path, "corpus.jsonl")
        if os.path.isfile(jsonl):
            yield from _stream_jsonl(jsonl)
            return
        jsonls = sorted(
            os.path.join(path, fn)
            for fn in os.listdir(path)
            if fn.endswith(".jsonl") and os.path.isfile(os.path.join(path, fn))
        )
        if jsonls:
            for fp in jsonls:
                yield from _stream_jsonl(fp)
            return
        for fn in sorted(os.listdir(path)):
            fp = os.path.join(path, fn)
            if not os.path.isfile(fp) or fn.startswith("."):
                continue
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                yield os.path.splitext(fn)[0], f.read()
        return

    if path.startswith("ir_datasets:") or (
        not os.path.exists(path) and "/" in path and not path.endswith(".jsonl")
    ):
        dataset_id = path.replace("ir_datasets:", "")
        import ir_datasets
        dataset = ir_datasets.load(dataset_id)
        for doc in dataset.docs_iter():
            text = doc.default_text() if hasattr(doc, "default_text") else getattr(doc, "text", "")
            yield doc.doc_id, text
        return

    yield from _stream_jsonl(path)


def load_corpus(path: str) -> List[Tuple[str, str]]:
    """Read a corpus and return a list of (doc_id, text) pairs in file order."""
    return list(stream_corpus(path))
