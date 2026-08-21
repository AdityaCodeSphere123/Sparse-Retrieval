"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import gzip
import json
import os
import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization."""
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.doc_text: Dict[str, str] = {}  # doc_id -> raw text (handy for VSM/debugging)
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """Build the inverted index and corpus statistics from a list of (doc_id, text) pairs."""
        self.postings = {}
        self.doc_len = {}
        self.doc_text = {}
        self.N = len(corpus)
        total_tokens = 0

        for doc_id, text in corpus:
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            total_tokens += len(tokens)

            tf_map: Dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term, tf in tf_map.items():
                if term not in self.postings:
                    self.postings[term] = {}
                self.postings[term][doc_id] = tf

        self.avg_doc_len = (total_tokens / self.N) if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """
        Persist postings and statistics to `index_dir` as compressed JSON.

        The on-disk byte size is graded directly: aim to keep this small.
        Starting from this uncompressed baseline, you can reduce size by
        - using a compact postings encoding (e.g. gap encoding + variable-byte
          deltas for document IDs and term frequencies),
        - storing only what's needed for scoring (you can drop self.doc_text
          from the file if you don't use it at query time),
        - perhaps storing term frequencies as integers scaled by 100 or 1000
          and rounded, if that helps.
        The assignment spec's index-size-bonus section has more ideas.
        """
        os.makedirs(index_dir, exist_ok=True)
        data = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }
        index_file = os.path.join(index_dir, "index.json.gz")
        with gzip.open(index_file, "wt", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex from compressed disk representation in `index_dir`."""
        index = cls()
        index_file = os.path.join(index_dir, "index.json.gz")
        with gzip.open(index_file, "rt", encoding="utf-8") as f:
            data = json.load(f)

        index.postings = data["postings"]
        index.doc_len = data["doc_len"]
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        return index

