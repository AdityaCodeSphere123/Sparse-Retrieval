"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in the order build_index() saw them, ignoring the query
entirely — wired up below. It actually persists to disk and reloads
correctly, so it exercises the full build -> disk -> fresh process -> load
-> query path end-to-end from your very first commit. Its scores will be
close to zero; replace the logic, but keep the same
persist-in-build / reconstruct-in-load shape.
"""
import json
import os
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm

from submission.corpus_utils import load_corpus, stream_corpus
from submission.indexer import InvertedIndex

try:
    import cpp_engine
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

_INDEX: Optional[InvertedIndex] = None
_CPP_ENGINE: Optional[object] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build index using C++ engine if available (or Python fallback) with streaming."""
    if HAS_CPP:
        engine = cpp_engine.CPPEngine()
        corpus = load_corpus(corpus_path)
        engine.build_index(corpus, index_dir)
    else:
        index = InvertedIndex()
        index.build(stream_corpus(corpus_path))
        index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct index from `index_dir` using C++ engine (or Python fallback)."""
    global _INDEX, _CPP_ENGINE
    if HAS_CPP:
        _CPP_ENGINE = cpp_engine.CPPEngine()
        _CPP_ENGINE.load_index(index_dir)
    else:
        _INDEX = InvertedIndex.load(index_dir)
        bm25.build(_INDEX)
        boolean_vsm.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by BM25+."""
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve(query, k, k1=1.2, b=0.75, delta=1.0)
    else:
        if _INDEX is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return bm25.score(query, k, k1=1.2, b=0.75, delta=1.0)
