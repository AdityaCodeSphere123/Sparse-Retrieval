"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
    load_index(index_dir: str) -> None
    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]

Scoring is a BM25 first pass plus optional RM3 pseudo-relevance feedback.
All of that lives in the C++ extension (`submission/cpp_engine.cpp`); the
pure-Python path below is a correctness fallback for when the extension is
not built.
"""
import os
from typing import List, Optional, Tuple

from submission import bm25, custom_scorer
from submission.corpus_utils import load_corpus, stream_corpus
from submission.indexer import InvertedIndex

try:
    import cpp_engine
    HAS_CPP = True
except ImportError:
    try:
        from submission import cpp_engine
        HAS_CPP = True
    except ImportError:
        HAS_CPP = False

# ---------------------------------------------------------------------------
# All query-time tunables in one place. Sweep these on the dev set and report
# the sweep (assignment Section 8); do NOT tune them on the held-out topics.
#
#   k1, b     BM25 term-frequency saturation / length normalisation
#   delta     BM25+ lower-bound term (Lv & Zhai); 0.0 gives standard BM25
#   use_rm3   pseudo-relevance feedback on/off
#   fb_docs   number of first-pass documents treated as relevant
#   fb_terms  number of expansion terms kept from the relevance model
#   alpha     weight on the original query vs. the expansion (1.0 = no expansion)
#   mu        Dirichlet smoothing for P(w|D) in the relevance model
# ---------------------------------------------------------------------------

PARAMS = dict(
    k1=2.5,
    b=0.5,
    delta=0.0,
    use_rm3=True,
    fb_docs=5,
    fb_terms=20,
    alpha=0.7,
    mu=1000.0,
)

# The ANALYSIS CHAIN (stemmer + stopwords) is not here because it is not a
# query-time choice: it is baked into the index at build time and read back
# out of meta.bin, so query tokenisation always matches index tokenisation.
# Set it with environment variables when build_index() runs:
#
#     IR_STEMMER=2   0 = none, 1 = light S-stemmer, 2 = Porter   (default 2)
#     IR_STOPWORDS=1 0 = keep stopwords, 1 = remove them          (default 1)
#
# `python -m scripts.tune --sweep analysis` rebuilds and scores all four
# combinations for you.

_INDEX: Optional[InvertedIndex] = None
_CPP_ENGINE: Optional[object] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Build the index and persist everything retrieve() needs into index_dir."""
    os.makedirs(index_dir, exist_ok=True)

    # Resolve to a single streamable .jsonl file when possible, so a
    # directory-delivered corpus still goes through the C++ engine's
    # streaming path instead of being materialised twice in Python.
    stream_path = corpus_path
    if os.path.isdir(corpus_path):
        candidate = os.path.join(corpus_path, "corpus.jsonl")
        if os.path.isfile(candidate):
            stream_path = candidate

    if HAS_CPP:
        engine = cpp_engine.CPPEngine()
        if os.path.isfile(stream_path):
            engine.build_index_from_file(stream_path, index_dir)
        else:
            # Multi-file directory / ir_datasets specifier: no single
            # streamable file, fall back to materialising the corpus.
            engine.build_index(load_corpus(corpus_path), index_dir)
    else:
        index = InvertedIndex()
        index.build(stream_corpus(corpus_path))
        index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct query-time state from index_dir, and only from index_dir."""
    global _INDEX, _CPP_ENGINE
    if HAS_CPP:
        _CPP_ENGINE = cpp_engine.CPPEngine()
        _CPP_ENGINE.load_index(index_dir)
    else:
        _INDEX = InvertedIndex.load(index_dir)
        custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs, highest score first."""
    # PARAMS is the competition configuration (dev-tuned; see comment above).
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve(query, k, **PARAMS)

    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return custom_scorer.score(query, k)


def retrieve_bm25_only(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Plain BM25+ with no expansion.

    Not part of the harness interface — this is for the ablation table in the
    report (Boolean/VSM vs. BM25 vs. full system) and for the k1/b sweep,
    where you want to isolate BM25 from the RM3 stage.
    """
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve_bm25(
            query, k, PARAMS["k1"], PARAMS["b"], PARAMS["delta"]
        )
    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=PARAMS["k1"], b=PARAMS["b"], delta=PARAMS["delta"])