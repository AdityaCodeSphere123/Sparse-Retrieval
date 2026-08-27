"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
    load_index(index_dir: str) -> None
    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]

Scoring is a BM25+ first pass, RM3 pseudo-relevance feedback, then a
Sequential Dependence Model proximity rerank of the candidate pool. All of
that lives in the C++ extension (src/cpp_engine.cpp); the pure-Python path
below is a correctness fallback for when the extension is not built.
"""
import os
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm
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
#   use_sdm   proximity rerank on/off
#   lambda_o  weight on ordered (adjacent bigram) evidence
#   lambda_u  weight on unordered-window evidence
#   window_u  unordered window width in tokens
# ---------------------------------------------------------------------------
PARAMS = dict(
    k1=1.2,
    b=0.75,
    delta=1.0,
    use_rm3=True,
    fb_docs=10,
    fb_terms=25,
    alpha=0.6,
    mu=1000.0,
    use_sdm=True,
    sdm_rerank_depth=200,
    lambda_o=0.10,
    lambda_u=0.05,
    window_u=8,
    sdm_normalise=True,
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

    if HAS_CPP:
        engine = cpp_engine.CPPEngine()
        # Streaming straight off the file avoids materialising the whole corpus
        # in Python memory; fall back to the list form for ir_datasets specifiers.
        if os.path.isfile(corpus_path):
            engine.build_index_from_file(corpus_path, index_dir)
        else:
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
        bm25.build(_INDEX)
        boolean_vsm.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs, highest score first."""
    # NOTE: PARAMS below are the textbook defaults and are almost certainly
    # NOT the right values for this corpus. Run scripts/tune.py and paste the
    # winning configuration in before submitting.
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve(query, k, **PARAMS)

    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=PARAMS["k1"], b=PARAMS["b"], delta=PARAMS["delta"])


def retrieve_bm25_only(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Plain BM25+ with no expansion and no proximity rerank.

    Not part of the harness interface — this is for the ablation table in the
    report (Boolean/VSM vs. BM25 vs. full system) and for the k1/b sweep,
    where you want to isolate BM25 from the RM3 and SDM stages.
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