"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
    load_index(index_dir: str) -> None
    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]

Scoring is a two-pass sparse pipeline, all of it inside the C++ extension
(cpp_engine.cpp):

  1. BM25F+ over an inverted index. "F" because the leading tokens of a
     document are treated as a title field and scored with their own weight;
     "+" because a matched term gets a lower-bounded contribution (Lv & Zhai).
  2. RM3 pseudo-relevance feedback: build a relevance model from the top
     first-pass documents, interpolate it with the original query, re-score.

The pure-Python path below is a correctness fallback for when the extension
is not built; it is plain BM25 and scores lower.
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
#   k1, b        BM25 term-frequency saturation / length normalisation
#   delta        BM25+ lower-bound term (Lv & Zhai); 0.0 gives standard BM25
#   w_title      BM25F field weight: how much a title occurrence adds to the
#                effective term frequency BEFORE saturation. Swept to 0.0 —
#                title_bonus turned out to carry the whole field signal.
#   title_bonus  flat IDF-weighted credit for a query term occurring in the
#                title at all, added AFTER saturation. This is the field
#                signal that actually pays: see the C++ scoring loop.
#   use_rm3      pseudo-relevance feedback on/off
#   fb_docs      number of first-pass documents treated as relevant
#   fb_terms     number of expansion terms kept from the relevance model
#   alpha        weight on the original query vs. the expansion (1.0 = none)
#   mu           Dirichlet smoothing for P(w|D) in the relevance model
#
# These were chosen with scripts/tune_cv.py (greedy coordinate ascent under
# 5-fold cross-validation, taking plateau centres rather than the argmax) —
# see the report. Build-time settings live in cpp_engine.cpp's BuildCfg and
# are swept with scripts/sweep_build.py.
# ---------------------------------------------------------------------------
PARAMS = dict(
    k1=2.2,
    b=0.6,
    delta=0.8,
    w_title=0.0,
    title_bonus=0.7,
    use_rm3=True,
    fb_docs=30,
    fb_terms=30,
    alpha=0.65,
    mu=5000.0,
)

# The ANALYSIS CHAIN (stemmer + stopwords) is not here because it is not a
# query-time choice: it is baked into the index at build time and read back
# out of meta.bin, so query tokenisation always matches index tokenisation.
# Set it with environment variables when build_index() runs:
#
#     IR_STEMMER=2   0 = none, 1 = light S-stemmer, 2 = Porter   (default 2)
#     IR_STOPWORDS=1 0 = keep stopwords, 1 = remove them          (default 1)
#
# `python -m scripts.sweep_build --var IR_STEMMER --values 0,1,2` rebuilds and
# scores each setting for you; the same script sweeps IR_TITLE_WINDOW,
# IR_FWD_TERMS (default 24) and IR_RM3_DF_FRAC, which are baked in at build.

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
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve(query, k, **PARAMS)

    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=PARAMS["k1"], b=PARAMS["b"], delta=PARAMS["delta"])


def retrieve_bm25_only(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """First-pass BM25F+ only, with no relevance feedback.

    Not part of the harness interface — this is for the ablation table in the
    report (Boolean/VSM vs. BM25 vs. full system) and for the k1/b sweep,
    where you want to isolate first-pass scoring from the RM3 stage.
    """
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve_bm25(
            query, k, PARAMS["k1"], PARAMS["b"], PARAMS["delta"],
            PARAMS["w_title"], PARAMS["title_bonus"]
        )
    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=PARAMS["k1"], b=PARAMS["b"], delta=PARAMS["delta"])