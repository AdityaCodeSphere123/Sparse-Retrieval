"""
submission/boolean_vsm.py — Boolean retrieval + TF-IDF cosine VSM.

vsm_score() uses the full-document L2 norm (InvertedIndex.doc_norm), not
the query-term subspace. Query-only norms are not cosine:

    single-term query  -> every matching document scored exactly 1.0
    multi-term query   -> documents missing a query term were over-rewarded
                          (a doc matching 1 of 2 terms scored ~0.707)

Weighting is lnc.ltc-style — (1 + log tf) * ln(N/df) on both sides, then
cosine — controlled by indexer.USE_LOG_TF so index norms and query weights
cannot drift apart. Set USE_LOG_TF = False for plain w = tf * log(N/df).
"""
import math
from typing import Dict, List, Optional, Set, Tuple

from submission.indexer import InvertedIndex, tf_weight, tokenize, vsm_idf

_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    global _INDEX
    _INDEX = index


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Unranked doc_ids matching `query` as a conjunction or disjunction."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before boolean_search()")

    mode = mode.lower()
    if mode not in ("and", "or"):
        raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

    tokens = tokenize(query)
    if not tokens:
        return []

    matching: Optional[Set[str]] = None
    for term in tokens:
        doc_set = set(_INDEX.postings.get(term, {}).keys())
        if matching is None:
            matching = doc_set
        elif mode == "and":
            matching &= doc_set
        else:
            matching |= doc_set
        if mode == "and" and not matching:
            return []

    return sorted(matching) if matching else []


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs ranked by TF-IDF cosine similarity."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score()")

    tokens = tokenize(query)
    if not tokens or _INDEX.N == 0:
        return []

    q_tf: Dict[str, int] = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_weights: Dict[str, float] = {}
    q_norm_sq = 0.0
    for term, tf in q_tf.items():
        df = _INDEX.document_frequency(term)
        if df == 0:
            continue
        w = tf_weight(tf) * vsm_idf(df, _INDEX.N)
        if w == 0.0:
            continue
        q_weights[term] = w
        q_norm_sq += w * w

    if not q_weights or q_norm_sq <= 0.0:
        return []
    q_norm = math.sqrt(q_norm_sq)

    dots: Dict[str, float] = {}
    for term, q_w in q_weights.items():
        idf = vsm_idf(_INDEX.document_frequency(term), _INDEX.N)
        for doc_id, tf in _INDEX.postings.get(term, {}).items():
            dots[doc_id] = dots.get(doc_id, 0.0) + q_w * (tf_weight(tf) * idf)

    if not _INDEX.doc_norm:
        _INDEX._compute_doc_norms()

    scores: List[Tuple[str, float]] = []
    for doc_id, dot in dots.items():
        d_norm = _INDEX.doc_norm.get(doc_id, 0.0)
        if d_norm > 0.0:
            scores.append((doc_id, dot / (q_norm * d_norm)))

    # Score desc, then doc_id asc — stable ties for reproducibility.
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:k]
