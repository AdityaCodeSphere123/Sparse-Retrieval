"""
submission/custom_scorer.py — combined scorer (pure-Python path).

    combined(d) = w_bm25 * z(BM25(d)) + w_vsm * z(cosine(d))

z() is min-max normalisation over each signal's candidate pool. BM25 is
unbounded; cosine is in [0, 1], so blending raw scores would collapse to
one signal. A document in only one pool scores 0 on the other after
normalisation ("this signal did not rank you").
"""
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None

POOL = 200
W_BM25 = 0.85
W_VSM = 0.15
K1, B, DELTA = 0.9, 0.4, 0.0


def build(index: InvertedIndex) -> None:
    """Query-time setup only — nothing extra is persisted to index_dir."""
    global _INDEX
    _INDEX = index
    bm25.build(index)
    boolean_vsm.build(index)


def _minmax(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {d: 1.0 for d, _ in pairs}
    return {d: (s - lo) / (hi - lo) for d, s in pairs}


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs from the fused ranking."""
    if _INDEX is None:
        raise RuntimeError("custom_scorer.build() must be called before score()")

    b_pairs = bm25.score(query, POOL, k1=K1, b=B, delta=DELTA)
    v_pairs = boolean_vsm.vsm_score(query, POOL)
    if not b_pairs and not v_pairs:
        return []

    b_norm = _minmax(b_pairs)
    v_norm = _minmax(v_pairs)

    fused: Dict[str, float] = {}
    for doc_id in set(b_norm) | set(v_norm):
        fused[doc_id] = W_BM25 * b_norm.get(doc_id, 0.0) + W_VSM * v_norm.get(doc_id, 0.0)

    out = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return out[:k]
