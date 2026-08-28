"""
submission/bm25.py — Okapi BM25 / BM25+ ranking.

    score(D,Q) = sum_i IDF(qi) * [ tf*(k1+1) / (tf + k1*(1-b+b*|D|/avgdl)) + delta ]
    IDF(qi)    = ln( 1 + (N - df + 0.5) / (df + 0.5) )

k1 and b are parameters, not hard-coded. Ties break as score desc, then
doc_id asc. k1 <= 0 or b outside [0, 1] is rejected rather than producing
a non-monotone score.
"""
import math
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    global _INDEX
    _INDEX = index


def score(
    query: str,
    k: int,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 0.0,
) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs, BM25/BM25+ ranked, highest first.

    delta: BM25+ lower bound (Lv & Zhai). 0.0 is standard BM25, 1.0 is BM25+.
    """
    if _INDEX is None:
        raise RuntimeError("bm25.build() must be called before bm25.score()")
    if k1 <= 0.0:
        raise ValueError("k1 must be > 0")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b must be in [0, 1]")

    tokens = tokenize(query)
    if not tokens or _INDEX.N == 0 or _INDEX.avg_doc_len == 0:
        return []

    doc_scores: Dict[str, float] = {}

    for term in set(tokens):
        df = _INDEX.document_frequency(term)
        if df == 0:
            continue

        idf = math.log(1.0 + (_INDEX.N - df + 0.5) / (df + 0.5))

        for doc_id, tf in _INDEX.postings.get(term, {}).items():
            doc_len = _INDEX.doc_len.get(doc_id, _INDEX.avg_doc_len)
            len_ratio = doc_len / _INDEX.avg_doc_len
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * len_ratio))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + idf * (tf_norm + delta)

    scores = list(doc_scores.items())
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:k]
