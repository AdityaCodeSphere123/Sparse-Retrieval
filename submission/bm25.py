"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
import math
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    """Store reference to the loaded inverted index for BM25 scoring."""
    global _INDEX
    _INDEX = index


def score(
    query: str,
    k: int,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 0.0,
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25 / BM25+ ranked, highest score first.

    delta: addition term for BM25+ (Lv & Zhai). Set delta=0.0 for standard BM25, delta=1.0 for BM25+.
    """
    if _INDEX is None:
        raise RuntimeError("bm25.build() must be called before bm25.score()")

    tokens = tokenize(query)
    if not tokens or _INDEX.N == 0 or _INDEX.avg_doc_len == 0:
        return []

    doc_scores: Dict[str, float] = {}

    for term in set(tokens):
        df = _INDEX.document_frequency(term)
        if df == 0:
            continue

        # Robertson-Sparck Jones IDF with +1 smoothing to remain positive
        idf = math.log(1.0 + (_INDEX.N - df + 0.5) / (df + 0.5))

        postings = _INDEX.postings.get(term, {})
        for doc_id, tf in postings.items():
            doc_len = _INDEX.doc_len.get(doc_id, _INDEX.avg_doc_len)
            len_ratio = doc_len / _INDEX.avg_doc_len
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * len_ratio))
            term_score = idf * (tf_norm + delta)

            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

    scores = list(doc_scores.items())
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

