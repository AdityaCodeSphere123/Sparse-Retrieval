"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
import math
from typing import Dict, List, Optional, Set, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    """Store reference to the loaded inverted index for VSM/Boolean scoring."""
    global _INDEX
    _INDEX = index


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its terms."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before boolean_search()")

    tokens = tokenize(query)
    if not tokens:
        return []

    mode = mode.lower()
    matching_docs: Optional[Set[str]] = None

    for term in tokens:
        doc_set = set(_INDEX.postings.get(term, {}).keys())
        if mode == "and":
            if matching_docs is None:
                matching_docs = doc_set
            else:
                matching_docs &= doc_set
            if not matching_docs:
                break
        elif mode == "or":
            if matching_docs is None:
                matching_docs = doc_set
            else:
                matching_docs |= doc_set

    return sorted(list(matching_docs)) if matching_docs else []


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by TF-IDF cosine similarity."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score()")

    tokens = tokenize(query)
    if not tokens or _INDEX.N == 0:
        return []

    q_tf: Dict[str, int] = {}
    for t in tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    # Query vector weights & norm
    q_weights: Dict[str, float] = {}
    q_norm_sq = 0.0

    for term, tf in q_tf.items():
        df = _INDEX.document_frequency(term)
        if df > 0:
            idf = math.log(_INDEX.N / df)
            weight = (1.0 + math.log(tf)) * idf
            q_weights[term] = weight
            q_norm_sq += weight * weight

    if not q_weights or q_norm_sq == 0.0:
        return []

    q_norm = math.sqrt(q_norm_sq)

    # Accumulate dot products and document norms
    doc_dots: Dict[str, float] = {}
    doc_norms_sq: Dict[str, float] = {}

    for term, q_w in q_weights.items():
        postings = _INDEX.postings.get(term, {})
        df = _INDEX.document_frequency(term)
        idf = math.log(_INDEX.N / df)

        for doc_id, tf in postings.items():
            doc_w = (1.0 + math.log(tf)) * idf
            doc_dots[doc_id] = doc_dots.get(doc_id, 0.0) + (q_w * doc_w)
            doc_norms_sq[doc_id] = doc_norms_sq.get(doc_id, 0.0) + (doc_w * doc_w)

    scores: List[Tuple[str, float]] = []
    for doc_id, dot in doc_dots.items():
        doc_norm = math.sqrt(doc_norms_sq[doc_id])
        if doc_norm > 0:
            sim = dot / (q_norm * doc_norm)
            scores.append((doc_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

