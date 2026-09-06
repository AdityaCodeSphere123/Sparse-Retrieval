"""
harness/metrics.py — the exact metrics your leaderboard score is computed
from: nDCG@10 (primary), MAP@10 (tie-break), plus MRR and P@k for your own
error analysis. This file is shared, readable, and unit-tested (see
tests/test_metrics.py) precisely so there is no ambiguity about how
scoring works — the same code path is used for the public dev leaderboard
and (with a different, undisclosed topic/qrels file) the private held-out
leaderboard at grading time.

Conventions, matching standard trec_eval behaviour:
  - nDCG uses graded relevance and the (2^rel - 1) gain function.
  - MAP, MRR, and P@k use binarised relevance: a document counts as
    relevant if its qrels judgment is > 0.
  - Unjudged documents (not present in qrels for that query) are treated
    as non-relevant (relevance 0), not skipped.

A note on "MAP@10" specifically: `retrieve(query, k)` only ever returns up
to k documents, so this cannot be textbook unbounded MAP — full MAP
requires knowing precision at every rank down to wherever the last
relevant document appears, which in turn requires ranking (or at least
scoring) the whole corpus, not just returning k items. What's computed
here is Average Precision over the top-10 of what you returned, still
normalised by the *true* total number of relevant documents from the
qrels (not just the ones you found) — so a query with more relevant
documents than fit in the top 10 can never reach AP=1.0, by design. This
matches how `trec_eval` itself behaves on a truncated run file; it is a
real, standard, well-defined metric (conventionally written "MAP@k"), just
not the same number you'd get if you ranked and returned everything.

A note on duplicate doc_ids: a ranking must never contain the same doc_id
twice. `evaluate_run()` (below) deduplicates every ranked list — keeping
only each doc_id's first (best-ranked) occurrence — before computing any
metric, precisely because the individual metric functions above do not
protect themselves against this. Without that step, a run that repeats a
single known-relevant document k times would have its relevance counted
k times over: nDCG@10 and MAP@10 are both unbounded above by construction
(they sum a per-rank contribution with no cap), so this isn't a small
distortion — a query with one relevant document, graded relevance 3,
"ranked" as the same doc_id ten times in a row scores nDCG@10 ≈ 4.54 and
MAP@10 = 10.0, both wildly past their theoretical maximum of 1.0. See
tests/test_metrics.py for the worked example. `harness/run_harness.py`
separately rejects a live `retrieve()` call that returns duplicate
doc_ids outright (a RUNTIME_ERROR, treated as a submission bug); the dedup
here is defense-in-depth so the same exploit can't slip in through a
static run file (a baseline comparison run, or `--baseline-run`) that
never went through that check.
"""
import math
from typing import Dict, List, Tuple

RunType = Dict[str, List[Tuple[str, float]]]  # qid -> [(doc_id, score), ...], best first
QrelsType = Dict[str, Dict[str, int]]  # qid -> {doc_id: relevance}


def _relevance(qrels_for_q: Dict[str, int], doc_id: str) -> int:
    return qrels_for_q.get(doc_id, 0)


def _dedup_keep_first(doc_ids: List[str]) -> List[str]:
    """Keep each doc_id's first (best-ranked) occurrence only, dropping
    any later repeats. See the module docstring's "note on duplicate
    doc_ids" for why this exists — every metric below sums a per-rank
    contribution with no upper cap, so letting the same relevant document
    count more than once lets nDCG@10/MAP@10 blow past 1.0."""
    seen = set()
    out = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def dcg_at_k(gains: List[int], k: int) -> float:
    """gains: relevance values in ranked order (already truncated or not;
    only the first k are used). DCG = sum_i (2^rel_i - 1) / log2(i + 1),
    1-indexed."""
    dcg = 0.0
    for i, rel in enumerate(gains[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int = 10) -> float:
    if not qrels_for_q:
        return 0.0
    gains = [_relevance(qrels_for_q, doc_id) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(qrels_for_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    """Average Precision over `ranked_doc_ids`, exactly as given — this
    function does NOT truncate internally. The caller (evaluate_run,
    below) passes an already rank-10-truncated list to produce "MAP@10",
    matching nDCG@10's cutoff; pass a longer or shorter list yourself if
    you want AP at a different depth. Normalises by the *true* total
    relevant-document count from qrels_for_q, not by however many of
    those were found in `ranked_doc_ids` — so AP is capped below 1.0 for
    any query with more relevant documents than the list you passed in.
    """
    n_relevant = sum(1 for rel in qrels_for_q.values() if rel > 0)
    if n_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            hits += 1
            precision_sum += hits / i
    return precision_sum / n_relevant


def reciprocal_rank(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int]) -> float:
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if _relevance(qrels_for_q, doc_id) > 0:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked_doc_ids: List[str], qrels_for_q: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if _relevance(qrels_for_q, doc_id) > 0)
    return hits / k


def evaluate_run(run: RunType, qrels: QrelsType, k: int = 10) -> Dict:
    """Evaluate `run` against `qrels`. Returns a dict with per-query
    metrics and corpus-level (macro) averages. Queries present in `run`
    but absent from `qrels` are skipped with a warning key so they don't
    silently zero out your average; queries present in `qrels` but
    missing from `run` are scored as 0 across all metrics (you didn't
    answer them)."""
    per_query = {}
    skipped_no_qrels = []

    all_qids = set(qrels.keys()) | set(run.keys())
    for qid in sorted(all_qids):
        if qid not in qrels:
            skipped_no_qrels.append(qid)
            continue
        qrels_for_q = qrels[qid]
        ranked_doc_ids = [doc_id for doc_id, _score in run.get(qid, [])]
        ranked_doc_ids = _dedup_keep_first(ranked_doc_ids)
        per_query[qid] = {
            "ndcg@10": ndcg_at_k(ranked_doc_ids, qrels_for_q, k=10),
            # Fixed at 10 (not the `k` argument to this function) to match
            # nDCG@10's cutoff and keep "MAP@10" an honest label regardless
            # of what --k the harness happened to be invoked with — see
            # average_precision()'s docstring.
            "map@10": average_precision(ranked_doc_ids[:10], qrels_for_q),
            "mrr": reciprocal_rank(ranked_doc_ids, qrels_for_q),
            f"p@{k}": precision_at_k(ranked_doc_ids, qrels_for_q, k),
            "num_ranked": len(ranked_doc_ids),
        }

    n = len(per_query)
    if n == 0:
        aggregate = {"ndcg@10": 0.0, "map@10": 0.0, "mrr": 0.0, f"p@{k}": 0.0}
    else:
        aggregate = {
            "ndcg@10": sum(m["ndcg@10"] for m in per_query.values()) / n,
            "map@10": sum(m["map@10"] for m in per_query.values()) / n,
            "mrr": sum(m["mrr"] for m in per_query.values()) / n,
            f"p@{k}": sum(m[f"p@{k}"] for m in per_query.values()) / n,
        }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "num_queries_scored": n,
        "skipped_no_qrels": skipped_no_qrels,
    }
