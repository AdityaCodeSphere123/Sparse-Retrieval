"""
tests/test_metrics.py — unit tests for harness/metrics.py itself, with
hand-computable examples. These exist so you can trust the scoring code
(it's the same code that computes your grade) without having to take our
word for it — every expected value below is derived by hand in the
comments, not just asserted against the implementation's own output.
"""
import math

from harness.metrics import (
    average_precision,
    dcg_at_k,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


def test_ndcg_perfect_ranking_is_one():
    qrels = {"d1": 3, "d2": 2, "d3": 1, "d4": 0}
    ranked = ["d1", "d2", "d3", "d4"]  # already in relevance order
    assert ndcg_at_k(ranked, qrels, k=10) == 1.0


def test_ndcg_no_relevant_docs_retrieved_is_zero():
    qrels = {"d1": 3, "d2": 2}
    ranked = ["d9", "d10"]  # neither judged relevant (in fact unjudged)
    assert ndcg_at_k(ranked, qrels, k=10) == 0.0


def test_ndcg_worked_example():
    # Ranked relevances (in retrieval order): [2, 0, 1]
    # DCG = (2^2-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
    #     = 3/1 + 0/1.585 + 1/2
    #     = 3 + 0 + 0.5 = 3.5
    # Ideal order is [2, 1, 0] (sorted descending):
    # IDCG = 3/1 + 1/log2(3) + 0/log2(4) = 3 + 0.6309... = 3.6309...
    qrels = {"a": 2, "b": 0, "c": 1}
    ranked = ["a", "b", "c"]
    expected_dcg = 3.0 + 0.0 + 0.5
    expected_idcg = 3.0 + 1 / math.log2(3) + 0.0
    expected_ndcg = expected_dcg / expected_idcg
    assert math.isclose(ndcg_at_k(ranked, qrels, k=10), expected_ndcg, rel_tol=1e-9)


def test_dcg_at_k_matches_formula_directly():
    gains = [3, 2, 1, 0]
    expected = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(gains, start=1))
    assert math.isclose(dcg_at_k(gains, 10), expected, rel_tol=1e-9)


def test_average_precision_worked_example():
    # 2 relevant docs total, found at ranks 1 and 3.
    # AP = (precision@1 + precision@3) / 2 = (1/1 + 2/3) / 2 = 0.8333...
    qrels = {"rel1": 1, "irrelevant": 0, "rel2": 1}
    ranked = ["rel1", "irrelevant", "rel2"]
    assert math.isclose(average_precision(ranked, qrels), (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_average_precision_no_relevant_in_qrels_is_zero():
    assert average_precision(["a", "b"], {"a": 0, "b": 0}) == 0.0


def test_reciprocal_rank_worked_example():
    qrels = {"a": 0, "b": 0, "c": 1}
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, qrels) == 1.0 / 3


def test_reciprocal_rank_no_relevant_found_is_zero():
    qrels = {"a": 1}
    ranked = ["b", "c"]
    assert reciprocal_rank(ranked, qrels) == 0.0


def test_evaluate_run_rejects_score_inflation_from_duplicate_doc_ids():
    # A single relevant document (relevance 3), "ranked" as the same
    # doc_id ten times in a row. Before evaluate_run() deduplicated ranked
    # lists, this scored nDCG@10 ~= 4.54 and MAP@10 = 10.0 — both were
    # supposed to be capped at 1.0, since a run is only allowed to count
    # each doc_id's relevance once. A submission could otherwise dominate
    # the leaderboard just by repeating whichever single relevant doc it
    # happened to find, regardless of ranking quality.
    qrels = {"q1": {"d1": 3}}
    run_exploit = {"q1": [("d1", float(10 - i)) for i in range(10)]}
    result = evaluate_run(run_exploit, qrels, k=10)

    assert result["aggregate"]["ndcg@10"] == 1.0
    assert result["aggregate"]["map@10"] == 1.0
    assert result["per_query"]["q1"]["num_ranked"] == 1  # 10 slots, 1 unique doc


def test_map_at_10_is_truncated_to_top_10_and_normalised_by_full_relevant_count():
    # 15 relevant docs total (qrels), but retrieve() only ever returns up
    # to k results, and evaluate_run() only scores the top 10 of those for
    # "map@10" — matching nDCG@10's cutoff, and making the label accurate
    # regardless of what --k the harness was invoked with. This is exactly
    # why it's called MAP@10, not plain MAP: full MAP would need precision
    # at every rank where a relevant doc appears, which needs the whole
    # collection ranked, not just a top-k list.
    qrels_for_q = {f"r{i}": 1 for i in range(1, 16)}  # r1..r15, all relevant
    ranked = ["r1", "i1", "i2", "i3", "r2", "i4", "i5", "i6", "i7", "r3", "r4", "i8"]
    # Top 10 = [r1, i1, i2, i3, r2, i4, i5, i6, i7, r3] -> hits at ranks 1, 5, 10.
    # r4 sits at rank 11 and must NOT count toward MAP@10.
    expected_map_at_10 = (1 / 1 + 2 / 5 + 3 / 10) / 15

    qrels = {"q1": qrels_for_q}
    run = {"q1": [(doc_id, float(len(ranked) - i)) for i, doc_id in enumerate(ranked)]}
    result = evaluate_run(run, qrels, k=10)

    assert math.isclose(result["per_query"]["q1"]["map@10"], expected_map_at_10, rel_tol=1e-9)
    assert math.isclose(result["aggregate"]["map@10"], expected_map_at_10, rel_tol=1e-9)

    # Sanity check: if you (incorrectly) didn't truncate, r4 at rank 11
    # would also contribute and the value would come out higher.
    untruncated_ap = average_precision(ranked, qrels_for_q)
    assert untruncated_ap > expected_map_at_10


def test_precision_at_k_worked_example():
    qrels = {"a": 1, "b": 0, "c": 1, "d": 0}
    ranked = ["a", "b", "c", "d"]
    # top 2: a (rel), b (not) -> 1/2
    assert precision_at_k(ranked, qrels, k=2) == 0.5
    # top 4: a, c relevant out of 4 -> 2/4
    assert precision_at_k(ranked, qrels, k=4) == 0.5
