"""
harness/leaderboard.py — combine raw metrics into the weighted leaderboard
score defined in the assignment (Section 7):

    70% * nDCG@10  +  10% * MAP@10
  + efficiency bonus/penalty for build time & query latency (up to ±10%)
  + index-size score for your on-disk index footprint (0-10%)

Both the efficiency modifier and the index-size score are defined
*relative to the class* (your numbers vs. the class median across all
submissions), so a single team running this harness locally cannot
compute either in isolation. `compute_provisional_score()` below reports
the 80%-weight nDCG@10/MAP@10 component plus your raw timings and index
size, clearly labelled "provisional" — course staff apply the other two
components when aggregating the full class leaderboard (see
instructor-tools/aggregate_leaderboard.py).
"""
from typing import Dict, Optional

NDCG_WEIGHT = 0.70
MAP_WEIGHT = 0.10
MAX_EFFICIENCY_WEIGHT = 0.10
MAX_INDEX_SIZE_WEIGHT = 0.10


def compute_provisional_score(aggregate_metrics: Dict[str, float]) -> float:
    """aggregate_metrics: the `aggregate` dict from harness.metrics.evaluate_run
    (must contain "ndcg@10" and "map@10"). Returns the 80%-weight portion of
    the leaderboard score; the remaining ±10% efficiency modifier and 0-10%
    index-size score are applied at class-wide aggregation time."""
    return NDCG_WEIGHT * aggregate_metrics["ndcg@10"] + MAP_WEIGHT * aggregate_metrics["map@10"]


def efficiency_modifier(
    your_index_build_s: float,
    your_mean_query_latency_s: float,
    class_median_index_build_s: float,
    class_median_query_latency_s: float,
) -> float:
    """A simple, transparent efficiency modifier in [-MAX_EFFICIENCY_WEIGHT,
    +MAX_EFFICIENCY_WEIGHT]: reward being faster than the class median on
    both index build time and mean query latency, penalise being slower,
    capped at the stated maximum swing. Course staff may recalibrate the
    exact curve when aggregating the real class leaderboard; this
    implementation is the reference default.
    """
    def relative_speed(yours: float, median: float) -> float:
        if median <= 0:
            return 0.0
        # +1 if you're arbitrarily fast, -1 if you're >=3x slower than median,
        # linear in between.
        ratio = yours / median
        return max(-1.0, min(1.0, (1.0 - ratio) / 2.0 + 0.5)) * 2 - 1

    build_component = relative_speed(your_index_build_s, class_median_index_build_s)
    query_component = relative_speed(your_mean_query_latency_s, class_median_query_latency_s)
    combined = (build_component + query_component) / 2.0
    return combined * MAX_EFFICIENCY_WEIGHT


def index_size_score(your_index_bytes: float, class_median_index_bytes: float) -> float:
    """A standalone 0-to-MAX_INDEX_SIZE_WEIGHT component (not a ± modifier
    like efficiency_modifier — this is its own weighted piece of the
    leaderboard score, per the assignment's Section 7 weighting): full
    marks at half the class median size or smaller, zero marks at double
    the class median or larger, linear in between. Rewards genuine
    compression (delta/VByte-encoded postings, dropping redundant
    per-document text you don't need at query time, etc.) without being a
    pure "smallest wins" race — the metric is your footprint relative to
    how everyone else solved the same problem, on the same corpus.

    This score is entirely independent of ranking quality: a submission
    can have a tiny, highly compressed index and still rank poorly if its
    BM25/VSM logic is wrong, and vice versa. It only measures storage
    efficiency.
    """
    if class_median_index_bytes <= 0:
        return 0.0
    ratio = your_index_bytes / class_median_index_bytes
    if ratio <= 0.5:
        return MAX_INDEX_SIZE_WEIGHT
    if ratio >= 2.0:
        return 0.0
    # Linear interpolation between (0.5 -> full marks) and (2.0 -> zero).
    fraction = 1.0 - (ratio - 0.5) / 1.5
    return fraction * MAX_INDEX_SIZE_WEIGHT


def summarize(
    aggregate_metrics: Dict[str, float],
    baseline_aggregate_metrics: Optional[Dict[str, float]] = None,
) -> Dict:
    provisional = compute_provisional_score(aggregate_metrics)
    result = {
        "ndcg@10": aggregate_metrics["ndcg@10"],
        "map@10": aggregate_metrics["map@10"],
        "provisional_score_80pct": provisional,
        "note": (
            "This is nDCG@10 and MAP@10 combined at their 70%/10% weights "
            "(80% of the total). The remaining ±10% efficiency modifier "
            "and 0-10% index-size score are both computed relative to the "
            "whole class and are applied when course staff aggregate the "
            "real leaderboard."
        ),
    }
    if baseline_aggregate_metrics is not None:
        result["baseline_ndcg@10"] = baseline_aggregate_metrics["ndcg@10"]
        result["baseline_map@10"] = baseline_aggregate_metrics["map@10"]
        result["beats_baseline_ndcg@10"] = aggregate_metrics["ndcg@10"] > baseline_aggregate_metrics["ndcg@10"]
        result["beats_baseline_map@10"] = aggregate_metrics["map@10"] > baseline_aggregate_metrics["map@10"]
    return result
