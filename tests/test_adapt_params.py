"""Collection-adaptive knobs: short collections keep PARAMS; news uses _NEWS."""
from submission.retrieve import PARAMS, _NEWS, adapt_params


def test_short_small_collection_matches_shipped_params():
    p = adapt_params(171332, 114.59)
    assert abs(p["k1"] - 2.15) < 1e-9
    assert p["b"] == PARAMS["b"]
    assert p["delta"] == PARAMS["delta"]
    assert p["title_bonus"] == PARAMS["title_bonus"]
    assert p["fb_docs"] == PARAMS["fb_docs"]
    assert p["fb_terms"] == PARAMS["fb_terms"]
    assert abs(p["alpha"] - PARAMS["alpha"]) < 1e-9
    assert p["mu"] == PARAMS["mu"]
    assert p["rm3_pool"] == PARAMS["rm3_pool"]
    assert p["use_rm3"] is True


def test_long_large_collection_uses_newswire_operating_point():
    p = adapt_params(528155, 280.0)
    assert abs(p["k1"] - 0.9) < 1e-9
    assert abs(p["b"] - 0.4) < 1e-9
    assert abs(p["delta"] - 1.0) < 1e-9
    assert abs(p["mu"] - 700.0) < 1e-9
    assert p["fb_docs"] == 10
    assert p["fb_terms"] == 10
    assert p["use_rm3"] is True
    assert abs(p["alpha"] - 0.50) < 1e-9
    assert abs(p["title_bonus"] - 0.4) < 1e-9


def test_small_collection_caps_feedback_at_one_percent():
    p = adapt_params(1400, 80.0)
    assert p["fb_docs"] == 14
    assert p["fb_docs"] <= 1400 // 100
    assert p["rm3_pool"] <= max(50, 1400 // 4)


def test_very_short_docs_use_milder_bm25():
    p = adapt_params(400000, 20.0)
    assert abs(p["k1"] - 1.75) < 1e-9
    assert p["use_rm3"] is False


def test_small_paper_collection_keeps_first_pass():
    p = adapt_params(5000, 145.0)
    assert abs(p["k1"] - 2.35) < 1e-9
    assert p["use_rm3"] is False


def test_small_fulltext_collection_enables_light_rm3():
    p = adapt_params(3600, 156.0)
    assert p["use_rm3"] is True
    assert abs(p["alpha"] - 0.72) < 1e-9


def test_small_short_docs_enable_light_rm3():
    p = adapt_params(3204, 33.63)
    assert p["use_rm3"] is True
    assert abs(p["alpha"] - 0.55) < 1e-9
    p = adapt_params(11429, 24.84)
    assert p["use_rm3"] is True
    assert abs(p["alpha"] - 0.65) < 1e-9


def test_toy_sized_collection_does_not_force_rm3():
    p = adapt_params(20, 15.0)
    assert p["use_rm3"] is False
