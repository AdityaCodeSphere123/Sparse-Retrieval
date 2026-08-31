"""Collection-adaptive knobs: short collections keep PARAMS; news uses _NEWS."""
from submission.retrieve import PARAMS, _NEWS, adapt_params


def test_short_small_collection_matches_shipped_params():
    p = adapt_params(171332, 114.59)
    assert p["k1"] == PARAMS["k1"]
    assert p["b"] == PARAMS["b"]
    assert p["delta"] == PARAMS["delta"]
    assert p["title_bonus"] == PARAMS["title_bonus"]
    assert p["fb_docs"] == PARAMS["fb_docs"]
    assert p["fb_terms"] == PARAMS["fb_terms"]
    assert abs(p["alpha"] - PARAMS["alpha"]) < 1e-9
    assert p["mu"] == PARAMS["mu"]
    assert p["rm3_pool"] == PARAMS["rm3_pool"]


def test_long_large_collection_uses_newswire_operating_point():
    p = adapt_params(528155, 280.0)
    assert p["k1"] == _NEWS["k1"]
    assert p["b"] == _NEWS["b"]
    assert p["b"] < PARAMS["b"]
    assert p["mu"] == _NEWS["mu"]
    assert p["fb_docs"] == _NEWS["fb_docs"]
    assert p["fb_terms"] == _NEWS["fb_terms"]
    assert p["rm3_pool"] == _NEWS["rm3_pool"]
    assert p["title_bonus"] == 0.0


def test_small_collection_caps_feedback_at_one_percent():
    p = adapt_params(1400, 80.0)
    assert p["fb_docs"] == 14
    assert p["fb_docs"] <= 1400 // 100
    assert p["rm3_pool"] <= max(50, 1400 // 4)
