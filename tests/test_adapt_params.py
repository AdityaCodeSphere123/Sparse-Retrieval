"""Collection-adaptive knobs: short/small collections keep PARAMS."""
from submission.retrieve import PARAMS, adapt_params


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


def test_long_large_collection_moves_toward_newswire_defaults():
    p = adapt_params(528155, 280.0)
    assert p["k1"] < PARAMS["k1"]
    assert p["b"] > PARAMS["b"]
    assert p["mu"] < PARAMS["mu"]
    assert p["fb_docs"] < PARAMS["fb_docs"]
    assert p["fb_terms"] > PARAMS["fb_terms"]
    assert p["rm3_pool"] > PARAMS["rm3_pool"]
