"""Independent correctness checks for Boolean/VSM and BM25.

These are the required components graded against known small examples,
not leaderboard nDCG. Expected scores are
hand-derived from the formulas in the module docstrings.
"""
import hashlib
import math
import os
import tempfile

import pytest

from submission.indexer import InvertedIndex, tokenize
from submission import boolean_vsm, bm25, custom_scorer

CORPUS = [
    ("doc1", "information retrieval systems and inverted index"),
    ("doc2", "search engine architecture and information indexing"),
    ("doc3", "deep learning models for natural language processing"),
]

# Tiny corpus of unstemmed tokens so expected values are exact.
# Tokens: cat, dog, bird — none of these change under the S-stemmer.
TINY = [
    ("dA", "cat cat dog"),  # tf(cat)=2, tf(dog)=1, |D|=3
    ("dB", "cat"),          # tf(cat)=1, |D|=1
    ("dC", "dog bird"),     # tf(dog)=1, tf(bird)=1, |D|=2
]


def _tiny_index():
    index = InvertedIndex()
    index.build(TINY)
    return index


def test_inverted_index_build_save_load():
    with tempfile.TemporaryDirectory() as index_dir:
        index = InvertedIndex()
        index.build(CORPUS)

        assert index.N == 3
        assert index.document_frequency("information") == 2
        assert index.document_frequency("learning") == 1

        index.save(index_dir)
        loaded = InvertedIndex.load(index_dir)

        assert loaded.N == 3
        assert loaded.avg_doc_len == index.avg_doc_len
        assert loaded.document_frequency("information") == 2
        assert loaded.postings == index.postings
        assert loaded.doc_len == index.doc_len
        for doc_id, norm in index.doc_norm.items():
            assert loaded.doc_norm[doc_id] == pytest.approx(norm)


def test_boolean_search():
    index = InvertedIndex()
    index.build(CORPUS)
    boolean_vsm.build(index)

    and_docs = boolean_vsm.boolean_search("information index", mode="and")
    # doc2 has "information" and "indexing" (does not S-stem to "index").
    assert and_docs == ["doc1"]

    or_docs = boolean_vsm.boolean_search("learning retrieval", mode="or")
    assert "doc1" in or_docs
    assert "doc3" in or_docs


def test_boolean_and_or_on_known_set():
    boolean_vsm.build(_tiny_index())

    # AND: only dA has both cat and dog.
    assert set(boolean_vsm.boolean_search("cat dog", mode="and")) == {"dA"}
    # AND with a term that exists only in dC should be empty vs cat.
    assert boolean_vsm.boolean_search("cat bird", mode="and") == []
    # Missing term: conjunction is empty.
    assert boolean_vsm.boolean_search("cat unicorn", mode="and") == []
    # OR: cat OR bird hits all three.
    assert set(boolean_vsm.boolean_search("cat bird", mode="or")) == {"dA", "dB", "dC"}
    # OR of unknown terms is empty.
    assert boolean_vsm.boolean_search("unicorn griffin", mode="or") == []
    # Empty query.
    assert boolean_vsm.boolean_search("", mode="and") == []


def test_boolean_rejects_invalid_mode():
    boolean_vsm.build(_tiny_index())
    with pytest.raises(ValueError, match="mode"):
        boolean_vsm.boolean_search("cat", mode="xor")
    with pytest.raises(ValueError, match="mode"):
        boolean_vsm.boolean_search("cat", mode="AND NOT")


def test_boolean_and_or_results_are_sorted():
    boolean_vsm.build(_tiny_index())
    assert boolean_vsm.boolean_search("cat dog", mode="or") == ["dA", "dB", "dC"]
    assert boolean_vsm.boolean_search("cat", mode="and") == ["dA", "dB"]


def test_vsm_returns_at_most_k_and_empty_on_oov():
    boolean_vsm.build(_tiny_index())
    assert boolean_vsm.vsm_score("cat", k=1) == [
        boolean_vsm.vsm_score("cat", k=10)[0]
    ]
    assert boolean_vsm.vsm_score("", k=10) == []
    assert boolean_vsm.vsm_score("unicorn", k=10) == []


def test_vsm_tie_break_is_doc_id_ascending():
    # Need df(cat) < N so ln(N/df) is nonzero; otherwise every weight is 0
    # and cosine returns no documents.
    index = InvertedIndex()
    index.build([("z_doc", "cat"), ("a_doc", "cat"), ("other", "dog")])
    boolean_vsm.build(index)
    ranking = boolean_vsm.vsm_score("cat", k=10)
    assert [doc_id for doc_id, _ in ranking] == ["a_doc", "z_doc"]
    assert ranking[0][1] == pytest.approx(ranking[1][1])


def test_bm25_empty_query_and_oov_are_empty():
    bm25.build(_tiny_index())
    assert bm25.score("", k=10) == []
    assert bm25.score("unicorn", k=10) == []


def test_bm25_multi_term_is_sum_of_idf_weighted_tfs():
    """Two-term query: score(d) = BM25(cat) + BM25(dog) on that document."""
    index = _tiny_index()
    bm25.build(index)
    k1, b = 1.2, 0.75
    N, avgdl = 3, (3 + 1 + 2) / 3.0

    def idf(df):
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def sat(tf, doc_len):
        K = k1 * (1.0 - b + b * (doc_len / avgdl))
        return (tf * (k1 + 1.0)) / (tf + K)

    # dA: cat tf=2, dog tf=1, |D|=3
    expected = idf(2) * sat(2, 3) + idf(2) * sat(1, 3)
    got = dict(bm25.score("cat dog", k=10, k1=k1, b=b, delta=0.0))
    assert got["dA"] == pytest.approx(expected, rel=1e-12)
    assert "dB" in got  # cat only
    assert "dC" in got  # dog only

    boolean_vsm.build(_tiny_index())

    # AND: only dA has both cat and dog.
    assert set(boolean_vsm.boolean_search("cat dog", mode="and")) == {"dA"}
    # AND with a term that exists only in dC should be empty vs cat.
    assert boolean_vsm.boolean_search("cat bird", mode="and") == []
    # Missing term: conjunction is empty.
    assert boolean_vsm.boolean_search("cat unicorn", mode="and") == []
    # OR: cat OR bird hits all three.
    assert set(boolean_vsm.boolean_search("cat bird", mode="or")) == {"dA", "dB", "dC"}
    # OR of unknown terms is empty.
    assert boolean_vsm.boolean_search("unicorn griffin", mode="or") == []
    # Empty query.
    assert boolean_vsm.boolean_search("", mode="and") == []


def test_vsm_and_bm25_scoring():
    index = InvertedIndex()
    index.build(CORPUS)
    boolean_vsm.build(index)
    bm25.build(index)

    vsm_results = boolean_vsm.vsm_score("information retrieval", k=2)
    assert len(vsm_results) <= 2
    assert vsm_results[0][0] == "doc1"

    bm25_results = bm25.score("information retrieval", k=2, delta=0.0)
    assert len(bm25_results) <= 2
    assert bm25_results[0][0] == "doc1"

    bm25_plus_results = bm25.score("information retrieval", k=2, delta=1.0)
    assert len(bm25_plus_results) <= 2
    assert bm25_plus_results[0][0] == "doc1"


def test_vsm_cosine_is_full_document_norm_not_query_subspace():
    """A single-term query must NOT score every hit at 1.0.

    Query-term-only document norms collapse cosine to 1 for every document
    that contains the query term. True cosine uses ||d|| over all terms, so
    the focused document (dB: just 'cat') outranks the mixed one (dA: cat+dog).
    """
    boolean_vsm.build(_tiny_index())
    results = boolean_vsm.vsm_score("cat", k=10)
    scores = {doc_id: score for doc_id, score in results}

    assert "dA" in scores and "dB" in scores
    assert "dC" not in scores  # no 'cat'
    assert scores["dB"] > scores["dA"]
    assert scores["dB"] < 1.0 + 1e-12
    # dB is a pure 'cat' vector, so cosine with query 'cat' is exactly 1.
    assert scores["dB"] == pytest.approx(1.0)
    assert 0.0 < scores["dA"] < 1.0


def test_vsm_scores_match_hand_computed_tfidf_cosine():
    index = _tiny_index()
    boolean_vsm.build(index)
    N = 3
    idf = {
        "cat": math.log(N / 2),
        "dog": math.log(N / 2),
        "bird": math.log(N / 1),
    }

    def w(tf, term):
        return (1.0 + math.log(tf)) * idf[term]

    dA = {"cat": w(2, "cat"), "dog": w(1, "dog")}
    dB = {"cat": w(1, "cat")}
    dC = {"dog": w(1, "dog"), "bird": w(1, "bird")}

    def l2(vec):
        return math.sqrt(sum(v * v for v in vec.values()))

    q = {"cat": w(1, "cat")}
    q_norm = l2(q)

    def cosine(qvec, dvec):
        dot = sum(qvec[t] * dvec[t] for t in qvec if t in dvec)
        return dot / (q_norm * l2(dvec))

    expected = {
        "dA": cosine(q, dA),
        "dB": cosine(q, dB),
    }

    got = dict(boolean_vsm.vsm_score("cat", k=10))
    assert set(got) == set(expected)
    for doc_id, score in expected.items():
        assert got[doc_id] == pytest.approx(score, rel=1e-12)


def test_bm25_scores_match_hand_computed_okapi_formula():
    """Okapi BM25 as written in submission/bm25.py (RSJ IDF with +1)."""
    index = _tiny_index()
    bm25.build(index)

    N = 3
    avgdl = (3 + 1 + 2) / 3.0
    assert index.N == N
    assert index.avg_doc_len == pytest.approx(avgdl)
    assert index.doc_len["dA"] == 3
    assert index.doc_len["dB"] == 1

    k1, b = 1.2, 0.75
    df_cat = 2
    idf_cat = math.log(1.0 + (N - df_cat + 0.5) / (df_cat + 0.5))

    def tf_norm(tf, doc_len):
        K = k1 * (1.0 - b + b * (doc_len / avgdl))
        return (tf * (k1 + 1.0)) / (tf + K)

    expected_dA = idf_cat * tf_norm(2, 3)
    expected_dB = idf_cat * tf_norm(1, 1)

    got = dict(bm25.score("cat", k=10, k1=k1, b=b, delta=0.0))
    assert set(got) == {"dA", "dB"}
    assert got["dA"] == pytest.approx(expected_dA, rel=1e-12)
    assert got["dB"] == pytest.approx(expected_dB, rel=1e-12)
    # Shorter focused dB should beat mixed longer dA on this query.
    assert got["dB"] > got["dA"]


def test_bm25_plus_adds_delta_times_idf():
    index = _tiny_index()
    bm25.build(index)
    plain = dict(bm25.score("cat", k=10, k1=1.2, b=0.75, delta=0.0))
    plus = dict(bm25.score("cat", k=10, k1=1.2, b=0.75, delta=1.0))
    N, df = 3, 2
    idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
    assert plus["dA"] == pytest.approx(plain["dA"] + idf)
    assert plus["dB"] == pytest.approx(plain["dB"] + idf)


def test_bm25_k1_and_b_are_not_hardcoded():
    bm25.build(_tiny_index())
    a = dict(bm25.score("cat", k=10, k1=0.5, b=0.0, delta=0.0))
    b = dict(bm25.score("cat", k=10, k1=2.5, b=1.0, delta=0.0))
    assert a["dA"] != pytest.approx(b["dA"])
    assert a["dB"] != pytest.approx(b["dB"])


def test_bm25_rejects_invalid_k1_and_b():
    bm25.build(_tiny_index())
    with pytest.raises(ValueError, match="k1"):
        bm25.score("cat", k=10, k1=0.0, b=0.75)
    with pytest.raises(ValueError, match="b"):
        bm25.score("cat", k=10, k1=1.2, b=1.5)


def test_bm25_tie_break_is_doc_id_ascending():
    index = InvertedIndex()
    index.build([("z_doc", "cat"), ("a_doc", "cat")])
    bm25.build(index)
    ranking = bm25.score("cat", k=10, k1=1.2, b=0.75, delta=0.0)
    assert ranking[0][1] == pytest.approx(ranking[1][1])
    assert [doc_id for doc_id, _ in ranking] == ["a_doc", "z_doc"]


def test_bm25_b_zero_ignores_length():
    """b=0 => K=k1 for every document; only tf (and idf) remain."""
    bm25.build(_tiny_index())
    k1 = 1.2
    N, df = 3, 2
    idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def sat(tf):
        return (tf * (k1 + 1.0)) / (tf + k1)

    got = dict(bm25.score("cat", k=10, k1=k1, b=0.0, delta=0.0))
    assert got["dA"] == pytest.approx(idf * sat(2))
    assert got["dB"] == pytest.approx(idf * sat(1))


def test_index_save_is_byte_deterministic():
    index = InvertedIndex()
    index.build(CORPUS)
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        index.save(d1)
        index.save(d2)
        f1 = os.path.join(d1, "index.json.gz")
        f2 = os.path.join(d2, "index.json.gz")
        h1 = hashlib.sha256(open(f1, "rb").read()).hexdigest()
        h2 = hashlib.sha256(open(f2, "rb").read()).hexdigest()
        assert h1 == h2


def test_custom_scorer_fuses_both_signals():
    index = _tiny_index()
    custom_scorer.build(index)
    ranking = custom_scorer.score("cat", k=10)
    doc_ids = [d for d, _ in ranking]
    assert "dB" in doc_ids and "dA" in doc_ids
    assert ranking == sorted(ranking, key=lambda x: (-x[1], x[0]))
    assert len(doc_ids) == len(set(doc_ids))
    assert all(0.0 <= s <= 1.0 + 1e-12 for _, s in ranking)


def test_tokenize_s_stem_is_stable_on_plurals():
    # Guardrail: scoring tests assume this stemmer. If it changes, expected
    # rankings on the starter CORPUS (systems/index/indexing) change too.
    assert tokenize("Cats cats") == ["cat", "cat"]
    assert tokenize("information retrieval") == ["information", "retrieval"]
