import os
import tempfile
import pytest

from submission.indexer import InvertedIndex
from submission import boolean_vsm, bm25

CORPUS = [
    ("doc1", "information retrieval systems and inverted index"),
    ("doc2", "search engine architecture and information indexing"),
    ("doc3", "deep learning models for natural language processing"),
]


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


def test_boolean_search():
    index = InvertedIndex()
    index.build(CORPUS)
    boolean_vsm.build(index)

    and_docs = boolean_vsm.boolean_search("information index", mode="and")
    assert "doc1" in and_docs

    or_docs = boolean_vsm.boolean_search("learning retrieval", mode="or")
    assert "doc1" in or_docs
    assert "doc3" in or_docs


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
