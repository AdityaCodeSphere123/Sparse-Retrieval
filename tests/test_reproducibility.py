"""Index build/load determinism and two-process harness persistence.

Requirement: 'Runs end-to-end from the README; index build
and load are both deterministic.'
"""
import hashlib
import os
import tempfile

from submission import retrieve as submission_module
from harness.trec_io import read_queries

TOY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "toy")
CORPUS_PATH = os.path.join(TOY_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(TOY_DIR, "queries_dev.tsv")


def _file_hashes(root):
    hashes = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            hashes[rel] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return hashes


def test_two_index_builds_are_byte_identical():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        submission_module.build_index(CORPUS_PATH, d1)
        submission_module.build_index(CORPUS_PATH, d2)
        h1 = _file_hashes(d1)
        h2 = _file_hashes(d2)
        assert h1, "build_index wrote nothing"
        assert h1.keys() == h2.keys()
        assert h1 == h2, "index bytes differed across two builds of the same corpus"


def test_retrieve_is_deterministic_after_reload():
    queries = read_queries(QUERIES_PATH)
    with tempfile.TemporaryDirectory() as index_dir:
        submission_module.build_index(CORPUS_PATH, index_dir)
        submission_module.load_index(index_dir)
        first = {qid: submission_module.retrieve(text, 10) for qid, text in queries}

        # Simulate a fresh process: reload from the same directory.
        submission_module.load_index(index_dir)
        second = {qid: submission_module.retrieve(text, 10) for qid, text in queries}

        assert first == second
        for qid, ranking in first.items():
            doc_ids = [doc_id for doc_id, _score in ranking]
            assert len(doc_ids) == len(set(doc_ids))
            scores = [score for _doc_id, score in ranking]
            assert scores == sorted(scores, reverse=True)


def test_cpp_tie_break_is_external_doc_id_not_insertion_order():
    """Competition retrieve() must match bm25.py: score desc, then doc_id asc."""
    import pytest
    from submission import retrieve as r

    if not r.HAS_CPP:
        pytest.skip("cpp_engine not built")

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"doc_id": "docZ", "text": "cat cat cat"}\n')
            f.write('{"doc_id": "docA", "text": "cat cat cat"}\n')
        index_dir = os.path.join(tmp, "index")
        r.build_index(corpus, index_dir)
        r.load_index(index_dir)
        ranking = r.retrieve_bm25_only("cat", k=10)
        ids = [doc_id for doc_id, _ in ranking]
        assert ids[:2] == ["docA", "docZ"], ids
        assert ranking[0][1] == pytest.approx(ranking[1][1])


def test_topk_cut_on_score_tie_uses_external_doc_id():
    """nth_element and sort must share the spec comparator (score, then doc_id)."""
    import pytest
    from submission import retrieve as r

    if not r.HAS_CPP:
        pytest.skip("cpp_engine not built")

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            for did in ("docE", "docD", "docC", "docB", "docA"):
                f.write('{"doc_id": "%s", "text": "cat cat cat"}\n' % did)
        index_dir = os.path.join(tmp, "index")
        r.build_index(corpus, index_dir)
        r.load_index(index_dir)
        ranking = r.retrieve_bm25_only("cat", k=3)
        ids = [doc_id for doc_id, _ in ranking]
        assert ids == ["docA", "docB", "docC"], ids
        scores = [score for _doc_id, score in ranking]
        assert scores[0] == pytest.approx(scores[1])
        assert scores[1] == pytest.approx(scores[2])


def test_retrieve_bm25_only_skips_query_length_heuristics():
    """Long queries must not zero δ / title_bonus on the BM25 ablation path."""
    import pytest
    from submission import retrieve as r

    if not r.HAS_CPP:
        pytest.skip("cpp_engine not built")

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"doc_id": "d1", "text": "cat dog mouse"}\n')
            f.write('{"doc_id": "d2", "text": "cat bird fish"}\n')
        index_dir = os.path.join(tmp, "index")
        r.build_index(corpus, index_dir)
        r.load_index(index_dir)
        long_q = " ".join(["cat"] * 50)
        cut = r._CPP_ENGINE.retrieve(
            long_q, 10, 1.2, 0.75, 0.8, 0.0, 0.6,
            False, 10, 25, 0.6, 1000.0, 250, True,
        )
        pure = r._CPP_ENGINE.retrieve_bm25(long_q, 10, 1.2, 0.75, 0.8, 0.0, 0.6)
        assert cut and pure
        assert cut[0][1] != pytest.approx(pure[0][1])


def test_parallel_build_matches_serial(monkeypatch):
    """Contiguous doc-id shards must be byte-identical to a 1-worker build."""
    import pytest
    from submission import retrieve as r

    if not r.HAS_CPP:
        pytest.skip("cpp_engine not built")

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        monkeypatch.setenv("IR_BUILD_WORKERS", "1")
        r.build_index(CORPUS_PATH, d1)
        monkeypatch.setenv("IR_BUILD_WORKERS", "4")
        r.build_index(CORPUS_PATH, d2)
        assert _file_hashes(d1) == _file_hashes(d2)


def test_ragged_doc_ids_roundtrip_front_coded_docmap():
    """Held-out collections may mix id widths; id_len==0 must load back."""
    import struct
    import pytest
    from submission import retrieve as r

    if not r.HAS_CPP:
        pytest.skip("cpp_engine not built")

    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "corpus.jsonl")
        with open(corpus, "w", encoding="utf-8") as f:
            f.write('{"doc_id": "short", "text": "alpha beta"}\n')
            f.write('{"doc_id": "a-much-longer-doc-id", "text": "alpha gamma"}\n')
            f.write('{"doc_id": "midid", "text": "beta gamma"}\n')
        index_dir = os.path.join(tmp, "index")
        r.build_index(corpus, index_dir)
        raw = os.path.join(index_dir, "docmap.bin")
        gz = raw + ".gz"
        if os.path.isfile(gz):
            import gzip
            docmap = gzip.open(gz, "rb").read()
        else:
            docmap = open(raw, "rb").read()
        n, id_len = struct.unpack_from("<IH", docmap, 0)
        assert n == 3
        assert id_len == 0, "mixed widths must take the front-coded path"
        r.load_index(index_dir)
        ranking = r.retrieve("alpha", k=10)
        got = {doc_id for doc_id, _ in ranking}
        assert "short" in got and "a-much-longer-doc-id" in got
