"""Index build/load determinism and two-process harness persistence.

Assignment Section 7 (5%): 'Runs end-to-end from the README; index build
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
