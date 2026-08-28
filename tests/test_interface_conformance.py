"""
tests/test_interface_conformance.py — the exact check the CI conformance
job runs on every push (assignment Section 5, "Submission Interface &
Conformance Checking"). If this file passes, your submission has the
right shape for the grading harness to run it; it says nothing about
ranking quality.

Run it yourself any time with:
    pytest tests/test_interface_conformance.py -v
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import pytest

from harness.run_harness import _validate_and_sort_results, check_conformance
from harness.trec_io import read_queries, read_corpus
from submission import retrieve as submission_module

TOY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "toy")
CORPUS_PATH = os.path.join(TOY_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(TOY_DIR, "queries_dev.tsv")
QRELS_PATH = os.path.join(TOY_DIR, "qrels_dev.txt")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_required_functions_exist_with_correct_signature():
    problems = check_conformance(submission_module)
    assert not problems, f"Interface conformance problems: {problems}"


def test_build_index_writes_something_to_disk():
    with tempfile.TemporaryDirectory() as index_dir:
        submission_module.build_index(CORPUS_PATH, index_dir)
        written = [
            os.path.join(dp, fn)
            for dp, _dn, fns in os.walk(index_dir)
            for fn in fns
        ]
        assert written, (
            "build_index() did not write anything to index_dir. "
            "load_index() runs in a fresh process with no memory of "
            "build_index() — everything retrieve() needs must be on disk."
        )


def test_load_index_reconstructs_purely_from_disk_in_a_fresh_process():
    """The real proof that persistence works: build and query run as two
    genuinely separate `python -m harness.run_harness` subprocesses — the
    same mechanism the real grading harness uses — not just two in-process
    function calls that could accidentally succeed off leftover module
    state. If load_index() secretly depended on something build_index()
    left in memory, this is the test that would catch it."""
    with tempfile.TemporaryDirectory() as run_dir:
        report_path = os.path.join(run_dir, "report.json")
        result = subprocess.run(
            [
                sys.executable, "-m", "harness.run_harness",
                "--corpus", CORPUS_PATH,
                "--queries", QUERIES_PATH,
                "--qrels", QRELS_PATH,
                "--report-out", report_path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"harness exited {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        with open(report_path) as f:
            report = json.load(f)
        eff = report["efficiency"]
        assert eff["index_size_bytes"] > 0, "reported index size was 0 bytes"
        assert eff["index_build_seconds"] >= 0
        assert eff["index_load_seconds"] >= 0


def test_retrieve_may_not_return_a_duplicate_doc_id():
    # Anti-gaming check: a retrieve() that returns the same doc_id twice
    # would let that document's relevance get double-counted by nDCG@10/
    # MAP@10 (see tests/test_metrics.py for the exploit this closes), so
    # the harness treats it as a submission bug (RUNTIME_ERROR at grading
    # time), not something to silently tolerate or deduplicate away.
    with pytest.raises(ValueError, match="duplicate doc_id"):
        _validate_and_sort_results([("d1", 5.0), ("d1", 4.0)], qid="q1", k=10)


def test_retrieve_returns_well_formed_results():
    with tempfile.TemporaryDirectory() as index_dir:
        submission_module.build_index(CORPUS_PATH, index_dir)
        submission_module.load_index(index_dir)

        queries = read_queries(QUERIES_PATH)
        valid_doc_ids = {doc_id for doc_id, _text in read_corpus(CORPUS_PATH)}

        for qid, text in queries:
            results = submission_module.retrieve(text, 5)
            assert len(results) <= 5, f"qid={qid} returned more than k results"
            scores = [score for _doc_id, score in results]
            assert scores == sorted(scores, reverse=True), f"qid={qid} results not sorted descending"
            for doc_id, score in results:
                assert doc_id in valid_doc_ids, f"qid={qid} returned unknown doc_id {doc_id!r}"
                assert isinstance(score, (int, float)), f"qid={qid} score {score!r} is not numeric"


def test_retrieve_is_reasonably_fast_on_the_toy_set():
    """Not a correctness check — just catches an accidentally quadratic-in-corpus-size
    per-query implementation before it becomes a problem on the real corpus."""
    with tempfile.TemporaryDirectory() as index_dir:
        t0 = time.perf_counter()
        submission_module.build_index(CORPUS_PATH, index_dir)
        build_time = time.perf_counter() - t0
        submission_module.load_index(index_dir)

        queries = read_queries(QUERIES_PATH)
        latencies = []
        for _qid, text in queries:
            t0 = time.perf_counter()
            submission_module.retrieve(text, 10)
            latencies.append(time.perf_counter() - t0)

    assert build_time < 30, "Index build on a 20-document toy corpus took >30s — something is off."
    assert max(latencies) < 5, "A single query on a 20-document toy corpus took >5s — something is off."


def test_build_index_accepts_a_corpus_directory():
    """Held-out collection may arrive as a directory of documents."""
    import shutil

    from submission.corpus_utils import load_corpus

    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = os.path.join(tmp, "collection")
        os.makedirs(corpus_dir)
        shutil.copy(CORPUS_PATH, os.path.join(corpus_dir, "corpus.jsonl"))
        docs = load_corpus(corpus_dir)
        assert len(docs) == len(load_corpus(CORPUS_PATH))

        index_dir = os.path.join(tmp, "index")
        submission_module.build_index(corpus_dir, index_dir)
        submission_module.load_index(index_dir)
        results = submission_module.retrieve("cat", 5)
        assert results
        assert len(results) <= 5

