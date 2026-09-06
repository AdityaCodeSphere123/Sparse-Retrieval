"""
harness/run_harness.py — the grading harness. This is the exact code path
used for the public dev leaderboard (which you can run yourself) and, with
a different corpus/topics/qrels file that is never distributed, the
private held-out leaderboard at grading time.

Usage:
    python -m harness.run_harness \\
        --corpus data/toy/corpus.jsonl \\
        --queries data/toy/queries_dev.tsv \\
        --qrels data/toy/qrels_dev.txt \\
        --k 10 \\
        --submission submission.retrieve \\
        --run-out runs/dev_run.trec \\
        --report-out runs/dev_report.json \\
        --baseline-run data/toy/reference_bm25_run_dev.trec

Two-phase execution model
--------------------------
Your submission must expose three functions (see docs/SUBMISSION_INTERFACE.md):

    build_index(corpus_path, index_dir)   # construct the index, write it to index_dir
    load_index(index_dir)                 # reconstruct everything retrieve() needs, from disk only
    retrieve(query, k)                    # answer a query using what load_index() set up

This harness runs `build_index` and `load_index`/`retrieve` as two SEPARATE
subprocess invocations of this same script (internally, `--_phase build`
then `--_phase query`), not as two function calls inside one Python
process. That is deliberate: it is the only way to prove `load_index()`
genuinely reconstructs everything from disk, rather than quietly piggy-backing
on module-level state left over from `build_index()` having run earlier in
the same process. The query-phase subprocess starts with none of the build
phase's memory — if `load_index()` is incomplete, it fails or scores badly
right here, not silently. The on-disk byte size of `index_dir`, measured by
this script between the two phases, is what the class-relative index-size
score (assignment Section 7) is computed from — see
instructor-tools/aggregate_leaderboard.py.

You do not need to pass `--_phase` or `--_result-out` yourself; they're how
this script re-invokes itself and are not part of the interface you're
graded against.

Exit codes:
    0  ran to completion (see the printed/JSON report for scores)
    1  interface conformance failure (this is what the CI check greps for)
    2  runtime error while scoring (crash inside build_index/load_index/retrieve)
"""
import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Tuple

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries, read_run, write_run
from harness.leaderboard import summarize

REQUIRED_BUILD_PARAMS = ["corpus_path", "index_dir"]
REQUIRED_LOAD_PARAMS = ["index_dir"]
REQUIRED_RETRIEVE_PARAMS = ["query", "k"]


def check_conformance(submission_module) -> List[str]:
    """Returns a list of conformance problems (empty list = conforms)."""
    problems = []

    if not hasattr(submission_module, "build_index"):
        problems.append("submission module is missing a `build_index(corpus_path, index_dir)` function.")
    else:
        names = list(inspect.signature(submission_module.build_index).parameters.keys())
        if names[:2] != REQUIRED_BUILD_PARAMS:
            problems.append(
                f"build_index() must accept (corpus_path, index_dir, ...) as its first "
                f"two parameters (in that order); found {names}."
            )

    if not hasattr(submission_module, "load_index"):
        problems.append("submission module is missing a `load_index(index_dir)` function.")
    else:
        names = list(inspect.signature(submission_module.load_index).parameters.keys())
        if names[:1] != REQUIRED_LOAD_PARAMS:
            problems.append(
                f"load_index() must accept (index_dir, ...) as its first parameter; found {names}."
            )

    if not hasattr(submission_module, "retrieve"):
        problems.append("submission module is missing a `retrieve(query, k)` function.")
    else:
        names = list(inspect.signature(submission_module.retrieve).parameters.keys())
        if names[:2] != REQUIRED_RETRIEVE_PARAMS:
            problems.append(
                f"retrieve() must accept (query, k, ...) as its first two parameters "
                f"(in that order); found {names}."
            )

    return problems


def _import_or_die(submission_path: str):
    try:
        return importlib.import_module(submission_path)
    except Exception as e:
        print(f"CONFORMANCE FAIL: could not import '{submission_path}': {e}", file=sys.stderr)
        sys.exit(1)


def _check_conformance_or_die(submission_module) -> None:
    problems = check_conformance(submission_module)
    if problems:
        print("CONFORMANCE FAIL:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)


def _directory_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def _validate_and_sort_results(results, qid: str, k: int) -> List[Tuple[str, float]]:
    if not isinstance(results, list):
        raise TypeError(f"retrieve() must return a list; got {type(results)} for qid={qid}")
    if len(results) > k:
        raise ValueError(f"retrieve() returned {len(results)} results for qid={qid}, more than k={k}")
    seen_doc_ids = set()
    for item in results:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise TypeError(f"each result must be a (doc_id, score) pair; got {item!r} for qid={qid}")
        doc_id, _score = item
        if doc_id in seen_doc_ids:
            # Not just untidy output: nDCG@10 and MAP@10 both sum a
            # per-rank contribution with no upper cap, so a doc_id
            # repeated in the same result list lets its relevance get
            # counted more than once — a submission that returns one
            # known-relevant document k times over can drive both metrics
            # well past their theoretical maximum of 1.0. Reject it as a
            # submission bug rather than silently deduplicating, so it
            # shows up as something to fix, not a free ranking boost.
            raise ValueError(
                f"retrieve() returned duplicate doc_id {doc_id!r} for qid={qid}; "
                f"each doc_id must appear at most once in a query's result list."
            )
        seen_doc_ids.add(doc_id)
    # Defensive: sort by score descending even if the submission forgot to.
    return sorted(((doc_id, score) for doc_id, score in results), key=lambda pair: pair[1], reverse=True)


# --------------------------------------------------------------------------
# Phase entry points. Each is invoked as its own subprocess by
# `_spawn_phase()` below; neither one is meant to be called directly by
# students or graders.
# --------------------------------------------------------------------------

def _main_build_phase(args) -> None:
    if not args.corpus:
        print("internal error: --_phase build requires --corpus", file=sys.stderr)
        sys.exit(2)

    submission_module = _import_or_die(args.submission)
    _check_conformance_or_die(submission_module)

    os.makedirs(args.index_dir, exist_ok=True)
    try:
        t0 = time.perf_counter()
        submission_module.build_index(args.corpus, args.index_dir)
        build_time = time.perf_counter() - t0
    except Exception as e:
        print(f"RUNTIME ERROR during build_index(): {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    with open(args._result_out, "w", encoding="utf-8") as f:
        json.dump({"build_seconds": build_time}, f)


def _main_query_phase(args) -> None:
    if not args.queries:
        print("internal error: --_phase query requires --queries", file=sys.stderr)
        sys.exit(2)

    submission_module = _import_or_die(args.submission)
    _check_conformance_or_die(submission_module)

    queries = read_queries(args.queries)

    try:
        t0 = time.perf_counter()
        submission_module.load_index(args.index_dir)
        load_time = time.perf_counter() - t0

        run: Dict[str, List[Tuple[str, float]]] = {}
        latencies = []
        for qid, text in queries:
            t0 = time.perf_counter()
            results = submission_module.retrieve(text, args.k)
            latencies.append(time.perf_counter() - t0)
            run[qid] = _validate_and_sort_results(results, qid, args.k)
    except Exception as e:
        print(f"RUNTIME ERROR during load_index()/retrieve(): {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    with open(args._result_out, "w", encoding="utf-8") as f:
        json.dump({"load_seconds": load_time, "run": run, "latencies": latencies}, f)


# --------------------------------------------------------------------------
# Orchestrator: spawns the two phases above as fresh subprocesses, measures
# index_dir on disk in between, and assembles the final report. This is
# what runs when a student or `grade_submission.py` invokes
# `python -m harness.run_harness` normally (i.e. without `--_phase`).
# --------------------------------------------------------------------------

def _spawn_phase(args, phase: str, index_dir: str, extra: Dict[str, str]) -> dict:
    fd, result_path = tempfile.mkstemp(suffix=f".{phase}.json")
    os.close(fd)
    cmd = [
        sys.executable, "-m", "harness.run_harness",
        "--_phase", phase,
        "--_result-out", result_path,
        "--submission", args.submission,
        "--index-dir", index_dir,
    ]
    for flag, value in extra.items():
        cmd.extend([flag, value])

    try:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            # The phase subprocess already printed its own CONFORMANCE FAIL /
            # RUNTIME ERROR message to stderr; propagate the same exit code
            # so callers (CI, grade_submission.py) see the signal they
            # always have.
            sys.exit(proc.returncode)
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def _main_orchestrate(args) -> None:
    if not (args.corpus and args.queries and args.qrels):
        print("--corpus, --queries, and --qrels are all required.", file=sys.stderr)
        sys.exit(2)

    # Fast pre-check in this process, purely so a broken interface fails
    # immediately instead of after spawning subprocesses. The phases below
    # each re-check independently too — that's the check that actually
    # counts, since it runs in the same process that calls build_index/
    # load_index/retrieve.
    submission_module = _import_or_die(args.submission)
    _check_conformance_or_die(submission_module)

    owns_index_dir = args.index_dir is None
    index_dir = args.index_dir or tempfile.mkdtemp(prefix="a1_index_")
    os.makedirs(index_dir, exist_ok=True)

    try:
        build_result = _spawn_phase(args, "build", index_dir, {"--corpus": args.corpus})

        index_size_bytes = _directory_size_bytes(index_dir)
        if index_size_bytes == 0:
            print(
                "WARNING: index_dir is empty after build_index() returned — nothing "
                "was written to disk. build_index(corpus_path, index_dir) must persist "
                "everything retrieve() needs; load_index() runs in a brand-new process "
                "with no memory of this one, so an in-memory-only index will fail or "
                "score badly in the next phase.",
                file=sys.stderr,
            )

        query_result = _spawn_phase(
            args, "query", index_dir, {"--queries": args.queries, "--k": str(args.k)}
        )
    finally:
        if owns_index_dir:
            shutil.rmtree(index_dir, ignore_errors=True)

    run = {qid: [tuple(pair) for pair in pairs] for qid, pairs in query_result["run"].items()}
    latencies = query_result["latencies"]
    build_time = build_result["build_seconds"]
    load_time = query_result["load_seconds"]

    qrels = read_qrels(args.qrels)
    queries = read_queries(args.queries)

    if args.run_out:
        os.makedirs(os.path.dirname(args.run_out) or ".", exist_ok=True)
        write_run(args.run_out, run, run_tag=args.run_tag)

    eval_result = evaluate_run(run, qrels, k=args.k)

    baseline_summary = None
    if args.baseline_run and os.path.exists(args.baseline_run):
        baseline_run = read_run(args.baseline_run)
        baseline_eval = evaluate_run(baseline_run, qrels, k=args.k)
        baseline_summary = baseline_eval["aggregate"]

    leaderboard_summary = summarize(eval_result["aggregate"], baseline_summary)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    report = {
        "submission": args.submission,
        "num_queries": len(queries),
        "num_queries_scored": eval_result["num_queries_scored"],
        "skipped_no_qrels": eval_result["skipped_no_qrels"],
        "aggregate_metrics": eval_result["aggregate"],
        "leaderboard": leaderboard_summary,
        "efficiency": {
            "index_build_seconds": build_time,
            "index_load_seconds": load_time,
            "index_size_bytes": index_size_bytes,
            "mean_query_latency_seconds": mean_latency,
            "max_query_latency_seconds": max(latencies) if latencies else 0.0,
        },
        "per_query_metrics": eval_result["per_query"],
    }

    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    _print_human_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--submission", default="submission.retrieve", help="dotted module path exposing build_index()/load_index()/retrieve()")
    parser.add_argument("--run-out", default=None, help="where to write the TREC run file")
    parser.add_argument("--report-out", default=None, help="where to write the JSON report")
    parser.add_argument("--baseline-run", default=None, help="a precomputed baseline run file to compare against")
    parser.add_argument("--run-tag", default="submission")
    parser.add_argument(
        "--index-dir", default=None,
        help="on-disk directory for the index. Default: a throwaway temp dir, "
             "created and cleaned up automatically. Pass your own path to inspect "
             "the on-disk index afterwards (it will not be deleted in that case).",
    )
    parser.add_argument("--_phase", choices=["build", "query"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_result-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._phase == "build":
        _main_build_phase(args)
        return
    if args._phase == "query":
        _main_query_phase(args)
        return

    _main_orchestrate(args)


def _human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _print_human_report(report: dict) -> None:
    agg = report["aggregate_metrics"]
    eff = report["efficiency"]
    lb = report["leaderboard"]
    print("=" * 60)
    print(f"Submission:           {report['submission']}")
    print(f"Queries scored:       {report['num_queries_scored']} / {report['num_queries']}")
    if report["skipped_no_qrels"]:
        print(f"  (skipped, no qrels: {report['skipped_no_qrels']})")
    print("-" * 60)
    print(f"nDCG@10:              {agg['ndcg@10']:.4f}")
    print(f"MAP@10:               {agg['map@10']:.4f}")
    print(f"MRR:                  {agg['mrr']:.4f}")
    print(f"P@10:                 {agg.get('p@10', float('nan')):.4f}")
    print("-" * 60)
    print(f"Index build time:     {eff['index_build_seconds']:.4f}s")
    print(f"Index load time:      {eff['index_load_seconds']:.4f}s  (fresh process, disk only)")
    print(f"Index size on disk:   {_human_bytes(eff['index_size_bytes'])}  ({eff['index_size_bytes']} bytes)")
    print(f"Mean query latency:   {eff['mean_query_latency_seconds'] * 1000:.2f}ms")
    print(f"Max query latency:    {eff['max_query_latency_seconds'] * 1000:.2f}ms")
    print("-" * 60)
    print(f"Provisional score (80% weight, nDCG@10 + MAP@10): {lb['provisional_score_80pct']:.4f}")
    print("(Remaining ±10% efficiency modifier and 0-10% index-size score are")
    print(" class-relative — applied when course staff aggregate the full")
    print(" leaderboard, see instructor-tools/aggregate_leaderboard.py.)")
    if "beats_baseline_ndcg@10" in lb:
        print(f"Local reference nDCG@10: {lb['baseline_ndcg@10']:.4f}  (beat it: {lb['beats_baseline_ndcg@10']})")
        print(f"Local reference MAP@10:  {lb['baseline_map@10']:.4f}  (beat it: {lb['beats_baseline_map@10']})")
        print("(This is YOUR local comparison run, not the official grading baseline — see Section 7.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
