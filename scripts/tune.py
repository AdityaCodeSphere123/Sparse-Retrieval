"""
scripts/tune.py — dev-set parameter search for the retrieval parameters.

Why this exists
---------------
Every scoring parameter in this system is a QUERY-TIME argument, so a full
sweep needs the index built exactly once and then re-queries it in-process.
A 400-point grid over 50 dev queries takes seconds, not hours.

The two exceptions are the analysis-chain choices (stemmer, stopwords), which
are baked into the index at build time. Those are swept by rebuilding, which
this script does for you with --sweep analysis.

Usage
-----
    # one-off: build the index once, reuse it for every sweep below
    python -m scripts.tune --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt \
        --index-dir /tmp/tune_index --build

    # the sweep the report asks for (Section 8): nDCG@10 vs k1 and b
    python -m scripts.tune --queries ... --qrels ... --index-dir /tmp/tune_index \
        --sweep k1b --csv runs/sweep_k1b.csv

    # is RM3 / SDM earning its place at all?
    python -m scripts.tune --queries ... --qrels ... --index-dir /tmp/tune_index \
        --sweep ablation

    # greedy coordinate ascent over everything, honest about overfitting
    python -m scripts.tune --queries ... --qrels ... --index-dir /tmp/tune_index \
        --sweep auto --split

    # does Porter + stopword removal actually help on THIS corpus?
    python -m scripts.tune --corpus data/full/corpus.jsonl --queries ... --qrels ... \
        --sweep analysis

A warning that is worth taking seriously
----------------------------------------
Assignment Section 1.1 lists "experience, first-hand, how leaderboard-driven
tuning can overfit to a public dev set" as a learning objective, and the
private held-out set exists to punish exactly that. --split tunes on half the
queries and reports the score on the other half. If tune-half and
validate-half disagree sharply, the gain is noise and will not survive to the
held-out topics. Report the validation number, not the tuning number.

The metrics here are for RANKING CONFIGURATIONS AGAINST EACH OTHER. Once you
have a winner, confirm it by running the real harness, which is the code that
actually grades you.
"""
import argparse
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_queries(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                r = int(rel)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[docid] = r
    return qrels


# ---------------------------------------------------------------------------
# Metrics (trec_eval conventions: linear gain, log2(rank+1) discount)
# ---------------------------------------------------------------------------
def ndcg_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        g = rel.get(doc, 0)
        if g > 0:
            dcg += g / math.log2(i + 2)
    ideal = sorted([v for v in rel.values() if v > 0], reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    """Average Precision over the top k, normalised by the TRUE total number
    of relevant documents (assignment Section 4's definition of MAP@10)."""
    total_rel = sum(1 for v in rel.values() if v > 0)
    if total_rel == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, doc in enumerate(ranked[:k]):
        if rel.get(doc, 0) > 0:
            hits += 1
            s += hits / (i + 1)
    return s / total_rel


def evaluate(runner, queries, qrels, k=10) -> Tuple[float, float]:
    nd, ap, n = 0.0, 0.0, 0
    for qid, text in queries:
        rel = qrels.get(qid, {})
        if not rel:
            continue
        ranked = [d for d, _ in runner(text, k)]
        nd += ndcg_at_k(ranked, rel, k)
        ap += ap_at_k(ranked, rel, k)
        n += 1
    return (nd / n, ap / n) if n else (0.0, 0.0)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
BASE = dict(
    k1=1.2, b=0.75, delta=1.0,
    use_rm3=True, fb_docs=10, fb_terms=25, alpha=0.6, mu=1000.0,
    use_sdm=True, sdm_rerank_depth=200,
    lambda_o=0.10, lambda_u=0.05, window_u=8, sdm_normalise=True,
)

GRIDS = {
    "k1":       [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5],
    "b":        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9],
    "delta":    [0.0, 0.25, 0.5, 1.0, 1.5],
    "alpha":    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "fb_docs":  [5, 10, 15, 20, 30],
    "fb_terms": [10, 20, 30, 50, 80],
    "mu":       [500.0, 1000.0, 2000.0],
    "lambda_o": [0.0, 0.05, 0.10, 0.20, 0.30],
    "lambda_u": [0.0, 0.05, 0.10, 0.20],
    "window_u": [4, 8, 12],
}

# Order matters: BM25 first (it dominates), then expansion, then proximity.
ASCENT_ORDER = ["b", "k1", "delta", "alpha", "fb_terms", "fb_docs",
                "mu", "lambda_o", "lambda_u", "window_u"]


def make_runner(engine, params):
    def run(query, k):
        return engine.retrieve(query, k, **params)
    return run


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--queries", required=True)
    ap_.add_argument("--qrels", required=True)
    ap_.add_argument("--index-dir", default="/tmp/tune_index")
    ap_.add_argument("--corpus", default=None, help="required with --build or --sweep analysis")
    ap_.add_argument("--build", action="store_true", help="(re)build the index first")
    ap_.add_argument("--sweep", default="auto",
                     choices=["auto", "k1b", "ablation", "analysis", "one"])
    ap_.add_argument("--param", default=None, help="parameter name for --sweep one")
    ap_.add_argument("--split", action="store_true",
                     help="tune on odd queries, report on even queries")
    ap_.add_argument("--csv", default=None)
    ap_.add_argument("--k", type=int, default=10)
    args = ap_.parse_args()

    sys.path.insert(0, os.getcwd())
    from submission import retrieve as sub

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    queries = [q for q in queries if qrels.get(q[0])]
    if not queries:
        raise SystemExit("No queries have qrels — check the file formats.")

    if args.split:
        tune_q = [q for i, q in enumerate(queries) if i % 2 == 0]
        val_q = [q for i, q in enumerate(queries) if i % 2 == 1]
    else:
        tune_q, val_q = queries, None
    print(f"{len(queries)} queries with qrels"
          + (f"  (tune {len(tune_q)} / validate {len(val_q)})" if val_q else ""))

    # ---------------- analysis sweep needs a rebuild per configuration ------
    if args.sweep == "analysis":
        if not args.corpus:
            raise SystemExit("--sweep analysis needs --corpus")
        print(f"\n{'stemmer':>8} {'stopwords':>10} {'nDCG@10':>9} {'MAP@10':>9} {'index MB':>9}")
        rows = []
        for stemmer in (1, 2):
            for stop in (0, 1):
                d = f"{args.index_dir}_s{stemmer}_w{stop}"
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)
                env = dict(os.environ, IR_STEMMER=str(stemmer), IR_STOPWORDS=str(stop))
                subprocess.run(
                    [sys.executable, "-c",
                     "import sys;sys.path.insert(0,'.');from submission import retrieve as r;"
                     f"r.build_index({args.corpus!r}, {d!r})"],
                    env=env, check=True)
                sub.load_index(d)
                size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6
                nd, mp = evaluate(make_runner(sub._CPP_ENGINE, BASE), tune_q, qrels, args.k)
                name = {1: "s-stem", 2: "porter"}[stemmer]
                print(f"{name:>8} {stop:>10} {nd:>9.4f} {mp:>9.4f} {size:>9.1f}")
                rows.append(dict(stemmer=name, stopwords=stop, ndcg=nd, map=mp, size_mb=size))
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print("\nRebuild the index with the winning IR_STEMMER / IR_STOPWORDS before "
              "running the other sweeps.")
        return

    # ---------------- everything else reuses one index ----------------------
    if args.build:
        if not args.corpus:
            raise SystemExit("--build needs --corpus")
        shutil.rmtree(args.index_dir, ignore_errors=True)
        os.makedirs(args.index_dir, exist_ok=True)
        print("building index ...")
        sub.build_index(args.corpus, args.index_dir)

    sub.load_index(args.index_dir)
    engine = sub._CPP_ENGINE
    if engine is None:
        raise SystemExit("C++ extension not loaded — sweeping the Python fallback "
                         "would measure the wrong system.")

    def score(params, qs):
        return evaluate(make_runner(engine, params), qs, qrels, args.k)

    rows = []

    if args.sweep == "ablation":
        variants = {
            "BM25 only":                dict(BASE, use_rm3=False, use_sdm=False),
            "BM25 + RM3":               dict(BASE, use_rm3=True, use_sdm=False),
            "BM25 + SDM":               dict(BASE, use_rm3=False, use_sdm=True),
            "BM25 + RM3 + SDM (full)":  dict(BASE),
            "BM25 (delta=0, plain)":    dict(BASE, use_rm3=False, use_sdm=False, delta=0.0),
        }
        print(f"\n{'configuration':<26} {'nDCG@10':>9} {'MAP@10':>9}")
        for name, p in variants.items():
            nd, mp = score(p, tune_q)
            print(f"{name:<26} {nd:>9.4f} {mp:>9.4f}")
            rows.append(dict(config=name, ndcg=nd, map=mp))

    elif args.sweep == "k1b":
        print(f"\n{'k1':>5} {'b':>6} {'nDCG@10':>9} {'MAP@10':>9}   (RM3/SDM off)")
        best = None
        for k1, b in itertools.product(GRIDS["k1"], GRIDS["b"]):
            p = dict(BASE, k1=k1, b=b, use_rm3=False, use_sdm=False)
            nd, mp = score(p, tune_q)
            rows.append(dict(k1=k1, b=b, ndcg=nd, map=mp))
            print(f"{k1:>5.2f} {b:>6.2f} {nd:>9.4f} {mp:>9.4f}")
            if best is None or nd > best[0]:
                best = (nd, k1, b)
        print(f"\nbest BM25: k1={best[1]} b={best[2]}  nDCG@10={best[0]:.4f}")

    elif args.sweep == "one":
        if args.param not in GRIDS:
            raise SystemExit(f"--param must be one of {sorted(GRIDS)}")
        print(f"\n{args.param:>10} {'nDCG@10':>9} {'MAP@10':>9}")
        for v in GRIDS[args.param]:
            p = dict(BASE, **{args.param: v})
            nd, mp = score(p, tune_q)
            rows.append({args.param: v, "ndcg": nd, "map": mp})
            print(f"{v:>10} {nd:>9.4f} {mp:>9.4f}")

    else:  # auto — greedy coordinate ascent
        cur = dict(BASE)
        base_nd, _ = score(cur, tune_q)
        print(f"\nstart nDCG@10 = {base_nd:.4f}")
        best_nd = base_nd
        for rounds in range(2):
            improved = False
            for name in ASCENT_ORDER:
                trial_best, trial_val = best_nd, cur[name]
                for v in GRIDS[name]:
                    if v == cur[name]:
                        continue
                    p = dict(cur, **{name: v})
                    nd, _ = score(p, tune_q)
                    if nd > trial_best + 1e-6:
                        trial_best, trial_val = nd, v
                if trial_val != cur[name]:
                    print(f"  round {rounds+1}: {name} {cur[name]} -> {trial_val}"
                          f"   nDCG {best_nd:.4f} -> {trial_best:.4f}")
                    cur[name] = trial_val
                    best_nd = trial_best
                    improved = True
            # RM3 and SDM are worth testing as switches, not just as weights
            for flag in ("use_rm3", "use_sdm"):
                p = dict(cur, **{flag: not cur[flag]})
                nd, _ = score(p, tune_q)
                if nd > best_nd + 1e-6:
                    print(f"  round {rounds+1}: {flag} {cur[flag]} -> {not cur[flag]}"
                          f"   nDCG {best_nd:.4f} -> {nd:.4f}")
                    cur[flag] = not cur[flag]
                    best_nd = nd
                    improved = True
            if not improved:
                break

        print("\n--- best configuration found ---")
        print("PARAMS = dict(")
        for kk, vv in cur.items():
            print(f"    {kk}={vv!r},")
        print(")")
        nd_t, mp_t = score(cur, tune_q)
        print(f"\ntune half:     nDCG@10={nd_t:.4f}  MAP@10={mp_t:.4f}")
        if val_q:
            nd_v, mp_v = score(cur, val_q)
            nd_b, _ = score(BASE, val_q)
            print(f"validate half: nDCG@10={nd_v:.4f}  MAP@10={mp_v:.4f}"
                  f"   (defaults on the same half: {nd_b:.4f})")
            print("\nIf the validate-half gain is much smaller than the tune-half gain,\n"
                  "the extra points are dev-set noise and will not transfer.")
        rows.append(dict(cur, ndcg=nd_t, map=mp_t))

    if args.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()