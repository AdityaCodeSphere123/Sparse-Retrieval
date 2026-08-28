"""
scripts/tune.py — dev-set parameter search for the retrieval parameters.

Why this exists
---------------
Every scoring parameter in this system is a QUERY-TIME argument, so a full
sweep needs the index built exactly once and then re-queries it in-process.

The two exceptions are the analysis-chain choices (stemmer, stopwords), which
are baked into the index at build time. Those are swept by rebuilding, which
this script does for you with --sweep analysis.

nDCG@10 here uses the harness formula ((2^rel - 1) / log2(rank+1)), so the
number you see is the one the leaderboard uses — not the linear-gain variant.

Usage
-----
    # build the index once, then reuse it
    python -m scripts.tune --corpus data/full/corpus.jsonl \
        --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt \
        --index-dir /tmp/tune_index --build

    # default: every combination of k1,b,delta,RM3 knobs; print the winner
    python -m scripts.tune \
        --queries data/full/queries_dev.tsv --qrels data/full/qrels_dev.txt \
        --index-dir /tmp/tune_index

    # same, but tune on half the queries and report the other half
    python -m scripts.tune ... --index-dir /tmp/tune_index --split

    # denser cartesian product (more points, slower)
    python -m scripts.tune ... --index-dir /tmp/tune_index --dense

    # also try SDM proximity weights (no-op unless positions.bin is populated)
    python -m scripts.tune ... --index-dir /tmp/tune_index --include-sdm

    # nDCG@10 vs k1 and b only (the plot the report asks for)
    python -m scripts.tune ... --index-dir /tmp/tune_index --sweep k1b \
        --csv runs/sweep_k1b.csv

    python -m scripts.tune ... --index-dir /tmp/tune_index --sweep ablation
    python -m scripts.tune ... --index-dir /tmp/tune_index --sweep auto --split
    python -m scripts.tune --corpus data/full/corpus.jsonl ... --sweep analysis
"""
import argparse
import csv
import itertools
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

from harness.metrics import average_precision, ndcg_at_k


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


def evaluate(runner, queries, qrels, k=10) -> Tuple[float, float]:
    """Macro-average nDCG@k / MAP@k with the same formulae as the harness."""
    nd, ap, n = 0.0, 0.0, 0
    for qid, text in queries:
        rel = qrels.get(qid, {})
        if not rel:
            continue
        ranked = [d for d, _ in runner(text, k)]
        nd += ndcg_at_k(ranked, rel, k)
        ap += average_precision(ranked[:k], rel)
        n += 1
    return (nd / n, ap / n) if n else (0.0, 0.0)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
BASE = dict(
    k1=1.2, b=0.75, delta=1.0,
    use_rm3=True, fb_docs=10, fb_terms=25, alpha=0.6, mu=1000.0,
    use_sdm=False, sdm_rerank_depth=200,
    lambda_o=0.0, lambda_u=0.0, window_u=8, sdm_normalise=True,
)

# Fine 1-D grids used by --sweep one / auto / the polish pass after --sweep grid.
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

# Default cartesian product for --sweep grid. Dead RM3 knobs are not expanded
# when use_rm3=False, so this is every *distinct* ranking, not a naive product.
JOINT = {
    "k1":       [0.9, 1.2, 1.5, 1.8, 2.1, 2.5],
    "b":        [0.4, 0.6, 0.75, 0.9],
    "delta":    [0.0, 0.25, 1.0],
    "alpha":    [0.5, 0.6, 0.7, 0.8],
    "fb_docs":  [5, 10, 20],
    "fb_terms": [10, 25],
    "mu":       [1000.0, 2000.0],
    "lambda_o": [0.05, 0.10, 0.20],
    "lambda_u": [0.0, 0.05, 0.10],
    "window_u": [8],
}

# --dense: more points, still skips identical RM3-off duplicates.
DENSE = {
    "k1":       [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5],
    "b":        [0.3, 0.4, 0.5, 0.6, 0.75, 0.9],
    "delta":    [0.0, 0.25, 0.5, 1.0],
    "alpha":    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "fb_docs":  [5, 10, 15, 20],
    "fb_terms": [10, 20, 30],
    "mu":       [500.0, 1000.0, 2000.0],
    "lambda_o": [0.0, 0.05, 0.10, 0.20],
    "lambda_u": [0.0, 0.05, 0.10],
    "window_u": [4, 8],
}

ASCENT_ORDER = ["b", "k1", "delta", "alpha", "fb_terms", "fb_docs",
                "mu", "lambda_o", "lambda_u", "window_u"]


def make_runner(engine, params):
    def run(query, k):
        return engine.retrieve(query, k, **params)
    return run


def pack(k1, b, delta, use_rm3=False, fb_docs=10, fb_terms=25, alpha=0.6, mu=1000.0,
         use_sdm=False, lambda_o=0.0, lambda_u=0.0, window_u=8) -> dict:
    return dict(
        k1=k1, b=b, delta=delta,
        use_rm3=use_rm3, fb_docs=fb_docs, fb_terms=fb_terms, alpha=alpha, mu=mu,
        use_sdm=use_sdm, sdm_rerank_depth=200,
        lambda_o=lambda_o, lambda_u=lambda_u, window_u=window_u, sdm_normalise=True,
    )


def iter_grid(space: dict, include_sdm: bool) -> Iterable[dict]:
    """Yield every distinct (BM25 × optional RM3 × optional SDM) configuration."""
    bm25 = itertools.product(space["k1"], space["b"], space["delta"])
    rm3 = itertools.product(space["alpha"], space["fb_docs"], space["fb_terms"], space["mu"])
    sdm = itertools.product(space["lambda_o"], space["lambda_u"], space["window_u"])
    # Materialise RM3/SDM once; BM25 is cheap to restream.
    rm3_list = list(rm3)
    sdm_list = list(sdm)
    for k1, b, delta in bm25:
        yield pack(k1, b, delta, use_rm3=False, use_sdm=False)
        for alpha, fb_docs, fb_terms, mu in rm3_list:
            yield pack(k1, b, delta, True, fb_docs, fb_terms, alpha, mu,
                       use_sdm=False)
            if include_sdm:
                for lambda_o, lambda_u, window_u in sdm_list:
                    if lambda_o == 0.0 and lambda_u == 0.0:
                        continue
                    yield pack(k1, b, delta, True, fb_docs, fb_terms, alpha, mu,
                               True, lambda_o, lambda_u, window_u)


def count_grid(space: dict, include_sdm: bool) -> int:
    n_bm25 = len(space["k1"]) * len(space["b"]) * len(space["delta"])
    n_rm3 = (len(space["alpha"]) * len(space["fb_docs"])
             * len(space["fb_terms"]) * len(space["mu"]))
    n_sdm_live = sum(1 for lo, lu, _ in itertools.product(
        space["lambda_o"], space["lambda_u"], space["window_u"])
                     if not (lo == 0.0 and lu == 0.0))
    total = n_bm25 * (1 + n_rm3)
    if include_sdm:
        total += n_bm25 * n_rm3 * n_sdm_live
    return total


def print_params(params: dict) -> None:
    print("PARAMS = dict(")
    for kk, vv in params.items():
        print(f"    {kk}={vv!r},")
    print(")")


def coordinate_ascent(start: dict, score_fn, rounds: int = 2) -> Tuple[dict, float]:
    cur = dict(start)
    best_nd, _ = score_fn(cur)
    for rnd in range(rounds):
        improved = False
        for name in ASCENT_ORDER:
            if name in ("lambda_o", "lambda_u", "window_u") and not cur.get("use_sdm"):
                continue
            trial_best, trial_val = best_nd, cur[name]
            for v in GRIDS[name]:
                if v == cur[name]:
                    continue
                p = dict(cur, **{name: v})
                nd, _ = score_fn(p)
                if nd > trial_best + 1e-6:
                    trial_best, trial_val = nd, v
            if trial_val != cur[name]:
                print(f"  polish {rnd+1}: {name} {cur[name]} -> {trial_val}"
                      f"   nDCG {best_nd:.4f} -> {trial_best:.4f}")
                cur[name] = trial_val
                best_nd = trial_best
                improved = True
        for flag in ("use_rm3", "use_sdm"):
            p = dict(cur, **{flag: not cur[flag]})
            nd, _ = score_fn(p)
            if nd > best_nd + 1e-6:
                print(f"  polish {rnd+1}: {flag} {cur[flag]} -> {not cur[flag]}"
                      f"   nDCG {best_nd:.4f} -> {nd:.4f}")
                cur[flag] = not cur[flag]
                best_nd = nd
                improved = True
        if not improved:
            break
    return cur, best_nd


def run_cartesian(score_fn, space: dict, include_sdm: bool, top_n: int,
                  polish: bool, polish_starts: int) -> Tuple[dict, float, float, list]:
    n_total = count_grid(space, include_sdm)
    print(f"\ncartesian grid: {n_total} distinct configurations"
          + ("  (including SDM)" if include_sdm else "  (SDM off)"))
    print("scoring with harness nDCG@10  ((2^rel-1) / log2(rank+1))")

    ranked: List[Tuple[float, float, dict]] = []
    best: Optional[Tuple[float, float, dict]] = None
    t0 = time.time()
    for i, params in enumerate(iter_grid(space, include_sdm), start=1):
        nd, mp = score_fn(params)
        ranked.append((nd, mp, params))
        if best is None or nd > best[0] + 1e-12 or (abs(nd - best[0]) <= 1e-12 and mp > best[1]):
            best = (nd, mp, params)
            print(f"  [{i:>6}/{n_total}] NEW BEST  nDCG@10={nd:.4f}  "
                  f"k1={params['k1']} b={params['b']} delta={params['delta']}  "
                  f"rm3={params['use_rm3']} fb_docs={params['fb_docs']} "
                  f"fb_terms={params['fb_terms']} alpha={params['alpha']} mu={params['mu']}")
        elif i % 50 == 0 or i == n_total:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (n_total - i)
            print(f"  [{i:>6}/{n_total}] nDCG={nd:.4f}  best={best[0]:.4f}  "
                  f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m")

    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = ranked[:top_n]

    print(f"\n--- top {len(top)} of {n_total} ---")
    print(f"{'nDCG@10':>9} {'MAP@10':>8}  k1    b   delta  rm3  fb_d fb_t  alpha     mu  sdm")
    for nd, mp, p in top:
        print(f"{nd:>9.4f} {mp:>8.4f}  {p['k1']:<5} {p['b']:<4} {p['delta']:<6} "
              f"{str(p['use_rm3']):<4} {p['fb_docs']:<4} {p['fb_terms']:<4} "
              f"{p['alpha']:<6} {p['mu']:<6} {p['use_sdm']}")

    winner, best_nd, best_mp = top[0][2], top[0][0], top[0][1]
    if polish:
        print(f"\npolishing the top {min(polish_starts, len(top))} grid hits "
              f"with 1-D coordinate ascent ...")
        for nd0, _, start in top[:polish_starts]:
            print(f"\n  start nDCG@10={nd0:.4f}")
            cur, nd1 = coordinate_ascent(start, score_fn, rounds=2)
            mp1 = score_fn(cur)[1]
            if nd1 > best_nd + 1e-6:
                winner, best_nd, best_mp = cur, nd1, mp1
    return winner, best_nd, best_mp, top


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--queries", required=True)
    ap_.add_argument("--qrels", required=True)
    ap_.add_argument("--index-dir", default="/tmp/tune_index")
    ap_.add_argument("--corpus", default=None, help="required with --build or --sweep analysis")
    ap_.add_argument("--build", action="store_true", help="(re)build the index first")
    ap_.add_argument("--sweep", default="grid",
                     choices=["grid", "auto", "k1b", "ablation", "analysis", "one"],
                     help="grid = all combinations (default). auto = coordinate ascent only.")
    ap_.add_argument("--dense", action="store_true",
                     help="with --sweep grid, use the denser cartesian product")
    ap_.add_argument("--include-sdm", action="store_true",
                     help="also enumerate SDM weights (skip if positions.bin is empty)")
    ap_.add_argument("--no-polish", action="store_true",
                     help="skip the coordinate-ascent refinement after the grid")
    ap_.add_argument("--top", type=int, default=10, help="how many grid hits to print")
    ap_.add_argument("--polish-starts", type=int, default=3,
                     help="run coordinate ascent from this many top grid configs")
    ap_.add_argument("--param", default=None, help="parameter name for --sweep one")
    ap_.add_argument("--split", action="store_true",
                     help="tune on even queries, report on odd queries")
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
                size = sum(os.path.getsize(os.path.join(d, f)) / 1e6
                           for f in os.listdir(d))
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

    def score(params):
        return evaluate(make_runner(engine, params), tune_q, qrels, args.k)

    rows: list = []

    if args.sweep == "ablation":
        variants = {
            "BM25 only":                dict(BASE, use_rm3=False, use_sdm=False),
            "BM25 + RM3":               dict(BASE, use_rm3=True, use_sdm=False),
            "BM25 + SDM":               dict(BASE, use_rm3=False, use_sdm=True),
            "BM25 + RM3 + SDM (full)":  dict(BASE, use_rm3=True, use_sdm=True,
                                             lambda_o=0.10, lambda_u=0.05),
            "BM25 (delta=0, plain)":    dict(BASE, use_rm3=False, use_sdm=False, delta=0.0),
        }
        print(f"\n{'configuration':<26} {'nDCG@10':>9} {'MAP@10':>9}")
        for name, p in variants.items():
            nd, mp = score(p)
            print(f"{name:<26} {nd:>9.4f} {mp:>9.4f}")
            rows.append(dict(config=name, ndcg=nd, map=mp))

    elif args.sweep == "k1b":
        print(f"\n{'k1':>5} {'b':>6} {'nDCG@10':>9} {'MAP@10':>9}   (RM3/SDM off)")
        best = None
        for k1, b in itertools.product(GRIDS["k1"], GRIDS["b"]):
            p = pack(k1, b, 0.0, use_rm3=False, use_sdm=False)
            nd, mp = score(p)
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
            nd, mp = score(p)
            rows.append({args.param: v, "ndcg": nd, "map": mp})
            print(f"{v:>10} {nd:>9.4f} {mp:>9.4f}")

    elif args.sweep == "auto":
        cur = dict(BASE)
        base_nd, _ = score(cur)
        print(f"\nstart nDCG@10 = {base_nd:.4f}")
        cur, _ = coordinate_ascent(cur, score, rounds=2)
        nd_t, mp_t = score(cur)
        print("\n--- best configuration found ---")
        print_params(cur)
        print(f"\ntune nDCG@10={nd_t:.4f}  MAP@10={mp_t:.4f}")
        if val_q:
            nd_v, mp_v = evaluate(make_runner(engine, cur), val_q, qrels, args.k)
            nd_b, _ = evaluate(make_runner(engine, BASE), val_q, qrels, args.k)
            print(f"validate nDCG@10={nd_v:.4f}  MAP@10={mp_v:.4f}"
                  f"   (defaults on the same half: {nd_b:.4f})")
            print("\nIf the validate-half gain is much smaller than the tune-half gain,\n"
                  "the extra points are dev-set noise and will not transfer.")
        rows.append(dict(cur, ndcg=nd_t, map=mp_t))

    else:  # grid — every combination, then polish the winners
        space = DENSE if args.dense else JOINT
        winner, nd_t, mp_t, top = run_cartesian(
            score, space, args.include_sdm, args.top,
            polish=not args.no_polish, polish_starts=args.polish_starts,
        )
        print("\n--- best configuration found ---")
        print_params(winner)
        print(f"\ntune nDCG@10={nd_t:.4f}  MAP@10={mp_t:.4f}")
        if val_q:
            nd_v, mp_v = evaluate(make_runner(engine, winner), val_q, qrels, args.k)
            nd_b, _ = evaluate(make_runner(engine, BASE), val_q, qrels, args.k)
            print(f"validate nDCG@10={nd_v:.4f}  MAP@10={mp_v:.4f}"
                  f"   (defaults on the same half: {nd_b:.4f})")
            print("\nIf the validate-half gain is much smaller than the tune-half gain,\n"
                  "the extra points are dev-set noise and will not transfer.")
        for nd, mp, p in top:
            rows.append(dict(p, ndcg=nd, map=mp))
        rows.append(dict(winner, ndcg=nd_t, map=mp_t, polished=True))

    if args.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
