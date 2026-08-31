#!/usr/bin/env python
"""First-pass k1/b sweep via retrieve_bm25_only (no RM3).

Build+load once, then score nDCG@10 on a grid. Writes a CSV and, if
matplotlib is installed, a heatmap PNG for the report.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import retrieve as r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--index-dir", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", default="0.6,0.9,1.2,1.5,1.8,2.2")
    parser.add_argument("--b", default="0.2,0.4,0.6,0.75,0.9")
    parser.add_argument("--out-csv", default=os.path.join(ROOT, "runs", "sweep_k1_b.csv"))
    parser.add_argument("--out-png", default=os.path.join(ROOT, "runs", "sweep_k1_b.png"))
    args = parser.parse_args()

    k1s = [float(x) for x in args.k1.split(",") if x.strip()]
    bs = [float(x) for x in args.b.split(",") if x.strip()]
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    index_dir = args.index_dir or os.path.join(os.path.dirname(args.corpus), "_index")
    os.makedirs(index_dir, exist_ok=True)
    print(f"build_index -> {index_dir}", flush=True)
    r.build_index(args.corpus, index_dir)
    r.load_index(index_dir)
    print(
        f"N={r._CPP_ENGINE.N} avgdl={r._CPP_ENGINE.avg_doc_len:.2f} "
        f"adapt k1={r._ACTIVE['k1']:.3f} b={r._ACTIVE['b']:.3f}",
        flush=True,
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    best = None
    for k1 in k1s:
        for b in bs:
            run = {}
            for qid, qtext in queries:
                run[qid] = r.retrieve_bm25_only(qtext, k=args.k, k1=k1, b=b)
            metrics = evaluate_run(run, qrels, k=args.k)["aggregate"]
            ndcg = metrics["ndcg@10"]
            rows.append({"k1": k1, "b": b, "ndcg@10": ndcg, "map@10": metrics["map@10"]})
            print(f"k1={k1:.2f} b={b:.2f}  nDCG@10={ndcg:.4f}  MAP@10={metrics['map@10']:.4f}", flush=True)
            if best is None or ndcg > best[0]:
                best = (ndcg, k1, b)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k1", "b", "ndcg@10", "map@10"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out_csv}")
    if best:
        print(f"best first-pass: nDCG@10={best[0]:.4f} at k1={best[1]} b={best[2]}")
        print(f"_NEWS prior is k1=0.9 b=0.4 — {'match' if abs(best[1]-0.9)<1e-9 and abs(best[2]-0.4)<1e-9 else 'use the measured point'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; CSV only")
        return

    grid = np.zeros((len(bs), len(k1s)))
    for row in rows:
        i = k1s.index(row["k1"])
        j = bs.index(row["b"])
        grid[j, i] = row["ndcg@10"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(k1s)))
    ax.set_xticklabels([f"{v:.2f}" for v in k1s])
    ax.set_yticks(range(len(bs)))
    ax.set_yticklabels([f"{v:.2f}" for v in bs])
    ax.set_xlabel("k1")
    ax.set_ylabel("b")
    ax.set_title("nDCG@10  retrieve_bm25_only")
    fig.colorbar(im, ax=ax, label="nDCG@10")
    if best:
        ax.scatter([k1s.index(best[1])], [bs.index(best[2])], c="red", marker="x", s=80)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
