#!/usr/bin/env python3
"""Build once per collection, then sweep retrieve knobs. Prints nDCG@10."""
import argparse
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from harness.trec_io import read_queries, read_qrels
from harness.metrics import evaluate_run
from submission import retrieve as R


DATASETS = {
    "cranfield": "data/cranfield",
    "cacm": "data/cacm",
    "vaswani": "data/vaswani",
    "scifact": "data/proxy_scifact",
    "nfcorpus": "data/proxy_nfcorpus",
    "fiqa": "data/proxy_fiqa",
    "scidocs": "data/proxy_scidocs",
    "arguana": "data/proxy_arguana",
    "antique": "data/proxy_antique",
    "touche": "data/proxy_touche",
    "full": "data/full",
    "quora": "data/proxy_quora",
}


def ndcg(engine, queries, qrels, k=10, **kw):
    run = {}
    for qid, text in queries:
        if qid not in qrels:
            continue
        run[qid] = engine.retrieve(text, k, **kw)
    return evaluate_run(run, qrels, k=k)["aggregate"]["ndcg@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["cranfield", "scifact", "nfcorpus", "fiqa", "touche", "full"])
    ap.add_argument("--mode", choices=["diag", "news", "rm3", "covid"], default="diag")
    args = ap.parse_args()

    for name in args.sets:
        d = os.path.join(ROOT, DATASETS[name])
        corpus = os.path.join(d, "corpus.jsonl")
        if not os.path.isfile(corpus):
            print(f"SKIP {name}")
            continue
        queries = read_queries(os.path.join(d, "queries_dev.tsv"))
        qrels = read_qrels(os.path.join(d, "qrels_dev.txt"))
        print(f"\n======== {name} ========", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"tune_{name}_") as idx:
            eng = R.cpp_engine.CPPEngine()
            eng.build_index_from_file(corpus, idx)
            eng.load_index(idx)
            base = R.adapt_params(eng.N, eng.avg_doc_len)
            print(f"N={eng.N} avgdl={eng.avg_doc_len:.2f} tokens={eng.total_corpus_tokens}", flush=True)
            keys = ["k1", "b", "delta", "title_bonus", "w_title", "use_rm3",
                    "fb_docs", "fb_terms", "alpha", "mu", "rm3_pool"]
            print("adapt:", {k: base[k] for k in keys if k in base}, flush=True)

            def run_cfg(tag, **over):
                kw = dict(base)
                kw.update(over)
                s = ndcg(eng, queries, qrels, **kw)
                print(f"  {tag:42s} nDCG@10={s:.4f}", flush=True)
                return s

            run_cfg("current")
            if args.mode == "diag":
                continue

            if args.mode in ("news", "covid"):
                for k1 in (0.9, 1.2, 1.6, 2.0, 2.2):
                    for b in (0.30, 0.40, 0.50, 0.75):
                        run_cfg(f"k1={k1} b={b}", k1=k1, b=b, use_rm3=False)
                for delta in (0.0, 0.25, 0.5, 1.0):
                    run_cfg(f"delta={delta} no_rm3", delta=delta, use_rm3=False)
                for tb in (0.0, 0.3, 0.5, 0.75, 1.0):
                    run_cfg(f"title_bonus={tb} no_rm3", title_bonus=tb, use_rm3=False)

            if args.mode in ("rm3", "news", "covid"):
                run_cfg("rm3_off", use_rm3=False)
                for alpha in (0.4, 0.5, 0.6, 0.7):
                    for fbt in (10, 20, 40):
                        for fbd in (10, 20, 30):
                            run_cfg(f"rm3 a={alpha} t={fbt} d={fbd}",
                                    use_rm3=True, alpha=alpha, fb_terms=fbt, fb_docs=fbd)


if __name__ == "__main__":
    main()
