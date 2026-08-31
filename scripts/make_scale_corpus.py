#!/usr/bin/env python
"""Synthetic Robust04-shaped JSONL: Zipfian vocab, lognormal lengths, mixed ids.

Usage:
    python scripts/make_scale_corpus.py --out /tmp/scale.jsonl --n-docs 80000
    python scripts/make_scale_corpus.py --out /tmp/scale.jsonl --n-docs 528155
"""
from __future__ import annotations

import argparse
import math
import random


PREFIXES = ("FBIS3", "FT911", "FT921", "FT931", "FT941", "FR94", "LA")


def _zipf_ranks(n_types: int, s: float = 1.1) -> list[float]:
    weights = [1.0 / ((i + 1) ** s) for i in range(n_types)]
    z = sum(weights)
    return [w / z for w in weights]


def make_id(i: int) -> str:
    fam = PREFIXES[i % len(PREFIXES)]
    if fam == "LA":
        return f"LA{10189 + (i % 365):06d}-{i % 9000:04d}"
    if fam.startswith("FT"):
        return f"{fam}-{1 + (i % 99999)}"
    if fam == "FR94":
        return f"FR940104-0-{i % 99999:05d}"
    return f"FBIS3-{1 + (i % 999999)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n-docs", type=int, default=80_000)
    p.add_argument("--vocab", type=int, default=400_000)
    p.add_argument("--mean-tokens", type=float, default=475.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_types = args.vocab
    # Distinct surface forms; some collide after we skip ultra-short names.
    vocab = []
    seen = set()
    i = 0
    while len(vocab) < n_types and i < n_types * 4:
        w = f"w{i:x}"
        i += 1
        if len(w) <= 1 or w in seen:
            continue
        seen.add(w)
        vocab.append(w)
    cdf = []
    acc = 0.0
    for w in _zipf_ranks(len(vocab)):
        acc += w
        cdf.append(acc)

    def draw_term() -> str:
        u = rng.random()
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return vocab[lo]

    sigma = 0.55
    mu = math.log(args.mean_tokens) - 0.5 * sigma * sigma
    n_written = 0
    tok_sum = 0
    mn, mx = 10**9, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for d in range(args.n_docs):
            n = int(rng.lognormvariate(mu, sigma))
            n = max(40, min(n, 4000))
            toks = [draw_term() for _ in range(n)]
            rec = '{"doc_id": "%s", "text": "%s"}\n' % (make_id(d), " ".join(toks))
            f.write(rec)
            n_written += 1
            tok_sum += n
            mn = min(mn, n)
            mx = max(mx, n)
            if n_written % 20000 == 0:
                print(f"  {n_written} docs, mean so far {tok_sum / n_written:.1f}", flush=True)
    print(
        f"Wrote {n_written} docs to {args.out}  "
        f"mean_tokens={tok_sum / max(n_written, 1):.1f}  min={mn} max={mx}  types={len(vocab)}"
    )


if __name__ == "__main__":
    main()
