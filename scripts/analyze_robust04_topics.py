#!/usr/bin/env python
"""Tokenize public Robust04 titles with the competition analysis chain.

The corpus is licensed; the topic *titles* are public TREC statements.
This script does not download Disks 4/5. It reports query length after
Porter+stops (whether qn<=4 would fire) and title tokens dropped as
stopwords or length-1 junk.
"""
from __future__ import annotations

import argparse
import os
import re
import gzip
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "data", "robust04_topics", "titles.tsv")

TOPIC_URLS = [
    "https://trec.nist.gov/data/robust/04.testset.gz",
]


def _parse_trec_topics(text: str) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    if "<top>" in text.lower() or "<num>" in text.lower():
        blocks = re.split(r"(?i)<top>", text)
        for block in blocks:
            mnum = re.search(r"(?i)<num>\s*(?:Number:)?\s*(\d+)", block)
            mtitle = re.search(r"(?i)<title>\s*(.*?)(?:<desc>|<narr>|$)", block, re.S)
            if not mnum or not mtitle:
                continue
            title = re.sub(r"\s+", " ", mtitle.group(1)).strip()
            title = re.sub(r"(?i)^Topic:\s*", "", title).strip()
            if title:
                topics.append((mnum.group(1), title))
        return topics
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            qid, title = line.split("\t", 1)
            topics.append((qid.strip(), title.strip()))
        else:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                topics.append((parts[0], parts[1]))
    return topics


def fetch_topics(path: str) -> list[tuple[str, str]]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [(a, b) for a, b in (ln.rstrip("\n").split("\t", 1) for ln in f if ln.strip())]

    last_err = None
    raw = ""
    for url in TOPIC_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "COL764-A1-topic-fetch/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                blob = resp.read()
            if url.endswith(".gz") or blob[:2] == b"\x1f\x8b":
                blob = gzip.decompress(blob)
            raw = blob.decode("utf-8", errors="replace")
            if raw.strip():
                break
        except Exception as e:
            last_err = e
            raw = ""
    if not raw.strip():
        raise SystemExit(f"Could not download Robust04 titles ({last_err})")
    topics = _parse_trec_topics(raw)
    if not topics:
        raise SystemExit("Downloaded topic file but parsed 0 titles")
    with open(path, "w", encoding="utf-8") as f:
        for qid, title in topics:
            f.write(f"{qid}\t{title}\n")
    return topics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        import cpp_engine
    except ImportError:
        from submission import cpp_engine

    topics = fetch_topics(args.out)
    n = len(topics)
    qn_le4 = 0
    stop_hits: dict[str, int] = {}
    dropped_examples: list[tuple[str, str, list[str]]] = []

    print(f"Loaded {n} Robust04 titles from {args.out}")
    print(f"{'qid':<8} {'qn':>3}  title")
    print("-" * 72)
    for qid, title in topics:
        kept = cpp_engine.tokenize(title, True)
        raw = cpp_engine.tokenize(title, False)
        qn = len(kept)
        if qn <= 4:
            qn_le4 += 1
        dropped = [t for t in raw if t not in kept]
        for t in dropped:
            stop_hits[t] = stop_hits.get(t, 0) + 1
        if dropped and len(dropped_examples) < 15:
            dropped_examples.append((qid, title, dropped))
        flag = "  <-- qn<=4" if qn <= 4 else ""
        print(f"{qid:<8} {qn:>3}  {title}{flag}")

    print()
    print(f"qn <= 4 (newswire short-query branch): {qn_le4}/{n} ({100.0 * qn_le4 / n:.1f}%)")
    print(f"mean kept tokens: {sum(len(cpp_engine.tokenize(t, True)) for _, t in topics) / n:.2f}")
    if stop_hits:
        print("tokens removed by stop list (stemmed surface, excluding length-1 already gone):")
        for tok, c in sorted(stop_hits.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {tok:16} {c}")
    else:
        print("No title tokens were removed solely by INDEX_STOPWORDS.")
    if dropped_examples:
        print("examples of dropped tokens:")
        for qid, title, dropped in dropped_examples:
            print(f"  {qid}: {title!r}  dropped={dropped}")


if __name__ == "__main__":
    main()
