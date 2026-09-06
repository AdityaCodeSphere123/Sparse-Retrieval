# Data

## `toy/` — local development set (ships with this repo)

A hand-built, 20-document corpus with 6 queries and graded (0-3) relevance
judgments, covering a handful of clearly separated topics (cats, dogs,
programming languages, coffee, and IR itself). It exists so you can:

- develop and debug your indexer/retrievers in seconds without downloading
  anything,
- run the harness end-to-end and see sane, hand-checkable metrics,
- run the unit tests in `tests/`, which are written against this exact set.

**The toy set is not the assignment corpus and its scores do not count
toward your grade.** It is small enough that a correct implementation
should get every judged query essentially right — use it to catch bugs,
not to tune parameters.

Files:
- `toy/corpus.jsonl` — one JSON object per line: `{"doc_id": ..., "text": ...}`
- `toy/queries_dev.tsv` — `qid<TAB>query text`
- `toy/qrels_dev.txt` — TREC qrels format: `qid 0 doc_id relevance`

## The real assignment corpus

Run `python scripts/download_full_corpus.py` to fetch the actual
assignment collection via [`ir_datasets`](https://ir-datasets.com/) (a
mid-sized public TREC-style collection; see the assignment spec, Section 6,
for the exact dataset name announced for this offering). The script writes
`corpus.jsonl` in the same format as the toy set, plus the released dev
topics and qrels, so your code does not need to change between the toy
set and the real one.

**Held-out topics are never distributed.** The private leaderboard and
your final grade are computed by course staff running the same harness
against a topic/qrels file you never see. Do not build any query-specific
logic that depends on having seen every possible query in advance.
