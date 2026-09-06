# COL 7364/764 — Assignment 1: Sparse Retrieval

From-scratch inverted index with **BM25+** and **RM3**. Competition path is the
C++ engine (`submission/code/`) called from `submission/retrieve.py`.

**Author: Aditya Anand (2023CS50284)**

The PDF write-up is `report.tex` / `report.pdf`. This README is the
reproducible companion: how to run, layout, what is implemented, measured
numbers, and tests.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Compile the pybind11 extension (required for the competition retriever)
cd submission && python setup.py build_ext --inplace && cd ..

# Interface + metrics tests, then harness on toy / CORD-19 / Cranfield / CACM
bash scripts/smoke_test.sh
```

C++ is built during `docker build` (see `Dockerfile`), never inside
`build_index()` — compile time is not charged as index-build time.

### Harness (same path as the leaderboard)

```bash
python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --baseline-run data/full/reference_bm25_run_dev.trec \
  --k 10
```

`build_index` and `load_index`/`retrieve` run in **two processes**. Metrics
are `harness/metrics.py` (graded nDCG with \(2^{\mathrm{rel}}-1\); MAP/MRR/P@10
binarise \(\mathrm{rel}>0\)).

Fetch CORD-19 into `data/full/` if missing:

```bash
python scripts/download_full_corpus.py
```

Unit tests:

```bash
pytest tests/ -q
```

---

## File structure

```
.
├── assignment1.tex          # course spec
├── report.tex / report.pdf  # 4-page report
├── README.md
├── Dockerfile               # grading image; compiles cpp_engine
├── requirements.txt
├── conftest.py
├── data/
│   ├── toy/                 # 20-doc starter set
│   ├── cranfield/           # classical IR
│   ├── cacm/
│   └── README.md
├── submission/
│   ├── retrieve.py          # build_index / load_index / retrieve / adapt_params
│   ├── setup.py             # pybind11
│   ├── code/                # analysis, codecs, engine, JSONL, bindings
│   ├── indexer.py           # Python inverted index (Boolean/VSM/Python BM25)
│   ├── boolean_vsm.py       # AND/OR + lnc.ltc cosine
│   ├── bm25.py              # tunable Okapi / BM25+
│   └── custom_scorer.py     # min-max BM25⊕cosine (not used by retrieve())
├── harness/                 # scoring (do not edit for the grade)
├── tests/                   # see below
├── scripts/
│   ├── smoke_test.sh
│   ├── download_full_corpus.py
│   ├── error_analysis.py
│   └── package_submission.sh
└── docs/
    ├── SUBMISSION_INTERFACE.md
    └── DOCKER_SUBMISSION.md
```

No Lucene / Elasticsearch / Pyserini / Whoosh / `rank_bm25` inside
`retrieve()`. `ir_datasets` is only for downloading corpora.

---

## What is implemented

### Analysis (C++, index = query)

Lowercase, split on non-alnum; glue dotted initials (`U.S.` → `us`); drop
leftover length-1 tokens; Lucene English stops including question words;
Porter stemmer (own code). `IR_STEMMER=1|0` is S-stem / none. Flags live in
`meta.bin` (`INDEX_VERSION=14`).

Python `indexer.py` is a **different** chain (S-stem, no stops, no acronym
folding). That is the required Boolean/VSM/Python-BM25 path and the unit-test
corpus.

### Index

Parallel SPIMI on mmap’d JSONL, ≤4 workers (fewer above 400k/800k docs).
Persisted:

| File | Contents |
|---|---|
| `dict.bin` | Front-coded terms; packed df / codec / title flag; vbyte offset deltas |
| `postings.bin` | df&lt;8: vbyte; else Simple-9 (tf−1) + Elias–Fano; unit-tf skips Simple-9; sparse title sidecar |
| `docmap.bin` | Packed base-A ids, raw equal-length, or front-coded |
| `doclens.bin` | N, avgdl, token total, 1-byte \(64\log_2(1+\|d\|/\mathrm{avgdl})\) |
| `meta.bin` | Version + analysis flags (not gzipped) |

IDF, collection frequency, and the RM3 CSR are rebuilt at load. Gzip level 6
on postings (4 members), 9 on small files if smaller; `mtime=0`.

### Retrievers

- **Boolean:** AND / OR, unranked, sorted doc ids.
- **VSM:** `lnc.ltc` cosine with **full-document** L2 norms.
- **BM25 / BM25+:** RSJ IDF, tunable `k1`, `b`, Lv & Zhai \(\delta\). C++ adds a
  title bonus on the first 12 token positions and query-tf normalisation.
- **Custom (optional):** 0.85/0.15 min-max fusion of Python BM25 and cosine.
  On CORD-19 it **loses** to Python BM25, so it is not in `retrieve()`.
- **Competition:** C++ BM25+ + RM3. Dirichlet \(P(w\mid R)\), IDF-weighted
  expansion. Newswire: full second BM25 pass, uniform \(P(d\mid R)\). Abstracts:
  rerank first-pass pool (\(\lambda=0.25\) uniform + \(0.75\,\mathrm{log1p}\)
  score). `adapt_params(N, avgdl)` picks knobs from collection **shape** only;
  query-length overlay can turn RM3 off for document-as-query.

CORD-19 shipped point: \(k_1=2.15\), \(b=0.60\), \(\delta=0.80\),
\(\beta=0.60\), \(\alpha=0.62\), \(\mu=9000\), 30/40 feedback.

Ties everywhere: score descending, then doc id ascending.

---

## CORD-19 / TREC-COVID (50 released topics)

Two-process harness (re-measured): **nDCG@10 = 0.7458**, MAP@10 = 0.0198,
MRR = 0.9589, P@10 = 0.8260, build **2.20 s**, mean latency **2.28 ms**,
index **16.6 MB**. Local `rank_bm25` reference nDCG@10 = 0.4279 (beaten).

| Run | nDCG@10 | MAP@10 | MRR | P@10 |
|---|---:|---:|---:|---:|
| Boolean OR (unranked, doc-id order) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| VSM lnc.ltc (Python S-stem) | 0.2815 | 0.0067 | 0.5178 | 0.3160 |
| Python BM25 \(k_1=1.2\), \(b=0.75\) | 0.5714 | 0.0139 | 0.8393 | 0.6360 |
| Python BM25 ⊕ VSM fusion | 0.5514 | 0.0129 | 0.7992 | 0.6300 |
| C++ textbook BM25 \(k_1=1.2\), \(b=0.75\), \(\delta=0\) | 0.6356 | 0.0155 | 0.8875 | 0.6920 |
| C++ BM25+ \(k_1=2.15\), \(b=0.60\), \(\delta=0.80\), no title | 0.6323 | 0.0151 | 0.8990 | 0.6900 |
| + title \(\beta=0.60\) | 0.6815 | 0.0171 | 0.9283 | 0.7380 |
| + RM3 (competition `retrieve()`) | **0.7458** | 0.0198 | 0.9589 | 0.8260 |

Boolean is 0 because unranked Cord ids are lexicographic; the first 10 are not
the judged relevant set.

### \(k_1\) and \(b\) (re-run)

`retrieve_bm25_only`, \(\delta=0.80\), **no title**, RM3 off. Grid is
monotonic in \(k_1\); shipped \(k_1=2.15\) is **not** on that polyline
(nDCG@10 = 0.6323 at \(b=0.60\)).

**nDCG@10 vs \(k_1\)** (\(b=0.60\)):

| 0.6 | 0.9 | 1.2 | 1.5 | 1.8 | 2.1 | 2.4 | 2.8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5537 | 0.5801 | 0.5940 | 0.6107 | 0.6209 | 0.6337 | 0.6440 | 0.6548 |

**nDCG@10 vs \(b\)** (\(k_1=2.15\)):

| 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6155 | 0.6241 | 0.6319 | 0.6339 | 0.6323 | 0.6364 | 0.6338 | 0.6161 |

BM25-only peak is \(k_1=2.8\) / \(b\approx 0.70\). Shipped \((2.15, 0.60)\) is
the flat region that stacks with title + RM3, not that argmax.

---

## All collections vs textbook BM25

Same C++ analysis. **BM25** = `retrieve_bm25_only` with \(k_1=1.2\),
\(b=0.75\), \(\delta=0\), no title. **Ours** = `retrieve()` + `adapt_params`.
nDCG@10. BEIR rows were measured on test splits, then those extra dumps were
removed from `data/` (they are not part of the Moodle zip).

| Collection | N | avgdl | #q | BM25 | Ours | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Toy | 20 | 7.8 | 6 | 0.8793 | 0.8793 | 0.000 |
| Cranfield | 1,400 | 100.8 | 225 | 0.2922 | 0.3308 | +0.039 |
| CACM | 3,204 | 33.7 | 52 | 0.5020 | 0.5256 | +0.024 |
| Vaswani | 11,429 | 24.8 | 93 | 0.4356 | 0.4571 | +0.022 |
| SciFact | 5,183 | 145.5 | 300 | 0.6800 | 0.6832 | +0.003 |
| NFCorpus | 3,633 | 156.7 | 323 | 0.3240 | 0.3307 | +0.007 |
| FiQA | 57,638 | 71.6 | 648 | 0.2476 | 0.2624 | +0.015 |
| SciDocs | 25,657 | 115.2 | 1,000 | 0.1568 | 0.1654 | +0.009 |
| ArguAna | 8,674 | 96.5 | 1,406 | 0.3690 | 0.3538 | −0.015 |
| Touché-2020 | 382,545 | 155.3 | 49 | 0.4014 | 0.4899 | +0.089 |
| Quora | 522,931 | 5.8 | 10,000 | 0.7858 | 0.7853 | −0.001 |
| Antique | 403,666 | 20.4 | 200 | 0.3486 | 0.4033 | +0.055 |
| **CORD-19 / TREC-COVID (dev)** | 171,332 | 109.7 | 50 | 0.6356 | **0.7458** | **+0.110** |

ArguAna is below textbook BM25 because long document-as-query turns RM3/title
off. Quora is essentially tied (near-duplicate titles, RM3 off).

Weak CORD-19 queries for `retrieve()`: 4 (0.066), 12 (0.191), 31 (0.266),
15 (0.366), 49 (0.481). Vitamin D (42) is 0.948.

---

## Tests (`tests/`)

```bash
pytest tests/ -v
```

| File | What it checks |
|---|---|
| `test_interface_conformance.py` | CI contract: signatures, disk write, **fresh-process** `load_index`, no duplicate ids, well-formed `(doc_id, score)`, toy latency, directory corpus |
| `test_metrics.py` | Hand-derived nDCG / DCG / AP / MRR / P@k; duplicate-id inflation; MAP@10 truncated to top-10 |
| `test_retrievers.py` | Boolean AND/OR; VSM full-doc cosine vs query subspace; Okapi formula; BM25+ \(\delta\); tunable `k1`/`b`; ties; Python index save determinism; fusion; S-stem |
| `test_codecs.py` | Simple-9 / Elias–Fano vs bulk decoders; `U.S.` / `F.B.I.` folding |
| `test_reproducibility.py` | Byte-identical rebuilds; reload-stable ranking; doc-id (not insertion-order) ties; `retrieve_bm25_only` skips query-length cuts; parallel build = serial; ragged docmap |
| `test_adapt_params.py` | Shape regimes: CORD-19, newswire/Anserini, forums, Quora-length, Touché vs news RM3, feedback caps |

`scripts/smoke_test.sh` runs conformance + metrics tests, then the full
harness on toy, `data/full`, Cranfield, and CACM.

---

This is done by **Aditya Anand**.
