# COL764 Assignment 1 — Sparse Retrieval Report

## 1. Competition entry (one paragraph)

The submitted `retrieve()` is **BM25 with RM3 query expansion**, implemented in a from-scratch C++ engine (`submission/cpp_engine.cpp`) and called from `submission/retrieve.py`. There is no Boolean/VSM blend on the competition path: those models live in Python for the required correctness tests. Query-time knobs are a single dictionary in `retrieve.py` (`k1=2.5`, `b=0.5`, `delta=0`, RM3 on with 5 feedback documents, 20 expansion terms, `α=0.7`, Dirichlet `μ=1000`). On the released full corpus and 50 dev topics this configuration scored **nDCG@10 = 0.6661** and **MAP@10 = 0.0171**. Turning RM3 off dropped nDCG@10 to **0.6318**, so expansion is part of the submitted ranker, not an unused option.

## 2. Architecture

The grading harness only ever calls three functions in `retrieve.py`: `build_index`, `load_index`, and `retrieve`. Build and load run in **separate processes**, so everything retrieve needs must be written to `index_dir`.

`retrieve.py` is a thin adapter. If the pybind11 module `cpp_engine` is present (Docker/CI builds it with `cd submission && python setup.py build_ext --inplace`), it constructs a `CPPEngine` and forwards the three calls. If the corpus path is a directory containing `corpus.jsonl`, build streams that file in C++ (`build_index_from_file`) so the full collection is never materialised as a Python list. If the extension is missing, the same three functions fall back to a pure-Python inverted index, BM25, Boolean/VSM, and a min–max fusion scorer — enough to pass unit tests, not the leaderboard path.

Tokenisation is **not** a query-time parameter. Stemmer and stopword flags are stored in `meta.bin` at build time (`IR_STEMMER`, `IR_STOPWORDS`) and reapplied on queries so the analysis chain cannot drift.

## 3. Indexing (C++)

### 3.1 Analysis chain

Documents are lowercased, split on non-alphanumeric characters, and (by default) Porter-stemmed with a small closed stopword list at index time. A larger Indri-style list is used later only to prune RM3 expansion candidates. Porter is the default because a light S-stemmer leaves too many inflectional variants on biomedical/newswire text; stopword removal shrinks both postings and the forward index.

JSONL parsing is a small hand-written extractor for `doc_id` and `text`. Closing quotes are found by scanning with escape awareness, not `rfind('"')`, so a quote inside the body cannot truncate the document.

### 3.2 Construction

The indexer is **SPIMI**: documents are tokenised in batches of 50k, in-memory postings are spilled as sorted runs, then merged. While building, each document also spills a candidate term list used later for the forward index. The inverted index stores **doc-id + tf only** (no position lists). Positions were tried for proximity/SDM and dropped: they inflate `index_dir` without a corresponding nDCG gain on this collection.

### 3.3 What is persisted

| File | Role |
|---|---|
| `dict.bin` | Sorted vocabulary, **front-coded** strings, varint df/cf, postings offset, max-tf / min-length for bounds |
| `postings.bin` | Gap-coded doc ids (varint) + 1-byte tf; skip table only if a term spans more than one 128-posting block |
| `doclens.bin` | Collection stats + **1-byte log-quantised** `|D|/avgdl` per document |
| `docmap.bin` | External string doc ids |
| `fwd.bin` / `fwdoff.bin` | Pruned forward index for RM3 |
| `meta.bin` | Stemmer / stopword flags |

Dictionary lookup at query time is binary search on the sorted term array (the array index is the term id used by `fwd.bin`), not a hash map of millions of strings.

Document lengths are stored as `quantize(log2(1 + |D|/avgdl))` in one byte. BM25’s length penalty is then a 256-entry lookup table per query. RM3’s Dirichlet smoother reconstructs an approximate `|D|` from the same byte; that is accurate enough for expansion weights.

The forward index keeps at most **48 terms per document**, chosen from up to 128 candidates by `tf × idf`, after dropping very rare (`df < 5`) and very common (`df > 5%` of *N*) terms. Without it, RM3 would scan the vocabulary to discover terms in the feedback documents.

## 4. Ranking (C++)

### 4.1 BM25 / BM25+

Scoring is Okapi BM25 with Robertson–Sparck Jones IDF and an optional BM25+ floor `δ`:

\[
\mathrm{IDF}(t)=\ln\Bigl(1+\frac{N-\mathrm{df}+0.5}{\mathrm{df}+0.5}\Bigr),\qquad
s(t,D)=\mathrm{IDF}(t)\Bigl(\frac{\mathrm{tf}\,(k_1+1)}{\mathrm{tf}+k_1(1-b+b\,|D|/\mathrm{avgdl})}+\delta\Bigr).
\]

The submitted run uses **`δ = 0`** (plain BM25). A positive `δ` was tried (Lv & Zhai BM25+) and did not beat `δ = 0` at the chosen \(k_1,b\) on this collection.

Query processing is **term-at-a-time** into a dense score array of length *N*, with a “dirty” list so only touched documents are reset. Postings and the forward index are **mmap**’d; `load_index` does not decode bulk postings. The previous top-*k* heap-inside-the-loop was replaced by accumulate-then-`partial_sort`, which is the main latency win. Ties break by score descending, then internal doc id ascending.

Skip-table fields (`max_tf`, `min_qlen`) are stored **as raw statistics**, not as a precomputed max score, so a \(k_1,b\) sweep does not invalidate the index.

### 4.2 RM3

First pass: BM25 on the original query (query weights normalised over **in-vocabulary** terms so OOV tokens do not shrink the original-query mass). If the top feedback scores have almost no variance, expansion is skipped (the first-pass list is already collapsed). Otherwise a relevance model is estimated from the top `fb_docs` documents:

\[
P(w\mid R)=\sum_{D\in F} P(D\mid Q)\,P(w\mid D),\qquad
P(w\mid D)=\frac{\mathrm{tf}(w,D)+\mu\,P(w\mid C)}{|D|+\mu},
\]

with \(P(D\mid Q)\) proportional to the first-pass BM25 score. The top `fb_terms` words of \(P(w\mid R)\) are mixed with the original query as \(α\,P(w\mid Q)+(1-α)\,P(w\mid R)\) and BM25 is run a second time. Feedback term frequencies come only from `fwd.bin` (O(feedback docs), not O(vocabulary)).

On the released full collection this was worth about **+0.034 nDCG@10** versus BM25 alone at the same \(k_1,b\).

## 5. Python path (correctness, not the leaderboard)

The assignment requires independently testable Boolean, VSM, and BM25 implementations. Those are Python:

- **Boolean:** AND / OR over postings; invalid mode raises `ValueError`.
- **VSM:** lnc.ltc-style \((1+\log\mathrm{tf})\ln(N/\mathrm{df})\), cosine against the **full-document** L2 norm stored at index build (query-subspace norms would make every single-term hit score 1.0).
- **BM25:** same formula as C++, including optional `δ` and the same tie-break.
- **Fallback `retrieve()`:** min–max fusion of BM25 and VSM over a pool of 200 (`custom_scorer.py`). The C++ entry does **not** use this blend.

Python `save`/`load` uses gzip with `mtime=0` and sorted JSON keys so two builds of the same corpus are byte-identical.

## 6. Parameter search (dev only)

All numbers below are on the **released** corpus and topics. Held-out topics were not used.

**Analysis chain.** Porter + stopword removal was kept after comparing none / S-stemmer / Porter and stopwords on/off (`scripts/tune.py --sweep analysis`). Changing the analysis chain requires a rebuild; it is not mixed with \(k_1,b\) in the same sweep.

**\(k_1\) and \(b\) (BM25 only).** A grid over \(k_1\) and \(b\) with RM3 off preferred **higher \(k_1\)** and **moderate \(b\)** relative to the textbook \((1.2, 0.75)\). Length normalisation that is too strong (\(b\) near 1) hurt this collection. A representative BM25-only point was \(k_1=2.2\), \(b=0.6\) (nDCG@10 ≈ 0.640). The submitted pair **\(k_1=2.5\), \(b=0.5\)** was chosen jointly with RM3, not by taking the BM25-only grid winner and turning expansion on afterwards.

**RM3.** `use_rm3` on vs off at the submitted BM25 settings: 0.6661 vs 0.6318 nDCG@10. Feedback width was kept small (`fb_docs=5`, `fb_terms=20`) so expansion stays cheap and does not drown the original query (`α=0.7`). Dirichlet `μ=1000` is a standard Indri-scale smoother; a larger `μ` (2000) was used in an earlier checkpoint and was not clearly better once \(k_1,b\) moved.

**What not to trust.** An automatic multi-parameter walk that **restarts from textbook BM25** (e.g. `k1=1.2`, `b=0.75`) can report a locally good \((k_1,b)\) that is terrible once you leave that neighbourhood (high \(b\) with weak \(k_1\)). The submitted values are from the BM25 grid plus a RM3 on/off check, not from that auto restart.

A plot of nDCG@10 against \(k_1\) (fixed \(b\)) and against \(b\) (fixed \(k_1\)) should be attached from `scripts/tune.py --sweep k1b` when the report is typeset for Moodle.


## 7. Error analysis (queries that still fail)

Failures we actually see on this collection, rather than generic IR folklore:

1. **Very short or highly ambiguous topics.** RM3 expands from the first-pass top 5. If those five are the wrong sense (e.g. a common biomedical homonym), expansion reinforces the mistake. High `α` limits the damage but cannot recover a wrong first pass.
2. **Relevant documents that share few stemmed query terms.** Sparse BM25 never sees them; RM3 only helps if a feedback document already in the top 5 introduces the missing vocabulary.
3. **Near-duplicate / boilerplate abstracts.** High tf on generic methods terms (`patient`, `studi`, `result`) is why the forward index ranks candidates by **tf×idf**, not raw tf. Residual boilerplate still leaks into expansion.
4. **Length extremes.** `b=0.5` is a compromise: very long reviews can still outrank a short, on-topic abstract, while `b→1` over-penalised medium documents in the \(k_1,b\) grid.
5. **MAP@10 ceiling.** Many topics have far more than 10 relevant documents in the qrels, so MAP@10 cannot approach 1.0 even with a perfect top-10. Comparisons should be relative, not against textbook unbounded MAP.

Concrete topic IDs and rank lists for 3–5 worst nDCG queries should be pasted here from the harness run file before the Moodle report deadline.

## 8. AI-use disclosure

LLM assistance (Cursor, Claude, Gemini) was used for implementation, debugging, parameter-sweep plumbing, and drafting this report. Design choices (SPIMI layout, BM25+RM3 as the entry, dropping SDM/positions, \(k_1,b\) and RM3 settings) were selected and verified by running the course harness on the released data. All indexing and scoring logic is original code in this repository, not a wrapped search library.

## 9. Code provenance

- **Own code:** inverted index, codecs, Porter stemmer in C++, BM25, RM3, Python Boolean/VSM/BM25, harness adapter in `retrieve.py`.
- **Allowed tooling:** pybind11 (extension glue), `ir_datasets` (optional corpus fetch), `rank_bm25` only to generate the local reference run, not inside `retrieve()`.
