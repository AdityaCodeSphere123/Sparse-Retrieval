# COL764 Assignment 1 — Sparse Retrieval

## Competition entry

The submitted ranker is **BM25+ with a title hit bonus and RM3 expansion**, in a from-scratch C++ engine (`cpp_engine.cpp`) called from `retrieve.py`. Boolean retrieval and TF–IDF cosine live in Python for the required correctness tests; they are not blended into the competition `retrieve()`.

On the released full corpus and 50 dev topics (two-process harness): **nDCG@10 = 0.7348**, **MAP@10 = 0.0188**, **P@10 = 0.80**, index **41.2 MB**, mean latency **~2.9 ms**. Parameters were chosen on this dev set only.

Submitted knobs (`retrieve.py`): \(k_1=2.1\), \(b=0.6\), \(\delta=0.25\), RM3 on (`fb_docs=30`, `fb_terms=20`, \(\alpha=0.7\), \(\mu=1000\)), `title_bonus=0.75`.

## Indexing

**Analysis.** Lowercase, split on non-alphanumerics, Porter stemmer. Stopwords are NLTK English, checked *before* stemming, with two carve-outs: single letters stay indexed (vitamin D, T cells) and `don` is kept (Porter of *donning*). Question words (`what`, `how`, `which`, `who`, …) *are* stopped — the Lucene 33-word list left them in, so `what` had real IDF and ranked generic “What we should know” papers for “what is the origin of COVID-19”.

**Layout.** SPIMI runs → front-coded `dict.bin`, gap+varint `postings.bin`. Document length is one log-quantised byte. No position lists (proximity/SDM was not worth the size). The tf byte is packed: **6 bits body tf** (saturates at 63) and **2 bits title tf** in the first 12 token positions (median CORD-19 title length). On this corpus, only **147** (doc, term) pairs had body tf \(> 63\) (max 1369); BM25 is already flat there.

**RM3 auxiliary.** `fwd.bin` stores 24 tf×idf terms per document, after dropping \(df<5\) and \(df>13\%\) of \(N\). The 13% ceiling (was 5%) keeps mid-frequency clinical terms. Cutting 48→24 terms did not move nDCG; `fwdoff.bin` is \((N+1)\times 4\) bytes regardless.

## Ranking

Okapi BM25 with RSJ IDF and BM25+ \(\delta\), plus an additive **title bonus** \(w\cdot\texttt{title_bonus}\) when the term occurs in the leading 12 tokens (same shape as \(\delta\), title-conditioned). Query processing is term-at-a-time into a dense accumulator, then `partial_sort`. Load memory-maps postings and the forward index.

RM3: first-pass BM25, relevance model with Dirichlet \(\mu\) on the top 30 documents, mix with \(\alpha=0.7\), second BM25 pass. Query weights are normalised over in-vocabulary terms only.

## Required models vs competition entry (dev)

Boolean AND/OR and lnc.ltc cosine are implemented in Python (full-document L2 norms, not query-subspace). On the **6-query toy set**, cosine and BM25 are both strong (the set is too small to separate them). On the **50-query full dev set**, the C++ stack is what the harness scores:

| System | nDCG@10 | MAP@10 |
|---|---:|---:|
| Local library BM25 reference | 0.428 | 0.009 |
| C++ BM25+ only (no RM3, no title) | 0.672 | 0.017 |
| C++ BM25+RM3, `title_bonus=0` | 0.680 | 0.018 |
| **Submitted (RM3 + title)** | **0.735** | **0.019** |

MAP@10 is small because topics have hundreds of relevant documents; a perfect top-10 still cannot exceed \(\approx 0.027\). Relative ranking is what matters.

## Parameter search (dev only)

Held-out topics were not used.

- **Stoplist:** Lucene 33-word → NLTK English (question words removed) was the first large gain.
- **Title window 12** and packed tf: +0.055 nDCG with every other knob frozen (`title_bonus` 0.75 vs 0).
- **\(k_1,b\):** a BM25-only grid preferred higher \(k_1\) and moderate \(b\) over textbook \((1.2,0.75)\). Submitted **\(2.1, 0.6\)** was chosen together with RM3 and the title bonus, not by taking the BM25-only winner and turning extras on afterwards. \(\delta=0.25\) beat \(\delta=0\) in that neighbourhood.
- **RM3:** on vs off is a smaller step once the title field is in (0.735 vs 0.672 for BM25+ only). `fb_docs=30`, `fb_terms=20` beat narrower feedback on this set. Df ceiling 5%→13% is a **rebuild** choice.
- **Not used:** near-duplicate suppression (TREC-COVID judges both copies), letter/digit splitting of `SARS-CoV-2`, RRF of BM25 and RM3.

A 1-D \(k_1\) / \(b\) plot for the write-up can be regenerated with `python -m scripts.tune --sweep k1b` on a frozen index (RM3 off) if the Moodle PDF needs a figure.

## Error analysis

Five topics still have nDCG@10 \(< 0.3\):

| ID | Query | nDCG@10 | What goes wrong |
|---|---|---:|---|
| 15 | how long can the coronavirus live outside the body | 0.17 | Survival-on-surfaces language; many hits are generic COVID reviews. |
| 49 | recover from COVID-19 … immune response … prevent re-infection | 0.19 | Long, multi-facet NL question; RM3 can lock onto “recovery” or “antibody” alone. |
| 4 | what causes death from Covid-19? | 0.22 | “Death”/mortality is common; causes (ARDS, comorbidity) are not in the query. |
| 3 | will SARS-CoV2 infected people develop immunity? Is cross protection possible? | 0.23 | Two questions in one; first-pass docs often answer only “immunity”. |
| 31 | How does the coronavirus differ from seasonal flu? | 0.27 | Contrastive query; BM25 matches flu *or* COVID papers, not differences. |

RM3 cannot fix a wrong first pass. Sparse overlap still misses relevant abstracts that never use the query stems. Odd vs even topic means (0.66 vs 0.81) show the 0.735 average is not uniform — private topics can move.

## AI-use disclosure

LLM assistance (Cursor, Claude, Gemini) was used for implementation, debugging, and drafting this report. Index layout, stoplist, title packing, RM3 settings, and \(k_1,b,\delta\) were chosen and checked with the course harness on the released data. Scoring and indexing logic is original code, not a wrapped search library.

## Code provenance

Own: inverted index, codecs, Porter stemmer, BM25+, RM3, title packing, Python Boolean/VSM/BM25. Ideas (not copied engines): SPIMI, Okapi BM25 / BM25+, RM3 / Dirichlet, NLTK stopword inventory. Allowed glue: pybind11. `ir_datasets` and `rank_bm25` are not used inside `retrieve()`.
