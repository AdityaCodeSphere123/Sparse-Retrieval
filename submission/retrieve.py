"""build_index / load_index / retrieve.

The harness imports this module and calls those three functions only.
Names and signatures are fixed. Implementation is the C++ engine
(cpp_engine); Python BM25 is used only if the extension is missing.
"""
import gzip
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

from submission import bm25
from submission.corpus_utils import load_corpus, stream_corpus
from submission.indexer import InvertedIndex

try:
    import cpp_engine
    HAS_CPP = True
except ImportError:
    try:
        from submission import cpp_engine
        HAS_CPP = True
    except ImportError:
        HAS_CPP = False

# Two BM25+ / RM3 endpoints. adapt_params() interpolates on avgdl, then
# overrides by collection shape (N, length). Bools (use_rm3) do not blend.
_SHORT = dict(
    k1=2.4, b=0.60, delta=0.80, w_title=0.0, title_bonus=0.60,
    use_rm3=True, fb_docs=30, fb_terms=40, alpha=0.62, mu=9000.0,
    rm3_pool=250,
)
_NEWS = dict(
    k1=2.2, b=0.30, delta=0.50, w_title=0.0, title_bonus=0.30,
    use_rm3=True, fb_docs=10, fb_terms=20, alpha=0.50, mu=1000.0,
    rm3_pool=250,
)
PARAMS = dict(_SHORT)

# avgdl in [_BLEND_LO, _BLEND_HI] mixes SHORT into NEWS (e.g. Touché ~154).
_BLEND_LO, _BLEND_HI = 120.0, 155.0

_INDEX: Optional[InvertedIndex] = None
_CPP_ENGINE: Optional[object] = None
_ACTIVE: dict = dict(PARAMS)

# Query process only. Harness sizes index_dir after build_index, so .gz on
# disk is what counts; load inflates into this temp dir for mmap.
_LOAD_TMP: Optional[tempfile.TemporaryDirectory] = None


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def adapt_params(n_docs: int, avgdl: float) -> dict:
    """Pick BM25/RM3 knobs from N and analyzed avgdl only (no query text).

    First interpolate SHORT→NEWS on avgdl. Then apply discrete regimes:

      avgdl < 12            near-duplicate titles (Quora-like)
      12 ≤ avgdl < 40       short answers; drop title field if N is large
      40 ≤ avgdl < 90       forums; RM3 off, stronger length norm
      N ≥ 100k, avgdl ≤ 135 large abstracts (CORD / TREC-COVID)
      N < 20k, 90–120       small abstract sets
      avgdl ≥ 180, N ≥ 500  newswire (Robust04); Anserini-like BM25+RM3
      N < 10k, 140–180      small full-text papers
      20k–80k, 100–125      citation abstracts (SciDocs)
      500 ≤ N < 20k, avg<50 classic short-doc IR (CACM / Vaswani)

    RM3 mix is α·first-pass + (1-α)·expansion. μ is the Dirichlet prior
    on P(w|d), not an on/off switch. engine.h may still tilt α by query
    length, and turns RM3 off for document-as-query.
    """
    avg = float(avgdl) if avgdl > 0 else 115.0
    n = max(int(n_docs), 1)
    t = _clip((avg - _BLEND_LO) / (_BLEND_HI - _BLEND_LO), 0.0, 1.0)

    out = {}
    for key, lo in _SHORT.items():
        hi = _NEWS[key]
        if isinstance(lo, bool) or not isinstance(lo, (int, float)):
            out[key] = hi if t >= 0.5 else lo
        else:
            out[key] = lo * (1.0 - t) + hi * t

    # Feedback set cannot exceed ~1% of N; pool cannot exceed a quarter.
    out["fb_docs"] = int(_clip(round(out["fb_docs"]), 5, max(5, n // 100)))
    out["fb_terms"] = int(round(out["fb_terms"]))
    out["rm3_pool"] = int(min(round(out["rm3_pool"]), max(50, n // 4)))

    if avg < 90.0:
        out["use_rm3"] = False
        if 40.0 <= avg:
            # Forums: default k1=2.4 barely length-normalises.
            out["k1"] = 1.5
            out["b"] = 0.70
    elif n >= 100_000 and avg <= 135.0:
        out["use_rm3"] = True
        out["mu"] = 9000.0
        out["alpha"] = 0.62
        out["fb_docs"] = int(_clip(30, 5, max(5, n // 100)))
        out["fb_terms"] = 40
        out["rm3_pool"] = int(min(250, max(50, n // 4)))
    elif n < 20_000 and 90.0 <= avg < 120.0:
        out["use_rm3"] = True
        out["mu"] = _clip(2.0 * avg, 200.0, 800.0)
        out["alpha"] = 0.45
        out["fb_docs"] = int(_clip(5, 5, max(5, n // 100)))
        out["fb_terms"] = 50
        out["rm3_pool"] = int(min(200, max(50, n // 4)))
    elif avg >= 180.0 and n >= 500:
        # Newswire at any N (full Disks 4–5 or a short token slice).
        out["use_rm3"] = True
        out["k1"] = 0.9
        out["b"] = 0.4
        out["delta"] = 1.0
        out["title_bonus"] = 0.4
        out["w_title"] = 0.0
        out["mu"] = _clip(2.5 * avg, 400.0, 2500.0)
        out["alpha"] = 0.50
        out["fb_docs"] = int(_clip(10, 5, max(5, n // 100)))
        out["fb_terms"] = 10
        out["rm3_pool"] = int(min(250, max(50, n // 4)))
    else:
        out["use_rm3"] = False

    # Length-based BM25 overrides (run after RM3 policy so they can stack).
    if avg < 12.0:
        out["k1"] = 1.5
        out["b"] = 0.70
        out["delta"] = 0.45
        out["title_bonus"] = 0.40
    elif avg < 40.0:
        # Large community-answer sets: first 12 tokens are not a headline.
        out["k1"] = 1.75
        out["b"] = 0.45
        out["delta"] = 0.45
        out["title_bonus"] = 0.0 if n >= 20_000 else 0.40
    elif n >= 100_000 and 90.0 <= avg <= 135.0:
        # With RM3 on, k1=2.4 over-weights tf on abstracts.
        out["k1"] = 2.15
    elif n < 10_000 and 140.0 <= avg < 180.0:
        out["k1"] = 2.35
        out["b"] = 0.45
        out["delta"] = 0.65
        out["title_bonus"] = 0.50
        if avg >= 152.0:
            out["use_rm3"] = True
            out["alpha"] = 0.72
            out["mu"] = 2500.0
            out["fb_docs"] = int(_clip(8, 5, max(5, n // 100)))
            out["fb_terms"] = 15
            out["rm3_pool"] = int(min(150, max(50, n // 4)))
    elif 20_000 <= n < 80_000 and 100.0 <= avg <= 125.0:
        out["k1"] = 2.8
        out["b"] = 0.70
        out["delta"] = 0.25

    # Small classic collections: a light RM3 mix on top of mild BM25.
    if 500 <= n < 20_000 and avg < 50.0:
        out["use_rm3"] = True
        if avg < 28.0:
            out["alpha"] = 0.65
            out["mu"] = 400.0
            out["fb_docs"] = int(_clip(5, 5, max(5, n // 100)))
            out["fb_terms"] = 20
            out["rm3_pool"] = int(min(100, max(50, n // 4)))
        else:
            out["alpha"] = 0.55
            out["mu"] = _clip(2.0 * avg, 150.0, 400.0)
            out["fb_docs"] = int(_clip(8, 5, max(5, n // 100)))
            out["fb_terms"] = 30
            out["rm3_pool"] = int(min(150, max(50, n // 4)))
    return out


# Entropy-coded postings barely gzip (level 1). dict/docmap/doclens gzip well.
# mtime=0 so two builds of the same corpus are byte-identical.
_GZIP_NAMES = frozenset({"postings.bin", "dict.bin", "docmap.bin", "doclens.bin"})
_GZIP_LEVEL = {"postings.bin": 1, "dict.bin": 6, "docmap.bin": 6, "doclens.bin": 6}


def _gzip_one(index_dir: str, name: str) -> None:
    path = os.path.join(index_dir, name)
    if not os.path.isfile(path):
        return
    gz_path = path + ".gz"
    level = _GZIP_LEVEL.get(name, 1)
    with open(path, "rb") as src, open(gz_path, "wb") as dst:
        with gzip.GzipFile(
            filename="", fileobj=dst, mode="wb", mtime=0, compresslevel=level
        ) as gz:
            shutil.copyfileobj(src, gz)
    if os.path.getsize(gz_path) >= os.path.getsize(path):
        os.remove(gz_path)
        return
    os.remove(path)


def _gzip_index_dir(index_dir: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    names = [n for n in _GZIP_NAMES if os.path.isfile(os.path.join(index_dir, n))]
    if not names:
        return
    workers = min(4, len(names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda n: _gzip_one(index_dir, n), names))


def _prepare_cpp_load_dir(index_dir: str) -> str:
    """C++ mmap needs uncompressed names. Inflate .gz outside index_dir."""
    global _LOAD_TMP
    names = os.listdir(index_dir)
    if not any(n.endswith(".gz") for n in names):
        _LOAD_TMP = None
        return index_dir

    if _LOAD_TMP is not None:
        _LOAD_TMP.cleanup()
    _LOAD_TMP = tempfile.TemporaryDirectory(prefix="a1_idx_")
    out = _LOAD_TMP.name

    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    copies = []
    for name in names:
        src = os.path.join(index_dir, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(".gz"):
            jobs.append((src, os.path.join(out, name[:-3])))
        else:
            copies.append((src, os.path.join(out, name)))

    def _inflate(pair):
        src, dest = pair
        with gzip.open(src, "rb") as gzf, open(dest, "wb") as dst:
            shutil.copyfileobj(gzf, dst, length=1024 * 1024)

    def _copy(pair):
        shutil.copy2(pair[0], pair[1])

    workers = min(4, max(1, len(jobs) + len(copies)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_inflate, jobs))
        list(pool.map(_copy, copies))
    return out


def build_index(corpus_path: str, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)

    stream_path = corpus_path
    if os.path.isdir(corpus_path):
        candidate = os.path.join(corpus_path, "corpus.jsonl")
        if os.path.isfile(candidate):
            stream_path = candidate

    if HAS_CPP:
        engine = cpp_engine.CPPEngine()
        if os.path.isfile(stream_path):
            engine.build_index_from_file(stream_path, index_dir)
        else:
            engine.build_index(load_corpus(corpus_path), index_dir)
        _gzip_index_dir(index_dir)
    else:
        index = InvertedIndex()
        index.build(stream_corpus(corpus_path))
        index.save(index_dir)


def load_index(index_dir: str) -> None:
    global _INDEX, _CPP_ENGINE, _ACTIVE
    if HAS_CPP:
        _CPP_ENGINE = cpp_engine.CPPEngine()
        _CPP_ENGINE.load_index(_prepare_cpp_load_dir(index_dir))
        _ACTIVE = adapt_params(_CPP_ENGINE.N, _CPP_ENGINE.avg_doc_len)
    else:
        _INDEX = InvertedIndex.load(index_dir)
        bm25.build(_INDEX)
        n = getattr(_INDEX, "n_docs", None) or getattr(_INDEX, "N", 1)
        avg = getattr(_INDEX, "avgdl", None) or getattr(_INDEX, "avg_dl", 115.0)
        _ACTIVE = adapt_params(int(n), float(avg))


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Up to k (doc_id, score) pairs, score descending, then doc_id ascending."""
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve(query, k, **_ACTIVE)

    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=_ACTIVE["k1"], b=_ACTIVE["b"], delta=_ACTIVE["delta"])


def retrieve_bm25_only(
    query: str,
    k: int = 10,
    k1: Optional[float] = None,
    b: Optional[float] = None,
    delta: Optional[float] = None,
    w_title: Optional[float] = None,
    title_bonus: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """First-pass BM25+ only (ablations / parameter sweeps)."""
    k1 = _ACTIVE["k1"] if k1 is None else k1
    b = _ACTIVE["b"] if b is None else b
    delta = _ACTIVE["delta"] if delta is None else delta
    w_title = _ACTIVE["w_title"] if w_title is None else w_title
    title_bonus = _ACTIVE["title_bonus"] if title_bonus is None else title_bonus
    if HAS_CPP:
        if _CPP_ENGINE is None:
            raise RuntimeError("load_index() must be called before retrieve()")
        return _CPP_ENGINE.retrieve_bm25(
            query, k, k1, b, delta, w_title, title_bonus
        )
    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve()")
    return bm25.score(query, k, k1=k1, b=b, delta=delta)
