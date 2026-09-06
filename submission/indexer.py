"""
submission/indexer.py — inverted index (pure-Python reference path).

`doc_norm` is the L2 norm of the FULL document TF-IDF vector, computed at
build time. Cosine similarity cannot be recovered from query terms alone
(see boolean_vsm.vsm_score).

`doc_freq` is stored explicitly so `document_frequency()` is O(1) and IDF
stays correct if postings are later pruned.

save()/load() round-trip only the fields scoring needs. Gzip mtime is
forced to 0 and JSON keys are sorted so two builds of the same corpus are
byte-identical.
"""
import gzip
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# lnc.ltc-style log-tf. False -> w = tf * ln(N/df), matching the assignment
# text. Must stay in lockstep with boolean_vsm (imported from here).
USE_LOG_TF = True


def tf_weight(tf: float) -> float:
    return (1.0 + math.log(tf)) if USE_LOG_TF else float(tf)


def vsm_idf(df: int, N: int) -> float:
    return math.log(N / df) if df > 0 and N > 0 else 0.0


def s_stem(word: str) -> str:
    """Light S-stemmer (plural stripping only)."""
    if len(word) <= 3:
        return word
    if not word.endswith("s"):
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        third_last = word[-3]
        if third_last in ("s", "x", "z") or word.endswith("ches") or word.endswith("shes"):
            return word[:-2]
        return word[:-1]
    if word[-2] not in ("s", "u"):
        return word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization with S-stemming."""
    return [s_stem(t) for t in _TOKEN_RE.findall(text.lower())]


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: tf}
        self.doc_len: Dict[str, int] = {}  # doc_id -> token count
        self.doc_norm: Dict[str, float] = {}  # doc_id -> ||d|| (same weights as VSM)
        self.doc_freq: Dict[str, int] = {}  # term -> df
        self.N: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, corpus: Iterable[Tuple[str, str]]) -> None:
        self.postings = {}
        self.doc_len = {}
        self.doc_norm = {}
        self.doc_freq = {}
        self.N = 0
        total_tokens = 0

        for doc_id, text in corpus:
            self.N += 1
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            total_tokens += len(tokens)

            tf_map: Dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term, tf in tf_map.items():
                self.postings.setdefault(term, {})[doc_id] = tf

        self.avg_doc_len = (total_tokens / self.N) if self.N > 0 else 0.0
        self.doc_freq = {term: len(plist) for term, plist in self.postings.items()}
        self._compute_doc_norms()

    def _compute_doc_norms(self) -> None:
        """||d|| over the full document vector, same weights as vsm_score."""
        acc: Dict[str, float] = {d: 0.0 for d in self.doc_len}
        if self.N == 0:
            self.doc_norm = acc
            return
        for term, plist in self.postings.items():
            df = self.doc_freq.get(term, len(plist))
            idf = vsm_idf(df, self.N)
            if idf == 0.0:
                continue
            for doc_id, tf in plist.items():
                w = tf_weight(tf) * idf
                acc[doc_id] += w * w
        self.doc_norm = {d: math.sqrt(v) for d, v in acc.items()}

    # Keep the previous name so load()/vsm_score fallbacks still work.
    _recompute_doc_norms = _compute_doc_norms

    def document_frequency(self, term: str) -> int:
        if term in self.doc_freq:
            return self.doc_freq[term]
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        data = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "doc_norm": self.doc_norm,
            "doc_freq": self.doc_freq,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }
        index_file = os.path.join(index_dir, "index.json.gz")
        # mtime=0 + sorted keys: two builds of the same corpus are byte-identical.
        # gzip.open() would stamp the current time into the header.
        payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with open(index_file, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(payload)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        index = cls()
        with gzip.open(os.path.join(index_dir, "index.json.gz"), "rt", encoding="utf-8") as f:
            data = json.load(f)
        index.postings = data["postings"]
        index.doc_len = data["doc_len"]
        index.N = data["N"]
        index.avg_doc_len = data["avg_doc_len"]
        index.doc_freq = {t: int(df) for t, df in data.get("doc_freq", {}).items()}
        if not index.doc_freq:
            index.doc_freq = {term: len(plist) for term, plist in index.postings.items()}
        index.doc_norm = {d: float(v) for d, v in (data.get("doc_norm") or {}).items()}
        if not index.doc_norm:
            index._compute_doc_norms()
        return index
