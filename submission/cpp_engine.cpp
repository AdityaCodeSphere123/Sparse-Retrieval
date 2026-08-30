// ===========================================================================
// cpp_engine.cpp — from-scratch inverted index + BM25F+/RM3 retrieval.
//
// On-disk layout (v8). Everything is little-endian, written by build_*() and
// read back by load_index(); nothing survives between the two processes
// except these files.
//
//   docmap.bin    N, uint16 id_len (0 = variable). Fixed-width ids are packed
//                 with the length written once; variable ids are front-coded.
//   doclens.bin   N, avg_doc_len, total_tokens, then N x uint8 quantised
//                 length ratios (log-quantised, see quantize_doc_len)
//   dict.bin      terms in sorted order, front-coded: prefix, suffix,
//                 (df << 1) | has_title, delta postings offset. Codec is
//                 implied by df (vbyte iff df < POST_VBYTE_DF).
//   postings.bin  short lists: vbyte gaps with packed tf. Longer lists:
//                 Simple-9 (tf-1), optional sparse title hits (count +
//                 vbyte gaps with 2-bit title_tf), then Elias-Fano doc ids.
//                 Eight zero bytes are appended after the last list so
//                 bit_get cannot read past the mmap on the final EF decode.
//   fwd.bin       per doc: count | ef_len | Simple-9 tfs | Elias-Fano rids.
//                 Same 8-byte trailing pad as postings.bin.
//   fwdoff.bin    N, stride, then ceil(N/stride) x uint32 offsets (stride=8)
//   rm3map.bin    M, then M x (tid gap, cf-df) varints — cf only for RM3 terms
//
// ===========================================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <queue>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <cctype>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Fixed structural constants. Unlike BuildCfg below, these are not swept:
// they trade memory against I/O during the build, not ranking quality.
// ---------------------------------------------------------------------------
static constexpr uint32_t FWD_CANDIDATES   = 64;     // pre-df-filter margin
static constexpr uint32_t SPIMI_BATCH_DOCS = 50000;  // docs buffered per spill
static constexpr uint32_t FWD_OFF_STRIDE   = 8;      // checkpoint every 8 docs in fwdoff.bin
static constexpr int      RM3_POOL         = 250;    // first-pass docs re-scored with RM3
static constexpr uint8_t  INDEX_VERSION    = 8;

// ---------------------------------------------------------------------------
// Build-time settings that are swept rather than derived. Each is a plain
// default with an environment override, so scripts/sweep_build.py can A/B them
// by rebuilding instead of by editing and recompiling. None of them is needed
// at query time, so none of them goes into meta.bin.
//
//   IR_TITLE_WINDOW  leading raw tokens treated as the title field
//   IR_FWD_TERMS     forward-index width (terms/doc) available to RM3
//   IR_RM3_DF_FRAC   df ceiling, as a fraction of N, for RM3 candidate terms
//   IR_MIN_DF        terms rarer than this are dropped from dict+postings
// ---------------------------------------------------------------------------
struct BuildCfg {
    // title_window = 12 raw tokens: the median title length in this collection,
    // and the flat top of the sweep (8-16 are all within noise of each other,
    // 24+ falls away as abstract text starts leaking into the "title").
    uint32_t title_window   = 12;
    // 20: the 16–24 band is essentially flat on nDCG (~0.005 drop at 16).
    // fwd.bin is ~50 bytes/doc at 24; 20 trims that without leaving the
    // plateau. Override with IR_FWD_TERMS.
    uint32_t fwd_terms      = 20;
    // rm3_df_frac = 0.13: a term may be an RM3 expansion candidate if it occurs
    // in at most 13% of documents. The old 5% ceiling was too aggressive for a
    // single-topic collection — it threw away exactly the mid-frequency
    // clinical vocabulary ("mortality", "sequelae", "seroprevalence") that a
    // relevance model is supposed to find. Anything above ~20% starts pulling
    // in collection-wide noise, so the useful band is broad and this sits in it.
    float    rm3_df_frac    = 0.13f;
    // Default 1: keep hapaxes. Sweep 2–4 on dev; they shrink dict+postings
    // but any topic that uses a df=1 stem will lose a first-pass term.
    uint32_t min_df         = 1;
};

static BuildCfg read_build_cfg() {
    BuildCfg c;
    if (const char* e = std::getenv("IR_TITLE_WINDOW")) c.title_window = std::atoi(e);
    if (const char* e = std::getenv("IR_FWD_TERMS"))    c.fwd_terms    = std::atoi(e);
    if (const char* e = std::getenv("IR_RM3_DF_FRAC"))  c.rm3_df_frac  = std::atof(e);
    if (const char* e = std::getenv("IR_MIN_DF"))       c.min_df       = std::atoi(e);
    return c;
}

// ---------------------------------------------------------------------------
// Stopwords (Indri / INQUERY inspired subset), used to prune the forward
// index and the RM3 expansion candidate set.
// ---------------------------------------------------------------------------
static const std::unordered_set<std::string> STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "among", "an", "and", "another", "any", "anybody", "anyone",
    "anything", "anywhere", "are", "area", "areas", "around", "as", "ask", "asked", "asking", "asks", "at",
    "away", "b", "back", "backed", "backing", "backs", "be", "became", "because", "become", "becomes",
    "been", "before", "began", "behind", "being", "beings", "best", "better", "between", "big", "both",
    "but", "by", "c", "came", "can", "cannot", "case", "cases", "certain", "certainly", "clear", "clearly",
    "come", "could", "d", "did", "differ", "different", "differently", "do", "does", "done", "down",
    "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending",
    "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything",
    "everywhere", "f", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four",
    "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general",
    "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great",
    "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "has", "have", "having",
    "he", "her", "here", "herself", "high", "higher", "highest", "him", "himself", "his", "how", "however",
    "i", "if", "important", "in", "interest", "interested", "interesting", "interests", "into", "is",
    "it", "its", "itself", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows",
    "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "like", "likely",
    "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member",
    "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "my", "myself",
    "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next",
    "no", "nobody", "non", "noone", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of",
    "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening",
    "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "our", "out", "over",
    "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed",
    "pointing", "points", "possible", "present", "presented", "presenting", "presents", "problem",
    "problems", "put", "puts", "q", "quite", "r", "rather", "really", "right", "room", "rooms", "s",
    "said", "same", "saw", "say", "says", "second", "seconds", "see", "seem", "seemed", "seeming",
    "seems", "sees", "several", "shall", "she", "should", "show", "showed", "showing", "shows", "side",
    "sides", "since", "small", "smaller", "smallest", "so", "some", "somebody", "someone", "something",
    "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "the",
    "their", "them", "then", "there", "therefore", "these", "they", "thing", "things", "think", "thinks",
    "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today",
    "together", "too", "took", "toward", "turn", "turned", "turning", "turns", "two", "u", "under",
    "until", "up", "upon", "us", "use", "used", "uses", "v", "very", "w", "want", "wanted", "wanting",
    "wants", "was", "way", "ways", "we", "well", "wells", "went", "were", "what", "when", "where",
    "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within", "without",
    "work", "worked", "working", "works", "would", "x", "y", "year", "years", "yet", "you", "your", "yours"
};

// ---------------------------------------------------------------------------
// Document length quantisation: store |D|/avgdl in one byte on a log scale,
// so the length-normalisation term costs 1 byte per document instead of 4.
// ---------------------------------------------------------------------------
static inline uint8_t quantize_doc_len(float len_ratio) {
    if (len_ratio <= 0.0f) return 0;
    float val = 64.0f * std::log2(1.0f + len_ratio);
    if (val >= 255.0f) return 255;
    return static_cast<uint8_t>(val);
}

static inline float dequantize_doc_len(uint8_t q) {
    return std::pow(2.0f, static_cast<float>(q) / 64.0f) - 1.0f;
}

// ---------------------------------------------------------------------------
// Term-frequency byte packing — "why the title field is free".
//
// A posting already spends exactly one byte on tf. Six bits are plenty for it:
// BM25 saturates tf through tf/(tf + k1*norm), so at k1 <= 4 the difference in
// score between tf = 63 and tf = 200 is under 1%, and in a collection of
// abstracts (median 168 words) a tf above 63 essentially does not occur. That
// frees the top two bits to hold the title-field tf, saturated at 3.
//
// So the title field costs ZERO extra index bytes. That matters here: index
// size is its own graded component, and a positional index — the usual way to
// get "where in the document did this term occur" — would have roughly
// doubled the index for a smaller gain (see the report's rejected-ideas note).
// ---------------------------------------------------------------------------
static constexpr uint32_t TF_MAX       = 63;   // 6 bits
static constexpr uint32_t TITLE_TF_MAX = 3;    // 2 bits
// Crossover: vbyte for df < this, Elias-Fano otherwise. At N≈171k, df=8
// already has EF at ~17 B for the ids vs ~24 B of 3-byte gaps; df=4 is a
// wash once Simple-9 tfs are counted. 8 is the size minimum of {4,8,16,32}.
static constexpr uint32_t POST_VBYTE_DF = 8;

static inline uint8_t pack_tf(uint32_t body_tf, uint32_t title_tf) {
    uint32_t b = std::min(body_tf, TF_MAX);
    uint32_t t = std::min(title_tf, TITLE_TF_MAX);
    return static_cast<uint8_t>(b | (t << 6));
}

// Simple-9 for tf/title streams: locally adaptive, so one tf=63 does not
// force 6-bit packing of the whole list.
static const uint8_t S9_N[9] = {28, 14, 9, 7, 5, 4, 3, 2, 1};
static const uint8_t S9_B[9] = { 1,  2, 3, 4, 5, 7, 9,14,28};

static void simple9_pack(std::vector<uint8_t>& out, const uint32_t* vals, uint32_t n) {
    uint32_t i = 0;
    while (i < n) {
        int sel = 8;
        for (int s = 0; s < 9; ++s) {
            const uint32_t k = S9_N[s];
            const uint32_t b = S9_B[s];
            const uint32_t maxv = (b >= 32) ? 0xFFFFFFFFu : ((1u << b) - 1u);
            const uint32_t take = std::min(k, n - i);
            bool ok = true;
            for (uint32_t j = 0; j < take; ++j) {
                if (vals[i + j] > maxv) { ok = false; break; }
            }
            if (ok) { sel = s; break; }
        }
        const uint32_t k = S9_N[sel];
        const uint32_t b = S9_B[sel];
        const uint32_t mask = (b >= 32) ? 0xFFFFFFFFu : ((1u << b) - 1u);
        uint32_t word = static_cast<uint32_t>(sel) << 28;
        const uint32_t take = std::min(k, n - i);
        for (uint32_t j = 0; j < take; ++j)
            word |= (vals[i + j] & mask) << (j * b);
        out.push_back(static_cast<uint8_t>(word));
        out.push_back(static_cast<uint8_t>(word >> 8));
        out.push_back(static_cast<uint8_t>(word >> 16));
        out.push_back(static_cast<uint8_t>(word >> 24));
        i += take;
    }
}

static void simple9_unpack(const uint8_t*& ptr, uint32_t* vals, uint32_t n) {
    uint32_t i = 0;
    while (i < n) {
        uint32_t word = static_cast<uint32_t>(ptr[0])
                      | (static_cast<uint32_t>(ptr[1]) << 8)
                      | (static_cast<uint32_t>(ptr[2]) << 16)
                      | (static_cast<uint32_t>(ptr[3]) << 24);
        ptr += 4;
        const uint32_t sel = word >> 28;
        const uint32_t k = S9_N[sel];
        const uint32_t b = S9_B[sel];
        const uint32_t mask = (b >= 32) ? 0xFFFFFFFFu : ((1u << b) - 1u);
        for (uint32_t j = 0; j < k && i < n; ++j, ++i)
            vals[i] = (word >> (j * b)) & mask;
    }
}

static void bit_put(std::vector<uint8_t>& out, uint64_t& acc, int& filled,
                    uint64_t val, int bits) {
    if (bits <= 0) return;
    acc |= val << filled;
    filled += bits;
    while (filled >= 8) {
        out.push_back(static_cast<uint8_t>(acc));
        acc >>= 8;
        filled -= 8;
    }
}

static void bit_flush(std::vector<uint8_t>& out, uint64_t& acc, int& filled) {
    if (filled > 0) {
        out.push_back(static_cast<uint8_t>(acc));
        acc = 0;
        filled = 0;
    }
}

static uint32_t bit_get(const uint8_t*& p, uint64_t& acc, int& filled, int bits) {
    if (bits <= 0) return 0;
    while (filled < bits) {
        acc |= static_cast<uint64_t>(*p++) << filled;
        filled += 8;
    }
    const uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    uint32_t v = static_cast<uint32_t>(acc) & mask;
    acc >>= bits;
    filled -= bits;
    return v;
}

// Elias-Fano over strictly increasing doc ids. Interleaved unary high-deltas
// and l low bits; sequential decode only (TAAT never needs random access).
static void ef_pack(std::vector<uint8_t>& out, const uint32_t* docs, uint32_t n) {
    uint32_t U = docs[n - 1] + 1;
    uint8_t l = 0;
    if (U >= n) {
        uint32_t q = U / n;
        while (l < 31 && (1u << (l + 1)) <= q) ++l;
    }
    out.push_back(l);
    uint64_t acc = 0;
    int filled = 0;
    uint32_t prev_h = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t d = docs[i];
        uint32_t h = (l == 0) ? d : (d >> l);
        uint32_t low = (l == 0) ? 0 : (d & ((1u << l) - 1u));
        uint32_t zeros = h - prev_h;
        while (zeros > 24) { bit_put(out, acc, filled, 0, 24); zeros -= 24; }
        if (zeros) bit_put(out, acc, filled, 0, static_cast<int>(zeros));
        bit_put(out, acc, filled, 1, 1);
        bit_put(out, acc, filled, low, l);
        prev_h = h;
    }
    bit_flush(out, acc, filled);
}

static void ef_unpack_docs(const uint8_t*& ptr, uint32_t* docs, uint32_t n) {
    uint8_t l = *ptr++;
    uint64_t acc = 0;
    int filled = 0;
    uint32_t h = 0;
    for (uint32_t i = 0; i < n; ++i) {
        while (bit_get(ptr, acc, filled, 1) == 0) ++h;
        uint32_t low = bit_get(ptr, acc, filled, l);
        docs[i] = (l == 0) ? h : ((h << l) | low);
    }
}

// Streaming decoders for the scoring loop. Cross a unary zero-run with
// ctz instead of a branch per bit; hand back one value at a time so the
// loop never materialises scratch_gaps / scratch_tfs. simple9_unpack is
// the reference S9Reader / s9_skip are checked against (check_codecs).
// ef_unpack_docs is that reference for EFReader and also how RM3 reads
// the forward-index rid stream.
struct EFReader {
    const uint8_t* p = nullptr;
    uint64_t buf = 0;
    int      nbits = 0;
    uint32_t h = 0;
    uint8_t  l = 0;

    inline void init(const uint8_t* q) { p = q; l = *p++; buf = 0; nbits = 0; h = 0; }
    inline void fill() { while (nbits <= 56) { buf |= static_cast<uint64_t>(*p++) << nbits; nbits += 8; } }

    inline uint32_t next() {
        for (;;) {
            if (nbits < 8) fill();
            // Only bits [0, nbits) are valid. ctzll on the raw buf can see a
            // stale 1 above nbits and skip a unary zero-run (wrong doc ids).
            const uint64_t live = (nbits >= 64) ? buf
                                 : (buf & ((nbits > 0) ? ((1ull << nbits) - 1ull) : 0ull));
            if (live == 0) {
                h += static_cast<uint32_t>(nbits);
                buf = 0;
                nbits = 0;
                fill();
                continue;
            }
            const int t = __builtin_ctzll(live);
            h += static_cast<uint32_t>(t);
            buf >>= (t + 1);
            nbits -= (t + 1);
            if (nbits > 0 && nbits < 64) buf &= (1ull << nbits) - 1ull;
            else if (nbits <= 0) { buf = 0; nbits = 0; }
            break;
        }
        if (l == 0) return h;
        if (nbits < l) fill();
        const uint32_t low = static_cast<uint32_t>(buf & ((1ull << l) - 1ull));
        buf >>= l;
        nbits -= l;
        if (nbits > 0 && nbits < 64) buf &= (1ull << nbits) - 1ull;
        else if (nbits <= 0) { buf = 0; nbits = 0; }
        return (h << l) | low;
    }
};

struct S9Reader {
    const uint8_t* p = nullptr;
    uint32_t word = 0, mask = 0, b = 0, shift = 0, left = 0;
    inline void init(const uint8_t* q) { p = q; left = 0; }
    inline uint32_t next() {
        if (!left) {
            word = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
                 | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
            p += 4;
            const uint32_t sel = word >> 28;
            left = S9_N[sel]; b = S9_B[sel];
            mask = (b >= 32) ? 0xFFFFFFFFu : ((1u << b) - 1u);
            shift = 0;
        }
        const uint32_t v = (word >> shift) & mask;
        shift += b; --left;
        return v;
    }
};

static inline const uint8_t* s9_skip(const uint8_t* p, uint32_t n) {
    uint32_t i = 0;
    while (i < n) { i += S9_N[static_cast<uint32_t>(p[3]) >> 4]; p += 4; }
    return p;
}

static void check_one_list(const uint32_t* tfs, const uint32_t* docs, uint32_t n, int trial) {
    std::vector<uint8_t> s9, ef;
    simple9_pack(s9, tfs, n);
    ef_pack(ef, docs, n);
    s9.insert(s9.end(), 8, 0);
    ef.insert(ef.end(), 8, 0);

    std::vector<uint32_t> ref_tf(n), ref_doc(n);
    const uint8_t* p = s9.data();
    simple9_unpack(p, ref_tf.data(), n);
    const uint8_t* q = ef.data();
    ef_unpack_docs(q, ref_doc.data(), n);

    S9Reader sr; sr.init(s9.data());
    EFReader er; er.init(ef.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (sr.next() != ref_tf[i] || ref_tf[i] != tfs[i] ||
            er.next() != ref_doc[i] || ref_doc[i] != docs[i]) {
            throw std::runtime_error("codec mismatch trial=" + std::to_string(trial) +
                                    " i=" + std::to_string(i));
        }
    }
    if (s9_skip(s9.data(), n) != p) {
        throw std::runtime_error("s9_skip cursor drift trial=" + std::to_string(trial));
    }
}

static void check_codecs() {
    std::mt19937 rng(1);
    auto run = [&](const std::vector<uint32_t>& tfs,
                   const std::vector<uint32_t>& docs, int trial) {
        check_one_list(tfs.data(), docs.data(), static_cast<uint32_t>(tfs.size()), trial);
    };
    for (uint32_t n : {1u, 2u, 7u, 8u, 28u, 29u, 64u, 1000u, 4096u}) {
        std::vector<uint32_t> tfs(n), docs(n);
        for (uint32_t i = 0; i < n; ++i) { tfs[i] = rng() % 64; docs[i] = i; }
        run(tfs, docs, -static_cast<int>(n));
    }
    const uint32_t sel_hi[9] = {1, 3, 7, 15, 31, 127, 511, 16383, (1u << 28) - 1u};
    for (int s = 0; s < 9; ++s) {
        uint32_t n = S9_N[s] + (s & 1u);
        std::vector<uint32_t> tfs(n, sel_hi[s]), docs(n);
        for (uint32_t i = 0; i < n; ++i) docs[i] = i * 3u + 1u;
        run(tfs, docs, -100 - s);
    }
    { std::vector<uint32_t> tfs{0, 0}, docs{0u, 1u << 20}; run(tfs, docs, -200); }
    for (int trial = 0; trial < 2000; ++trial) {
        uint32_t n = 1 + rng() % 5000;
        std::vector<uint32_t> tfs(n), docs(n);
        for (auto& t : tfs) t = rng() % 64;
        uint32_t d = 0;
        for (auto& x : docs) { d += 1 + rng() % 200; x = d; }
        run(tfs, docs, trial);
    }
}

// ---------------------------------------------------------------------------
// Variable-byte (varint) codecs. 7 payload bits per byte, high bit = continue.
// ---------------------------------------------------------------------------
static inline void write_vbyte(std::vector<uint8_t>& out, uint32_t val) {
    while (val >= 128) {
        out.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val & 0x7F));
}

static inline void write_vbyte64(std::vector<uint8_t>& out, uint64_t val) {
    while (val >= 128) {
        out.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val & 0x7F));
}

static inline uint32_t read_vbyte(const uint8_t*& ptr) {
    uint32_t val = 0;
    int shift = 0;
    while (true) {
        uint8_t b = *ptr++;
        val |= static_cast<uint32_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return val;
}

static inline uint64_t read_vbyte64(const uint8_t*& ptr) {
    uint64_t val = 0;
    int shift = 0;
    while (true) {
        uint8_t b = *ptr++;
        val |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Tokenisation.
//
// Three things happen per token, in this order, and the order matters:
//   1. lowercase + split on non-alphanumerics
//   2. stopword test, on the RAW lowercased token
//   3. stemming
//
// Step 2 must precede step 3: Porter maps "was" -> "wa", "this" -> "thi",
// "his" -> "hi", so a stopword list checked after stemming silently stops
// matching. Positions are the ORIGINAL token offsets and are assigned before
// any removal, so dropping stopwords does not corrupt proximity distances.
//
// Whichever configuration built the index is recorded in meta.bin and reused
// at query time, because query tokenisation must match index tokenisation
// exactly or nothing retrieves.
// ---------------------------------------------------------------------------


class PorterStemmer {
    std::string b;
    int k = 0;   // index of the last character of the word
    int j = 0;   // general offset used by the step helpers

    bool cons(int i) const {
        switch (b[i]) {
            case 'a': case 'e': case 'i': case 'o': case 'u': return false;
            case 'y': return (i == 0) ? true : !cons(i - 1);
            default: return true;
        }
    }

    // m() measures the number of vowel-consonant sequences in b[0..j]
    int m() const {
        int n = 0, i = 0;
        for (;;) {
            if (i > j) return n;
            if (!cons(i)) break;
            i++;
        }
        i++;
        for (;;) {
            for (;;) {
                if (i > j) return n;
                if (cons(i)) break;
                i++;
            }
            i++;
            n++;
            for (;;) {
                if (i > j) return n;
                if (!cons(i)) break;
                i++;
            }
            i++;
        }
    }

    bool vowelinstem() const {
        for (int i = 0; i <= j; i++) if (!cons(i)) return true;
        return false;
    }

    bool doublec(int i) const {
        if (i < 1) return false;
        if (b[i] != b[i - 1]) return false;
        return cons(i);
    }

    // cvc(i): consonant-vowel-consonant where the final consonant is not w, x or y
    bool cvc(int i) const {
        if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        char ch = b[i];
        if (ch == 'w' || ch == 'x' || ch == 'y') return false;
        return true;
    }

    bool ends(const char* s) {
        int len = static_cast<int>(std::strlen(s));
        if (len > k + 1) return false;
        if (b.compare(k - len + 1, len, s) != 0) return false;
        j = k - len;
        return true;
    }

    void setto(const char* s) {
        int len = static_cast<int>(std::strlen(s));
        b.replace(j + 1, k - j, s);
        k = j + len;
    }

    void r(const char* s) { if (m() > 0) setto(s); }

    void step1ab() {
        if (b[k] == 's') {
            if (ends("sses")) k -= 2;
            else if (ends("ies")) setto("i");
            else if (b[k - 1] != 's') k--;
        }
        if (ends("eed")) { if (m() > 0) k--; }
        else if ((ends("ed") || ends("ing")) && vowelinstem()) {
            k = j;
            if (ends("at")) setto("ate");
            else if (ends("bl")) setto("ble");
            else if (ends("iz")) setto("ize");
            else if (doublec(k)) {
                k--;
                char ch = b[k];
                if (ch == 'l' || ch == 's' || ch == 'z') k++;
            } else if (m() == 1 && cvc(k)) {
                setto("e");
            }
        }
    }

    void step1c() {
        if (ends("y") && vowelinstem()) b[k] = 'i';
    }

    void step2() {
        if (k == 0) return;
        switch (b[k - 1]) {
            case 'a': if (ends("ational")) { r("ate"); break; }
                      if (ends("tional"))  { r("tion"); break; }
                      break;
            case 'c': if (ends("enci"))    { r("ence"); break; }
                      if (ends("anci"))    { r("ance"); break; }
                      break;
            case 'e': if (ends("izer"))    { r("ize"); break; }
                      break;
            case 'l': if (ends("abli"))    { r("able"); break; }
                      if (ends("alli"))    { r("al"); break; }
                      if (ends("entli"))   { r("ent"); break; }
                      if (ends("eli"))     { r("e"); break; }
                      if (ends("ousli"))   { r("ous"); break; }
                      break;
            case 'o': if (ends("ization")) { r("ize"); break; }
                      if (ends("ation"))   { r("ate"); break; }
                      if (ends("ator"))    { r("ate"); break; }
                      break;
            case 's': if (ends("alism"))   { r("al"); break; }
                      if (ends("iveness")) { r("ive"); break; }
                      if (ends("fulness")) { r("ful"); break; }
                      if (ends("ousness")) { r("ous"); break; }
                      break;
            case 't': if (ends("aliti"))   { r("al"); break; }
                      if (ends("iviti"))   { r("ive"); break; }
                      if (ends("biliti"))  { r("ble"); break; }
                      break;
        }
    }

    void step3() {
        switch (b[k]) {
            case 'e': if (ends("icate")) { r("ic"); break; }
                      if (ends("ative")) { r(""); break; }
                      if (ends("alize")) { r("al"); break; }
                      break;
            case 'i': if (ends("iciti"))  { r("ic"); break; }
                      break;
            case 'l': if (ends("ical"))   { r("ic"); break; }
                      if (ends("ful"))    { r(""); break; }
                      break;
            case 's': if (ends("ness"))   { r(""); break; }
                      break;
        }
    }

    void step4() {
        if (k == 0) return;
        switch (b[k - 1]) {
            case 'a': if (ends("al")) break; return;
            case 'c': if (ends("ance")) break; if (ends("ence")) break; return;
            case 'e': if (ends("er")) break; return;
            case 'i': if (ends("ic")) break; return;
            case 'l': if (ends("able")) break; if (ends("ible")) break; return;
            case 'n': if (ends("ant")) break; if (ends("ement")) break;
                      if (ends("ment")) break; if (ends("ent")) break; return;
            case 'o': if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
                      if (ends("ou")) break; return;
            case 's': if (ends("ism")) break; return;
            case 't': if (ends("ate")) break; if (ends("iti")) break; return;
            case 'u': if (ends("ous")) break; return;
            case 'v': if (ends("ive")) break; return;
            case 'z': if (ends("ize")) break; return;
            default: return;
        }
        if (m() > 1) k = j;
    }

    void step5() {
        j = k;
        if (b[k] == 'e') {
            int a = m();
            if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
        }
        if (b[k] == 'l' && doublec(k) && m() > 1) k--;
    }

public:
    std::string stem(const std::string& word) {
        if (word.size() <= 2) return word;
        b = word;
        k = static_cast<int>(word.size()) - 1;
        step1ab();
        if (k > 0) step1c();
        if (k > 0) step2();
        if (k > 0) step3();
        if (k > 0) step4();
        if (k > 0) step5();
        return b.substr(0, k + 1);
    }
};

// Light S-stemmer (plural stripping only) — the previous default, kept so the
// two stemmers can be compared for the report.
static inline std::string s_stem(const std::string& word) {
    if (word.size() <= 3) return word;
    if (word.back() != 's') return word;
    if (word.size() > 4 && word.compare(word.size() - 3, 3, "ies") == 0)
        return word.substr(0, word.size() - 3) + "y";
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "es") == 0) {
        char third_last = word[word.size() - 3];
        if (third_last == 's' || third_last == 'x' || third_last == 'z' ||
            (word.size() > 4 && word.compare(word.size() - 4, 2, "ch") == 0) ||
            (word.size() > 4 && word.compare(word.size() - 4, 2, "sh") == 0))
            return word.substr(0, word.size() - 2);
        return word.substr(0, word.size() - 1);
    }
    if (word.back() == 's' && word[word.size() - 2] != 's' && word[word.size() - 2] != 'u')
        return word.substr(0, word.size() - 1);
    return word;
}

// Index/query stopword list: the standard NLTK English set, which is a pure
// function-word list (pronouns, determiners, prepositions, auxiliaries,
// interrogatives). Two deliberate deviations from NLTK's literal contents:
//
//   * single letters ("s", "t", "d", "m", "o", "y", ...) are NOT stopped.
//     NLTK only lists them as leftovers of its contraction tokeniser, but in
//     a biomedical collection they are content: "vitamin D", "T cells",
//     "B cells", "IgM". Dropping them silently breaks those queries.
//   * contraction fragments ("don", "won", "ain", "isn", ...) are not stopped
//     either. They are tokeniser artefacts, and "don" collides with the stem
//     of "donning" (as in donning PPE).
//
// This is deliberately larger than the 33-word Lucene/BEIR set that was here
// before. The reason is the query side, not the document side: these topics
// are natural-language questions, so they lead with "what", "how", "which",
// "does", "who". Those are not in the Lucene set, they have enough IDF to
// matter (df("what") = 2.7% of the corpus), and a document that merely
// repeats "What ... What ..." in its title was outranking documents that were
// actually on topic. Note it is still a function-word list only: the content
// words the large INQUERY list below drops ("case", "state", "number",
// "group", "point", "work") stay indexed, because in a biomedical collection
// they are content. The INQUERY list is still used, separately, to prune RM3
// expansion candidates, where being aggressive is appropriate.
static const std::unordered_set<std::string> INDEX_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of","at","by","for",
    "with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when","where","why","how","all",
    "any","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","can","will","just","should","now"
};

struct TokenizerCfg {
    uint8_t stemmer = 2;       // 0 = none, 1 = S-stemmer, 2 = Porter
    uint8_t remove_stop = 1;   // 0 = keep stopwords, 1 = drop them
};

static thread_local PorterStemmer g_porter;
static thread_local std::unordered_map<std::string, std::string> g_stem_cache;

static inline std::string apply_stem(const std::string& w, uint8_t stemmer) {
    if (stemmer == 0) return w;
    if (stemmer == 1) return s_stem(w);
    auto it = g_stem_cache.find(w);
    if (it != g_stem_cache.end()) return it->second;
    std::string s = g_porter.stem(w);
    g_stem_cache.emplace(w, s);
    return s;
}

// Returns (term, original_token_position) pairs.
static inline std::vector<std::pair<std::string, uint32_t>>
cpp_tokenize(const std::string& text, const TokenizerCfg& cfg) {
    std::vector<std::pair<std::string, uint32_t>> tokens;
    tokens.reserve(text.size() / 6);
    std::string current;
    uint32_t pos = 0;

    auto emit = [&]() {
        if (current.empty()) return;
        uint32_t this_pos = pos++;                       // position counted first
        if (!(cfg.remove_stop && INDEX_STOPWORDS.count(current)))
            tokens.push_back({apply_stem(current, cfg.stemmer), this_pos});
        current.clear();
    };

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            current += std::tolower(static_cast<unsigned char>(c));
        else
            emit();
    }
    emit();
    return tokens;
}

// ---------------------------------------------------------------------------
// mmap wrapper: postings and the forward index are read straight out of the
// page cache, so load_index() does no parsing of the bulk data at all.
// ---------------------------------------------------------------------------
struct MmapBuffer {
    const uint8_t* data = nullptr;
    size_t length = 0;
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;

    bool map_file(const std::string& path) {
        unmap();
        // FILE_SHARE_DELETE matters: without it the mapping locks the file for
        // the life of the process, and a caller that deletes index_dir while the
        // engine is still loaded (the harness does exactly that, and so does
        // pytest's temporary-directory cleanup) fails with a sharing violation.
        // POSIX mmap has no such restriction, which is why the branch below
        // needs no equivalent.
        hFile = CreateFileA(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz) || sz.QuadPart == 0) { unmap(); return false; }
        length = static_cast<size_t>(sz.QuadPart);
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { unmap(); return false; }
        void* addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { unmap(); return false; }
        data = static_cast<const uint8_t*>(addr);
        return true;
    }

    void unmap() {
        if (data) { UnmapViewOfFile(const_cast<uint8_t*>(data)); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
        length = 0;
    }
#else
    bool map_file(const std::string& path) {
        unmap();
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) return false;
        struct stat st;
        if (fstat(fd, &st) == -1 || st.st_size == 0) { close(fd); return false; }
        length = st.st_size;
        void* addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) { data = nullptr; length = 0; return false; }
        data = static_cast<const uint8_t*>(addr);
        return true;
    }

    void unmap() {
        if (data) {
            munmap(const_cast<uint8_t*>(data), length);
            data = nullptr;
            length = 0;
        }
    }
#endif

    ~MmapBuffer() { unmap(); }
};

// ---------------------------------------------------------------------------
// Per-term dictionary entry.
//
// Dictionary entries hold df, cf and a postings offset. A previous version
// also stored per-term max_tf / min_qlen for block-max WAND; those were
// written and then never consulted, so they are gone.
// ---------------------------------------------------------------------------
struct TermMeta {
    uint32_t df = 0;
    uint8_t  has_title = 0;     // 1 iff the EF list has a sparse title-hit stream
    uint64_t cf = 0;
    uint64_t postings_offset = 0;
};

struct DocPosting {
    uint32_t doc_id;
    uint32_t tf;
    uint32_t title_tf = 0;      // occurrences within the leading title-window tokens
};

// Minimal JSONL field extraction for {"doc_id": ..., "text": ...}
static bool parse_jsonl_line(const std::string& line, std::string& doc_id, std::string& text) {
    if (line.empty()) return false;
    size_t id_pos = line.find("\"doc_id\"");
    if (id_pos == std::string::npos) return false;
    size_t id_colon = line.find(':', id_pos);
    if (id_colon == std::string::npos) return false;
    size_t id_quote1 = line.find('"', id_colon);
    if (id_quote1 == std::string::npos) return false;
    size_t id_quote2 = line.find('"', id_quote1 + 1);
    while (id_quote2 != std::string::npos && line[id_quote2 - 1] == '\\') {
        id_quote2 = line.find('"', id_quote2 + 1);
    }
    if (id_quote2 == std::string::npos) return false;
    doc_id = line.substr(id_quote1 + 1, id_quote2 - id_quote1 - 1);

    size_t text_pos = line.find("\"text\"");
    if (text_pos == std::string::npos) return false;
    size_t text_colon = line.find(':', text_pos);
    if (text_colon == std::string::npos) return false;
    size_t text_quote1 = line.find('"', text_colon);
    if (text_quote1 == std::string::npos) return false;
    size_t text_quote2 = line.rfind('"');
    if (text_quote2 <= text_quote1) return false;
    while (text_quote2 > text_quote1 && line[text_quote2 - 1] == '\\') {
        size_t prev = line.rfind('"', text_quote2 - 1);
        if (prev <= text_quote1) break;
        text_quote2 = prev;
    }

    std::string raw_text = line.substr(text_quote1 + 1, text_quote2 - text_quote1 - 1);
    text.clear();
    text.reserve(raw_text.size());
    for (size_t i = 0; i < raw_text.size(); ++i) {
        if (raw_text[i] == '\\' && i + 1 < raw_text.size()) {
            char next = raw_text[i + 1];
            if (next == 'n') { text += '\n'; ++i; }
            else if (next == 't') { text += '\t'; ++i; }
            else if (next == '"') { text += '"'; ++i; }
            else if (next == '\\') { text += '\\'; ++i; }
            else if (next == 'r') { ++i; }
            else { text += raw_text[i]; }
        } else {
            text += raw_text[i];
        }
    }
    return true;
}

class CPPEngine {
public:
    // ---- corpus-level state -------------------------------------------------
    std::vector<std::string> doc_ids;
    std::vector<uint8_t> quantized_doc_lens;
    std::vector<uint32_t> doc_lens;
    float avg_doc_len = 0.0f;
    uint32_t N = 0;
    uint64_t total_corpus_tokens = 0;
    TokenizerCfg tokcfg;          // read from meta.bin at load; must match build
    BuildCfg buildcfg;            // build-time only; never needed at query time

    // ---- dictionary, held as parallel sorted arrays -------------------------
    // Sorted term array + binary search rather than a hash map: on a corpus
    // with millions of distinct terms this loads faster and uses far less
    // memory, and the array index doubles as the term id used by fwd.bin.
    std::vector<std::string> terms_sorted;
    std::vector<TermMeta> term_metas;
    std::vector<uint32_t> rm3_to_tid;   // dense RM3 id -> global term id
    uint32_t fwd_off_stride = FWD_OFF_STRIDE;

    MmapBuffer postings_mmap;
    MmapBuffer fwd_mmap;
    MmapBuffer fwdoff_mmap;

    // ---- query-time scratch -------------------------------------------------
    // A dense accumulator plus a dirty list. Scoring touches acc[doc] directly
    // (no hashing) and only the touched entries are reset after each query.
    std::vector<float> acc;
    std::vector<uint32_t> touched;
    std::vector<uint32_t> scratch_gaps;
    std::vector<uint32_t> scratch_tfs;
    std::vector<uint32_t> scratch_titles;
    std::vector<uint32_t> fwd_order;
    std::vector<float> fb_wts;
    std::vector<uint32_t> rid_scratch;

    struct BatchFwd {
        std::vector<std::pair<uint32_t, uint32_t>> arena;
        std::vector<uint32_t> start;
        std::vector<uint32_t> len;
    };

    CPPEngine() = default;

    int32_t find_term(const std::string& t) const {
        auto it = std::lower_bound(terms_sorted.begin(), terms_sorted.end(), t);
        if (it == terms_sorted.end() || *it != t) return -1;
        return static_cast<int32_t>(it - terms_sorted.begin());
    }

    // =======================================================================
    // BUILD
    // =======================================================================
    //
    // Three stages:
    //   1. SPIMI  — tokenise, buffer postings in memory, spill sorted runs.
    //               While doing this, also spill each document's top candidate
    //               terms (by tf) to a temp file, so the forward index costs
    //               no second tokenisation pass over the corpus.
    //   2. MERGE  — k-way merge the runs into postings.bin, writing dict.bin
    //               front-coded in the same sorted pass.
    //   3. FWD    — map the buffered candidate terms to term ids, drop the
    //               ones the df filter rejects, write fwd.bin / fwdoff.bin.
    //
    template <typename Supplier>
    void build_common(Supplier&& next_doc, const std::string& index_dir) {
        postings_mmap.unmap();
        fwd_mmap.unmap();
        fwdoff_mmap.unmap();

        doc_ids.clear();
        doc_lens.clear();
        total_corpus_tokens = 0;
        N = 0;

        // Analysis chain is chosen at build time and frozen into the index.
        // Overridable from the environment so the two configurations can be
        // A/B-ed without editing and recompiling:
        //     IR_STEMMER=0|1|2   IR_STOPWORDS=0|1
        buildcfg = read_build_cfg();
        tokcfg = TokenizerCfg{};
        if (const char* e = std::getenv("IR_STEMMER"))
            tokcfg.stemmer = static_cast<uint8_t>(std::atoi(e));
        if (const char* e = std::getenv("IR_STOPWORDS"))
            tokcfg.remove_stop = static_cast<uint8_t>(std::atoi(e));

        std::vector<std::string> run_files;
        uint32_t run_count = 0;

        std::unordered_map<std::string, std::vector<DocPosting>> temp_postings;
        std::unordered_map<std::string, uint64_t> collection_freq;

        const std::string fwdtmp_path = index_dir + "/fwdtmp.bin";
        std::ofstream f_fwdtmp(fwdtmp_path, std::ios::binary);

        auto flush_spimi_run = [&](uint32_t run_idx) {
            std::string run_path = index_dir + "/spimi_run_" + std::to_string(run_idx) + ".bin";
            std::ofstream f(run_path, std::ios::binary);

            std::vector<std::string> terms;
            terms.reserve(temp_postings.size());
            for (const auto& kv : temp_postings) terms.push_back(kv.first);
            std::sort(terms.begin(), terms.end());

            uint32_t num_terms = terms.size();
            f.write(reinterpret_cast<const char*>(&num_terms), sizeof(num_terms));

            for (const auto& term : terms) {
                const auto& plist = temp_postings[term];
                uint16_t t_len = term.size();
                f.write(reinterpret_cast<const char*>(&t_len), sizeof(t_len));
                f.write(term.data(), t_len);

                uint32_t df = plist.size();
                f.write(reinterpret_cast<const char*>(&df), sizeof(df));

                for (const auto& p : plist) {
                    f.write(reinterpret_cast<const char*>(&p.doc_id), sizeof(p.doc_id));
                    f.write(reinterpret_cast<const char*>(&p.tf), sizeof(p.tf));
                    f.write(reinterpret_cast<const char*>(&p.title_tf), sizeof(p.title_tf));
                }
            }
            temp_postings.clear();
            run_files.push_back(run_path);
        };

        std::string ext_id, text;
        uint32_t doc_int_id = 0;

        while (next_doc(ext_id, text)) {
            doc_ids.push_back(ext_id);

            auto tokens = cpp_tokenize(text, tokcfg);
            // Document length is the number of INDEXED tokens, so BM25's
            // length normalisation is consistent with what is actually scored.
            uint32_t d_len = static_cast<uint32_t>(tokens.size());
            doc_lens.push_back(d_len);
            total_corpus_tokens += d_len;

            std::unordered_map<std::string, uint32_t> term_data;  // term -> tf
            std::unordered_map<std::string, uint32_t> title_tf;   // term -> tf in the title window
            for (const auto& tp : tokens) {
                term_data[tp.first]++;
                if (tp.second < buildcfg.title_window) title_tf[tp.first]++;
                collection_freq[tp.first]++;
            }

            // Forward-index candidates: the highest-tf non-stopword terms in
            // this document. Written now, resolved to term ids after merge.
            std::vector<std::pair<uint32_t, const std::string*>> cands;
            cands.reserve(term_data.size());
            for (const auto& kv : term_data) {
                if (STOPWORDS.count(kv.first)) continue;
                cands.push_back({kv.second, &kv.first});
            }
            size_t keep = std::min<size_t>(cands.size(), FWD_CANDIDATES);
            std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            uint32_t n_cand = static_cast<uint32_t>(keep);
            f_fwdtmp.write(reinterpret_cast<const char*>(&n_cand), sizeof(n_cand));
            for (size_t i = 0; i < keep; ++i) {
                uint16_t tl = static_cast<uint16_t>(cands[i].second->size());
                uint16_t tf16 = static_cast<uint16_t>(std::min<uint32_t>(cands[i].first, 65535u));
                f_fwdtmp.write(reinterpret_cast<const char*>(&tl), sizeof(tl));
                f_fwdtmp.write(cands[i].second->data(), tl);
                f_fwdtmp.write(reinterpret_cast<const char*>(&tf16), sizeof(tf16));
            }

            for (const auto& kv : term_data) {
                auto ttf = title_tf.find(kv.first);
                temp_postings[kv.first].push_back(
                    DocPosting{doc_int_id, kv.second,
                               (ttf == title_tf.end()) ? 0u : ttf->second});
            }

            doc_int_id++;
            if (doc_int_id % SPIMI_BATCH_DOCS == 0) flush_spimi_run(run_count++);
        }

        if (!temp_postings.empty()) flush_spimi_run(run_count++);
        f_fwdtmp.close();

        N = doc_int_id;
        avg_doc_len = (N > 0) ? static_cast<float>(static_cast<double>(total_corpus_tokens) / N) : 0.0f;

        quantized_doc_lens.resize(N);
        for (uint32_t i = 0; i < N; ++i) {
            float ratio = (avg_doc_len > 0) ? (static_cast<float>(doc_lens[i]) / avg_doc_len) : 1.0f;
            quantized_doc_lens[i] = quantize_doc_len(ratio);
        }

        // -------- meta.bin: analysis config, needed to tokenise queries -----
        {
            std::ofstream f(index_dir + "/meta.bin", std::ios::binary);
            uint8_t version = INDEX_VERSION;
            f.write(reinterpret_cast<const char*>(&version), 1);
            f.write(reinterpret_cast<const char*>(&tokcfg.stemmer), 1);
            f.write(reinterpret_cast<const char*>(&tokcfg.remove_stop), 1);
        }

        // -------- docmap.bin / doclens.bin ---------------------------------
        {
            std::ofstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            uint16_t id_len = 0;
            if (N > 0) {
                id_len = static_cast<uint16_t>(doc_ids[0].size());
                for (const auto& id : doc_ids) {
                    if (id.size() != id_len) { id_len = 0; break; }
                }
            }
            f.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
            if (id_len > 0) {
                // TREC-COVID cord_uids are all 8 bytes — one length in the
                // header, then packed ids. Saves a uint16 per document.
                for (const auto& id : doc_ids) f.write(id.data(), id_len);
            } else {
                // Variable-width ids (toy set): front-code against the
                // previous id, same codec as dict.bin.
                std::vector<uint8_t> buf;
                buf.reserve(static_cast<size_t>(N) * 8);
                std::string prev;
                for (const auto& id : doc_ids) {
                    size_t prefix = 0;
                    size_t maxp = std::min(prev.size(), id.size());
                    while (prefix < maxp && prev[prefix] == id[prefix]) prefix++;
                    if (prefix > 255) prefix = 255;
                    write_vbyte(buf, static_cast<uint32_t>(prefix));
                    write_vbyte(buf, static_cast<uint32_t>(id.size() - prefix));
                    buf.insert(buf.end(), id.begin() + prefix, id.end());
                    prev = id;
                }
                f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
            }
        }
        {
            std::ofstream f(index_dir + "/doclens.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            f.write(reinterpret_cast<const char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.write(reinterpret_cast<const char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            f.write(reinterpret_cast<const char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));
        }

        // -------- stage 2: k-way merge -------------------------------------
        struct RunReader {
            std::ifstream file;
            uint32_t num_terms = 0;
            uint32_t terms_read = 0;
            std::string curr_term;
            uint32_t curr_df = 0;
            bool active = false;

            bool read_next_term() {
                if (terms_read >= num_terms) { active = false; curr_term.clear(); return false; }
                uint16_t t_len;
                if (!file.read(reinterpret_cast<char*>(&t_len), sizeof(t_len))) {
                    active = false; curr_term.clear(); return false;
                }
                curr_term.resize(t_len);
                file.read(&curr_term[0], t_len);
                file.read(reinterpret_cast<char*>(&curr_df), sizeof(curr_df));
                terms_read++;
                active = true;
                return true;
            }
        };

        std::vector<RunReader> readers(run_files.size());
        for (size_t i = 0; i < run_files.size(); ++i) {
            readers[i].file.open(run_files[i], std::ios::binary);
            readers[i].file.read(reinterpret_cast<char*>(&readers[i].num_terms), sizeof(readers[i].num_terms));
            readers[i].read_next_term();
        }

        std::ofstream f_postings(index_dir + "/postings.bin", std::ios::binary);
        std::ofstream f_dict(index_dir + "/dict.bin", std::ios::binary | std::ios::out);

        uint32_t placeholder = 0;
        f_dict.write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));

        uint64_t current_postings_offset = 0;
        uint64_t prev_postings_offset = 0;
        std::string prev_term;
        uint32_t num_written_terms = 0;

        std::vector<uint8_t> dict_buf;
        dict_buf.reserve(1 << 20);

        std::vector<DocPosting> merged;
        std::vector<uint8_t> block_data;
        std::vector<uint64_t> cfs_in_order;

        while (true) {
            const std::string* min_term = nullptr;
            for (auto& r : readers) {
                if (!r.active) continue;
                if (!min_term || r.curr_term < *min_term) min_term = &r.curr_term;
            }
            if (!min_term) break;
            std::string term = *min_term;

            uint32_t df = 0;
            for (auto& r : readers) if (r.active && r.curr_term == term) df += r.curr_df;

            merged.clear();
            merged.reserve(df);
            uint64_t merged_cf = 0;

            for (auto& r : readers) {
                if (!r.active || r.curr_term != term) continue;
                for (uint32_t p_idx = 0; p_idx < r.curr_df; ++p_idx) {
                    DocPosting dp;
                    r.file.read(reinterpret_cast<char*>(&dp.doc_id), sizeof(dp.doc_id));
                    r.file.read(reinterpret_cast<char*>(&dp.tf), sizeof(dp.tf));
                    r.file.read(reinterpret_cast<char*>(&dp.title_tf), sizeof(dp.title_tf));
                    merged_cf += dp.tf;
                    merged.push_back(std::move(dp));
                }
                r.read_next_term();
            }

            // Runs are produced in increasing doc_id order and merged in run
            // order, so `merged` is already sorted by doc_id.
            if (df < buildcfg.min_df) continue;

            uint64_t term_postings_start = current_postings_offset;
            block_data.clear();
            scratch_gaps.resize(df);
            scratch_tfs.resize(df);
            scratch_titles.resize(df);
            uint8_t has_title = 0;
            uint32_t n_title_hits = 0;
            for (uint32_t i = 0; i < df; ++i) {
                scratch_gaps[i] = merged[i].doc_id;
                uint32_t tf = std::min(merged[i].tf, TF_MAX);
                scratch_tfs[i] = tf ? tf - 1 : 0;   // Simple-9 stores tf-1
                scratch_titles[i] = std::min(merged[i].title_tf, TITLE_TF_MAX);
                if (scratch_titles[i]) { has_title = 1; ++n_title_hits; }
            }
            if (df < POST_VBYTE_DF) {
                uint32_t prev_doc_id = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    uint32_t gap = scratch_gaps[i] - prev_doc_id;
                    uint8_t packed = pack_tf(merged[i].tf, scratch_titles[i]);
                    uint32_t plain = (packed == 1) ? 1u : 0u;
                    write_vbyte(block_data, (gap << 1) | plain);
                    if (!plain) block_data.push_back(packed);
                    prev_doc_id = scratch_gaps[i];
                }
            } else {
                simple9_pack(block_data, scratch_tfs.data(), df);
                if (has_title) {
                    write_vbyte(block_data, n_title_hits);
                    uint32_t prev = 0;
                    for (uint32_t i = 0; i < df; ++i) {
                        if (!scratch_titles[i]) continue;
                        write_vbyte(block_data, ((i - prev) << 2) | scratch_titles[i]);
                        prev = i;
                    }
                }
                ef_pack(block_data, scratch_gaps.data(), df);
            }
            f_postings.write(reinterpret_cast<const char*>(block_data.data()), block_data.size());
            current_postings_offset += block_data.size();
            cfs_in_order.push_back(merged_cf);

            // -------- dict entry, front-coded against the previous term ----
            size_t prefix = 0;
            size_t maxp = std::min(prev_term.size(), term.size());
            while (prefix < maxp && prev_term[prefix] == term[prefix]) prefix++;
            if (prefix > 255) prefix = 255;   // keep the shared prefix in one byte

            write_vbyte(dict_buf, static_cast<uint32_t>(prefix));
            write_vbyte(dict_buf, static_cast<uint32_t>(term.size() - prefix));
            dict_buf.insert(dict_buf.end(), term.begin() + prefix, term.end());
            write_vbyte(dict_buf, (df << 1) | has_title);
            write_vbyte64(dict_buf, term_postings_start - prev_postings_offset);
            prev_postings_offset = term_postings_start;
            prev_term = term;
            num_written_terms++;

            if (dict_buf.size() > (1 << 20)) {
                f_dict.write(reinterpret_cast<const char*>(dict_buf.data()), dict_buf.size());
                dict_buf.clear();
            }
        }

        if (!dict_buf.empty()) {
            f_dict.write(reinterpret_cast<const char*>(dict_buf.data()), dict_buf.size());
            dict_buf.clear();
        }
        f_dict.seekp(0);
        f_dict.write(reinterpret_cast<const char*>(&num_written_terms), sizeof(num_written_terms));
        f_dict.close();
        // Pad so the last Elias-Fano list can over-read up to 7 bytes in
        // bit_get without walking off the mmap (page-multiple postings.bin).
        const uint8_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        f_postings.write(reinterpret_cast<const char*>(pad), 8);
        f_postings.close();

        for (auto& r : readers) r.file.close();
        for (const auto& rfile : run_files) std::remove(rfile.c_str());

        // -------- stage 3: forward index -----------------------------------
        // We need the dictionary in memory to resolve term ids and apply the
        // df filter, so load what we just wrote and then stream fwdtmp.bin.
        load_dictionary(index_dir);
        for (size_t i = 0; i < term_metas.size() && i < cfs_in_order.size(); ++i)
            term_metas[i].cf = cfs_in_order[i];
        write_forward_index(index_dir, fwdtmp_path);
        std::remove(fwdtmp_path.c_str());
    }

    void build_index(const std::vector<std::pair<std::string, std::string>>& corpus,
                     const std::string& index_dir) {
        size_t i = 0;
        build_common([&](std::string& id, std::string& text) {
            if (i >= corpus.size()) return false;
            id = corpus[i].first;
            text = corpus[i].second;
            ++i;
            return true;
        }, index_dir);
    }

    void build_index_from_file(const std::string& corpus_path, const std::string& index_dir) {
        std::ifstream file(corpus_path);
        if (!file.is_open()) throw std::runtime_error("Cannot open corpus file: " + corpus_path);
        std::string line;
        build_common([&](std::string& id, std::string& text) {
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                if (parse_jsonl_line(line, id, text)) return true;
            }
            return false;
        }, index_dir);
    }

    // -----------------------------------------------------------------------
    // Forward index: doc -> its top buildcfg.fwd_terms terms (term id + tf),
    // restricted to terms RM3 would ever consider as expansion candidates.
    //
    // This exists purely so RM3 does not have to find "which terms occur in
    // these 10 feedback documents" by scanning the entire dictionary and
    // decoding every postings list, which is O(vocabulary) per query.
    // -----------------------------------------------------------------------
    void write_forward_index(const std::string& index_dir, const std::string& fwdtmp_path) {
        std::ifstream f_in(fwdtmp_path, std::ios::binary);
        std::ofstream f_fwd(index_dir + "/fwd.bin", std::ios::binary);
        std::ofstream f_off(index_dir + "/fwdoff.bin", std::ios::binary);

        const uint32_t df_min = 5;
        const uint32_t df_max = std::max<uint32_t>(
            df_min, static_cast<uint32_t>(buildcfg.rm3_df_frac * static_cast<float>(N)));

        // Dense RM3 id space: only terms that can appear in fwd.bin get an
        // id. Assigned in dictionary (tid) order so rm3map.bin can delta-code
        // tids with an unsigned gap. Ascending-df order compresses Elias-Fano
        // rids slightly more but needs a zigzag map; unsigned deltas are the
        // format the rest of the engine already uses.
        std::vector<int32_t> tid_to_rm3(term_metas.size(), -1);
        rm3_to_tid.clear();
        for (uint32_t tid = 0; tid < term_metas.size(); ++tid) {
            uint32_t df = term_metas[tid].df;
            if (df >= df_min && df <= df_max) {
                tid_to_rm3[tid] = static_cast<int32_t>(rm3_to_tid.size());
                rm3_to_tid.push_back(tid);
            }
        }

        {
            std::ofstream f_map(index_dir + "/rm3map.bin", std::ios::binary);
            uint32_t M = static_cast<uint32_t>(rm3_to_tid.size());
            f_map.write(reinterpret_cast<const char*>(&M), sizeof(M));
            std::vector<uint8_t> mapbuf;
            mapbuf.reserve(M * 2);
            uint32_t prev = 0;
            for (uint32_t tid : rm3_to_tid) {
                write_vbyte(mapbuf, tid - prev);
                write_vbyte64(mapbuf, term_metas[tid].cf - term_metas[tid].df);
                prev = tid;
            }
            f_map.write(reinterpret_cast<const char*>(mapbuf.data()), mapbuf.size());
        }

        fwd_off_stride = FWD_OFF_STRIDE;
        f_off.write(reinterpret_cast<const char*>(&N), sizeof(N));
        f_off.write(reinterpret_cast<const char*>(&fwd_off_stride), sizeof(fwd_off_stride));

        uint32_t offset = 0;
        std::vector<uint8_t> out;
        std::vector<std::pair<uint32_t, uint32_t>> kept;   // (rm3_id, tf)
        std::vector<std::tuple<float, uint32_t, uint32_t>> pool;  // (tf*idf, rid, tf)
        std::vector<uint32_t> ef_scratch;
        std::vector<uint8_t> ef_block;
        std::string term;

        for (uint32_t d = 0; d < N; ++d) {
            if (d % fwd_off_stride == 0)
                f_off.write(reinterpret_cast<const char*>(&offset), sizeof(offset));

            uint32_t n_cand = 0;
            f_in.read(reinterpret_cast<char*>(&n_cand), sizeof(n_cand));
            pool.clear();

            for (uint32_t i = 0; i < n_cand; ++i) {
                uint16_t tl, tf16;
                f_in.read(reinterpret_cast<char*>(&tl), sizeof(tl));
                term.resize(tl);
                f_in.read(&term[0], tl);
                f_in.read(reinterpret_cast<char*>(&tf16), sizeof(tf16));

                // Consume every candidate — do not break early or the next
                // document reads from the wrong offset.
                int32_t tid = find_term(term);
                if (tid < 0) continue;
                int32_t rid = tid_to_rm3[static_cast<uint32_t>(tid)];
                if (rid < 0) continue;
                uint32_t df = term_metas[static_cast<uint32_t>(tid)].df;
                float idf = (df > 0 && N > 0)
                    ? std::log(static_cast<float>(N) / static_cast<float>(df))
                    : 0.0f;
                pool.push_back({static_cast<float>(tf16) * idf,
                                static_cast<uint32_t>(rid), tf16});
            }

            // Prefer distinctive terms over raw-tf collection generalists so
            // the 24 slots actually feed RM3.
            size_t take = std::min<size_t>(pool.size(), buildcfg.fwd_terms);
            if (take < pool.size()) {
                std::partial_sort(pool.begin(), pool.begin() + take, pool.end(),
                                  [](const auto& a, const auto& b) {
                                      return std::get<0>(a) > std::get<0>(b);
                                  });
                pool.resize(take);
            }
            kept.clear();
            kept.reserve(pool.size());
            for (const auto& t : pool) kept.push_back({std::get<1>(t), std::get<2>(t)});
            std::sort(kept.begin(), kept.end());

            // count | ef_len | Simple-9 tfs | Elias-Fano rids.
            // Splitting streams avoids stuffing 3 tf bits into a ~16K mean
            // rid gap (which pushed almost every varint over a byte boundary).
            // Simple-9 keeps the real tf for P(w|D); nibbles capped at 15
            // cost ~0.03 nDCG on this collection.
            out.clear();
            const uint32_t nk = static_cast<uint32_t>(kept.size());
            write_vbyte(out, nk);
            ef_scratch.clear();
            std::vector<uint32_t> tfs_u32;
            tfs_u32.reserve(nk);
            for (uint32_t i = 0; i < nk; ++i) {
                ef_scratch.push_back(kept[i].first);
                tfs_u32.push_back(kept[i].second);
            }
            std::vector<uint8_t> tf_block;
            if (nk) simple9_pack(tf_block, tfs_u32.data(), nk);
            ef_block.clear();
            if (nk) ef_pack(ef_block, ef_scratch.data(), nk);
            write_vbyte(out, static_cast<uint32_t>(ef_block.size()));
            out.insert(out.end(), tf_block.begin(), tf_block.end());
            out.insert(out.end(), ef_block.begin(), ef_block.end());
            f_fwd.write(reinterpret_cast<const char*>(out.data()), out.size());
            offset += static_cast<uint32_t>(out.size());
        }
        const uint8_t fwd_pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        f_fwd.write(reinterpret_cast<const char*>(fwd_pad), 8);
    }

    // =======================================================================
    // LOAD
    // =======================================================================
    void load_dictionary(const std::string& index_dir) {
        std::ifstream f(index_dir + "/dict.bin", std::ios::binary | std::ios::ate);
        if (!f.is_open()) throw std::runtime_error("Cannot open dict.bin");
        std::streamsize sz = f.tellg();
        f.seekg(0);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);

        uint32_t num_terms = 0;
        std::memcpy(&num_terms, buf.data(), sizeof(num_terms));

        terms_sorted.clear();
        term_metas.clear();
        terms_sorted.reserve(num_terms);
        term_metas.reserve(num_terms);

        const uint8_t* p = buf.data() + sizeof(num_terms);
        std::string prev_term;
        uint64_t prev_postings_offset = 0;

        for (uint32_t i = 0; i < num_terms; ++i) {
            uint32_t prefix = read_vbyte(p);
            uint32_t suffix_len = read_vbyte(p);
            std::string term = prev_term.substr(0, prefix);
            term.append(reinterpret_cast<const char*>(p), suffix_len);
            p += suffix_len;

            TermMeta m;
            uint32_t df_coded = read_vbyte(p);
            m.df = df_coded >> 1;
            m.has_title = static_cast<uint8_t>(df_coded & 1u);
            m.postings_offset = prev_postings_offset + read_vbyte64(p);
            prev_postings_offset = m.postings_offset;

            terms_sorted.push_back(term);
            term_metas.push_back(m);
            prev_term = std::move(term);
        }
    }

    void load_rm3_map(const std::string& index_dir) {
        rm3_to_tid.clear();
        std::ifstream f(index_dir + "/rm3map.bin", std::ios::binary | std::ios::ate);
        if (!f.is_open()) return;
        std::streamsize sz = f.tellg();
        if (sz < static_cast<std::streamsize>(sizeof(uint32_t))) return;
        f.seekg(0);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        uint32_t M = 0;
        std::memcpy(&M, buf.data(), sizeof(M));
        rm3_to_tid.resize(M);
        const uint8_t* p = buf.data() + sizeof(M);
        uint32_t tid = 0;
        for (uint32_t i = 0; i < M; ++i) {
            tid += read_vbyte(p);
            rm3_to_tid[i] = tid;
            uint64_t cf_extra = read_vbyte64(p);
            if (tid < term_metas.size())
                term_metas[tid].cf = cf_extra + term_metas[tid].df;
        }
    }

    void load_index(const std::string& index_dir) {
        {
            // Query tokenisation must match index tokenisation exactly.
            std::ifstream f(index_dir + "/meta.bin", std::ios::binary);
            if (f.is_open()) {
                uint8_t version = 0;
                f.read(reinterpret_cast<char*>(&version), 1);
                if (version != INDEX_VERSION) {
                    throw std::runtime_error(
                        "index version mismatch (got " + std::to_string(version) +
                        ", need " + std::to_string(INDEX_VERSION) + "); rebuild");
                }
                f.read(reinterpret_cast<char*>(&tokcfg.stemmer), 1);
                f.read(reinterpret_cast<char*>(&tokcfg.remove_stop), 1);
            }
        }
        {
            std::ifstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            uint16_t id_len = 0;
            f.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
            doc_ids.assign(N, std::string());
            if (id_len > 0) {
                for (uint32_t i = 0; i < N; ++i) {
                    doc_ids[i].resize(id_len);
                    f.read(&doc_ids[i][0], id_len);
                }
            } else {
                std::ifstream::pos_type here = f.tellg();
                f.seekg(0, std::ios::end);
                std::streamsize rest = f.tellg() - here;
                f.seekg(here);
                std::vector<uint8_t> buf(static_cast<size_t>(rest));
                f.read(reinterpret_cast<char*>(buf.data()), rest);
                const uint8_t* p = buf.data();
                std::string prev;
                for (uint32_t i = 0; i < N; ++i) {
                    uint32_t prefix = read_vbyte(p);
                    uint32_t slen = read_vbyte(p);
                    std::string id = prev.substr(0, prefix);
                    id.append(reinterpret_cast<const char*>(p), slen);
                    p += slen;
                    doc_ids[i] = id;
                    prev = std::move(id);
                }
            }
        }
        {
            std::ifstream f(index_dir + "/doclens.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            f.read(reinterpret_cast<char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.read(reinterpret_cast<char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            quantized_doc_lens.resize(N);
            f.read(reinterpret_cast<char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));

            // Approximate exact lengths back out of the quantised ratios; RM3's
            // Dirichlet smoothing only needs them to within a few percent.
            doc_lens.resize(N);
            for (uint32_t i = 0; i < N; ++i) {
                doc_lens[i] = static_cast<uint32_t>(dequantize_doc_len(quantized_doc_lens[i]) * avg_doc_len);
                if (doc_lens[i] == 0) doc_lens[i] = 1;
            }
        }

        load_dictionary(index_dir);
        load_rm3_map(index_dir);

        postings_mmap.map_file(index_dir + "/postings.bin");
        fwd_mmap.map_file(index_dir + "/fwd.bin");
        fwdoff_mmap.map_file(index_dir + "/fwdoff.bin");
        if (fwdoff_mmap.data && fwdoff_mmap.length >= 8) {
            std::memcpy(&fwd_off_stride, fwdoff_mmap.data + sizeof(uint32_t), sizeof(uint32_t));
            if (fwd_off_stride == 0) fwd_off_stride = 1;
        }

        acc.assign(N, 0.0f);
        touched.clear();
        touched.reserve(N);
    }

    // =======================================================================
    // SCORING
    // =======================================================================
    //
    // Term-at-a-time accumulation into a dense array, then one partial_sort.
    //
    // The previous version maintained the top-k list inside the posting loop
    // with a linear search plus a full std::sort *per posting*, which is
    // O(P * k log k) for P postings — that was the dominant query cost.
    //
    std::vector<std::pair<uint32_t, float>> raw_retrieve(
        const std::vector<std::pair<int32_t, float>>& qterms,
        int top_k, float k1, float b, float delta, float w_title, float title_bonus)
    {
        if (qterms.empty() || N == 0 || !postings_mmap.data) return {};

        // Length-normalisation denominator for each of the 256 quantised
        // length buckets, computed once per query rather than per posting.
        float length_penalty_lut[256];
        for (int q = 0; q < 256; ++q) {
            float ratio = dequantize_doc_len(static_cast<uint8_t>(q));
            length_penalty_lut[q] = k1 * (1.0f - b + b * ratio);
        }

        for (const auto& qt : qterms) {
            int32_t tid = qt.first;
            if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
            const TermMeta& meta = term_metas[tid];
            if (meta.df == 0) continue;

            float idf = std::log(1.0f + (static_cast<float>(N) - meta.df + 0.5f) / (meta.df + 0.5f));
            float w = qt.second * idf;
            if (w <= 0.0f) continue;

            const uint8_t* ptr = postings_mmap.data + meta.postings_offset;
            const float k1p1 = k1 + 1.0f;
            const uint32_t df = meta.df;
            if (df < POST_VBYTE_DF) {
                uint32_t current_doc = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    uint32_t coded = read_vbyte(ptr);
                    current_doc += coded >> 1;
                    uint8_t packed = (coded & 1u) ? 1 : *ptr++;
                    uint8_t ttf = packed >> 6;
                    float tf_eff = static_cast<float>(packed & 0x3F) +
                                   w_title * static_cast<float>(ttf);
                    float tf_norm = (tf_eff * k1p1) /
                                    (tf_eff + length_penalty_lut[quantized_doc_lens[current_doc]]);
                    float contrib = w * (tf_norm + delta + (ttf ? title_bonus : 0.0f));
                    if (acc[current_doc] == 0.0f) touched.push_back(current_doc);
                    acc[current_doc] += contrib;
                }
            } else {
                const uint8_t* p_ef = s9_skip(ptr, df);
                uint32_t n_title_hits = 0;
                if (meta.has_title) {
                    n_title_hits = read_vbyte(p_ef);
                    scratch_titles.resize(n_title_hits);
                    uint32_t prev = 0;
                    for (uint32_t h = 0; h < n_title_hits; ++h) {
                        uint32_t coded = read_vbyte(p_ef);
                        prev += coded >> 2;
                        scratch_titles[h] = (prev << 2) | (coded & 3u);
                    }
                }

                S9Reader s9; s9.init(ptr);
                EFReader ef; ef.init(p_ef);
                uint32_t hi = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    const uint32_t tf_raw      = s9.next();
                    const uint32_t current_doc = ef.next();
                    uint8_t ttf = 0;
                    if (hi < n_title_hits && (scratch_titles[hi] >> 2) == i) {
                        ttf = static_cast<uint8_t>(scratch_titles[hi] & 3u);
                        ++hi;
                    }
                    float tf_eff = static_cast<float>(tf_raw + 1u) +
                                   w_title * static_cast<float>(ttf);
                    float tf_norm = (tf_eff * k1p1) /
                                    (tf_eff + length_penalty_lut[quantized_doc_lens[current_doc]]);
                    float contrib = w * (tf_norm + delta + (ttf ? title_bonus : 0.0f));
                    if (acc[current_doc] == 0.0f) touched.push_back(current_doc);
                    acc[current_doc] += contrib;
                }
            }
        }

        size_t want = std::min(static_cast<size_t>(std::max(top_k, 0)), touched.size());
        // nth_element rather than partial_sort: we only need the top `want`
        // as a set (exact order is imposed by the std::sort below).
        if (want < touched.size()) {
            std::nth_element(touched.begin(), touched.begin() + want, touched.end(),
                             [this](uint32_t a, uint32_t x) {
                                 if (acc[a] != acc[x]) return acc[a] > acc[x];
                                 return a < x;
                             });
        }
        std::sort(touched.begin(), touched.begin() + want,
                  [this](uint32_t a, uint32_t x) {
                      if (acc[a] != acc[x]) return acc[a] > acc[x];
                      return doc_ids[a] < doc_ids[x];
                  });

        std::vector<std::pair<uint32_t, float>> results;
        results.reserve(want);
        for (size_t i = 0; i < want; ++i) results.push_back({touched[i], acc[touched[i]]});

        for (uint32_t d : touched) acc[d] = 0.0f;   // reset only what we dirtied
        touched.clear();
        return results;
    }

    // -----------------------------------------------------------------------
    // Term frequencies for the RM3 feedback documents, straight out of the
    // forward index. O(feedback docs) instead of O(vocabulary).
    // -----------------------------------------------------------------------
    void get_batch_doc_term_freqs(const std::vector<uint32_t>& target_docs, BatchFwd& out) {
        const size_t n = target_docs.size();
        out.start.assign(n, 0);
        out.len.assign(n, 0);
        out.arena.clear();
        if (!fwd_mmap.data || !fwdoff_mmap.data || fwdoff_mmap.length < 8) return;
        out.arena.reserve(n * 24);

        const uint32_t* offs = reinterpret_cast<const uint32_t*>(
            fwdoff_mmap.data + 2 * sizeof(uint32_t));
        const uint32_t n_chk = static_cast<uint32_t>(
            (fwdoff_mmap.length - 2 * sizeof(uint32_t)) / sizeof(uint32_t));
        const uint32_t stride = fwd_off_stride ? fwd_off_stride : 1;

        auto skip_record = [](const uint8_t*& p) {
            uint32_t cnt    = read_vbyte(p);
            uint32_t ef_len = read_vbyte(p);
            if (cnt) p = s9_skip(p, cnt);
            p += ef_len;
        };

        fwd_order.resize(n);
        for (size_t i = 0; i < n; ++i) fwd_order[i] = static_cast<uint32_t>(i);
        std::sort(fwd_order.begin(), fwd_order.end(),
                  [&target_docs](uint32_t a, uint32_t b) {
                      return target_docs[a] < target_docs[b];
                  });

        uint32_t cur_chk = UINT32_MAX;
        uint32_t cur_sub = 0;
        const uint8_t* p = nullptr;

        for (uint32_t pos : fwd_order) {
            const uint32_t d = target_docs[pos];
            if (d >= N) continue;
            const uint32_t chk = d / stride, sub = d % stride;
            if (chk >= n_chk) continue;
            if (chk != cur_chk || sub < cur_sub) {
                p = fwd_mmap.data + offs[chk];
                cur_chk = chk;
                cur_sub = 0;
            }
            while (cur_sub < sub) { skip_record(p); ++cur_sub; }

            const uint32_t cnt    = read_vbyte(p);
            const uint32_t ef_len = read_vbyte(p);
            const uint8_t* tfp    = p;
            if (cnt) p = s9_skip(p, cnt);
            const uint8_t* efp    = p;
            p += ef_len;
            ++cur_sub;

            if (!cnt) continue;
            rid_scratch.resize(cnt);
            const uint8_t* efc = efp;
            ef_unpack_docs(efc, rid_scratch.data(), cnt);
            S9Reader s9; s9.init(tfp);

            out.start[pos] = static_cast<uint32_t>(out.arena.size());
            uint32_t kept = 0;
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t rid = rid_scratch[i];
                const uint32_t tf  = s9.next();
                if (rid >= rm3_to_tid.size()) continue;
                out.arena.push_back({rm3_to_tid[rid], tf});
                ++kept;
            }
            out.len[pos] = kept;
        }
    }

    // =======================================================================
    // ENTRY POINT
    // =======================================================================
    std::vector<std::pair<std::string, float>> retrieve(
        const std::string& query,
        int k,
        float k1 = 1.2f,
        float b = 0.75f,
        float delta = 1.0f,
        float w_title = 0.0f,
        float title_bonus = 0.0f,
        bool use_rm3 = true,
        int fb_docs = 10,
        int fb_terms = 25,
        float alpha = 0.6f,
        float mu = 1000.0f,
        int rm3_pool = RM3_POOL,
        float fb_temp = 20.0f,
        int fusion_depth = 100,
        float rrf_c = 60.0f,
        float rrf_w = 0.4f,
        bool use_fusion = false,
        int fusion_kind = 1)
    {
        auto tokens = cpp_tokenize(query, tokcfg);
        if (tokens.empty() || N == 0) return {};

        std::unordered_map<int32_t, float> qw;
        for (const auto& tp : tokens) {
            int32_t tid = find_term(tp.first);
            if (tid >= 0) qw[tid] += 1.0f;
        }
        if (qw.empty()) return {};
        float qsum = 0.0f;
        for (const auto& kv : qw) qsum += kv.second;
        if (qsum <= 0.0f) return {};
        for (auto& kv : qw) kv.second /= qsum;

        std::vector<std::pair<int32_t, float>> q_vec(qw.begin(), qw.end());

        int pass1_depth = std::max(std::max(fb_docs, k), use_rm3 ? std::max(rm3_pool, k) : k);
        auto pass1 = raw_retrieve(q_vec, pass1_depth, k1, b, delta, w_title, title_bonus);
        if (pass1.empty()) return {};

        std::vector<std::pair<uint32_t, float>> pool;

        if (!use_rm3 || pass1.size() < 2) {
            pool = pass1;
        } else {
            int num_fb = std::min(static_cast<int>(pass1.size()), fb_docs);

            float mean = 0.0f;
            for (int i = 0; i < num_fb; ++i) mean += pass1[i].second;
            mean /= num_fb;
            float var = 0.0f;
            for (int i = 0; i < num_fb; ++i) {
                float d = pass1[i].second - mean;
                var += d * d;
            }
            var /= num_fb;

            if (var < 1e-6f) {
                pool = pass1;
            } else {
                float norm = 0.0f;
                for (int i = 0; i < num_fb; ++i) norm += pass1[i].second;
                if (norm <= 0.0f) norm = 1.0f;
                if (fb_temp < 19.5f) {
                    const float sd = std::sqrt(var);
                    const float tau = std::max(sd * std::max(fb_temp, 0.05f), 1e-6f);
                    const float smax = pass1[0].second;
                    fb_wts.resize(num_fb);
                    norm = 0.0f;
                    for (int i = 0; i < num_fb; ++i) {
                        fb_wts[i] = std::exp((pass1[i].second - smax) / tau);
                        norm += fb_wts[i];
                    }
                    if (norm <= 0.0f) norm = 1.0f;
                    for (int i = 0; i < num_fb; ++i) fb_wts[i] /= norm;
                } else {
                    fb_wts.resize(num_fb);
                    for (int i = 0; i < num_fb; ++i) fb_wts[i] = pass1[i].second / norm;
                }

                std::vector<uint32_t> pool_ids;
                pool_ids.reserve(pass1.size());
                for (const auto& p : pass1) pool_ids.push_back(p.first);
                BatchFwd batch;
                get_batch_doc_term_freqs(pool_ids, batch);

                std::unordered_map<int32_t, float> p_w_r;
                std::unordered_map<int32_t, float> prf2_raw;
                for (int i = 0; i < num_fb; ++i) {
                    uint32_t doc_id = pass1[i].first;
                    float p_d_q = fb_wts[i];
                    float d_len = static_cast<float>(doc_lens[doc_id]);
                    const float denom = d_len + mu;

                    const uint32_t s0 = batch.start[i], sn = batch.len[i];
                    for (uint32_t j = 0; j < sn; ++j) {
                        const auto& tf_kv = batch.arena[s0 + j];
                        int32_t tid = static_cast<int32_t>(tf_kv.first);
                        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
                        const float tf = static_cast<float>(tf_kv.second);
                        float p_w_c = (total_corpus_tokens > 0)
                            ? (static_cast<float>(term_metas[tid].cf) / total_corpus_tokens)
                            : 1e-4f;
                        // Present-term Dirichlet (standard RM3). The absent-term
                        // μ P(w|C) mass promotes high-cf noise when we then keep
                        // only fb_terms; PRF2 fusion is the complementary arm.
                        p_w_r[tid] += p_d_q * (tf + mu * p_w_c) / denom;
                        prf2_raw[tid] += std::log(1.0f + (static_cast<float>(N) - term_metas[tid].df + 0.5f)
                                                  / (term_metas[tid].df + 0.5f));
                    }
                }

                auto top_expand = [&](std::unordered_map<int32_t, float>& raw)
                    -> std::unordered_map<uint32_t, float> {
                    std::vector<std::pair<int32_t, float>> expansion(raw.begin(), raw.end());
                    size_t keep = std::min(static_cast<size_t>(std::max(fb_terms, 0)), expansion.size());
                    if (keep == 0) return {};
                    std::partial_sort(expansion.begin(), expansion.begin() + keep, expansion.end(),
                                      [](const auto& a, const auto& b) { return a.second > b.second; });
                    expansion.resize(keep);
                    float esum = 0.0f;
                    for (const auto& e : expansion) esum += e.second;
                    if (esum <= 0.0f) esum = 1.0f;
                    std::unordered_map<uint32_t, float> exp_w;
                    exp_w.reserve(expansion.size() * 2);
                    for (const auto& e : expansion) {
                        int32_t tid = e.first;
                        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
                        float idf = std::log(1.0f + (static_cast<float>(N) - term_metas[tid].df + 0.5f)
                                             / (term_metas[tid].df + 0.5f));
                        exp_w[static_cast<uint32_t>(tid)] = (1.0f - alpha) * (e.second / esum) * idf;
                    }
                    return exp_w;
                };

                auto exp_rm3 = top_expand(p_w_r);
                auto exp_prf2 = use_fusion ? top_expand(prf2_raw) : std::unordered_map<uint32_t, float>{};

                auto merge_exp = [&]() {
                    auto out = exp_rm3;
                    for (const auto& kv : exp_prf2) {
                        out[kv.first] += kv.second;
                    }
                    return out;
                };

                float length_penalty_lut[256];
                for (int q = 0; q < 256; ++q) {
                    float ratio = dequantize_doc_len(static_cast<uint8_t>(q));
                    length_penalty_lut[q] = k1 * (1.0f - b + b * ratio);
                }
                const float k1p1 = k1 + 1.0f;
                const size_t n = pass1.size();

                auto score_arm = [&](const std::unordered_map<uint32_t, float>& exp_w,
                                     std::vector<float>& out) {
                    out.assign(n, 0.0f);
                    for (size_t pi = 0; pi < n; ++pi) {
                        uint32_t d = pass1[pi].first;
                        float s = alpha * pass1[pi].second;
                        const uint32_t s0 = batch.start[pi], sn = batch.len[pi];
                        if (sn && !exp_w.empty()) {
                            float lp = length_penalty_lut[quantized_doc_lens[d]];
                            for (uint32_t j = 0; j < sn; ++j) {
                                const auto& tf_kv = batch.arena[s0 + j];
                                auto e = exp_w.find(tf_kv.first);
                                if (e == exp_w.end()) continue;
                                float tf_eff = static_cast<float>(tf_kv.second);
                                float tf_norm = (tf_eff * k1p1) / (tf_eff + lp);
                                s += e->second * (tf_norm + delta);
                            }
                        }
                        out[pi] = s;
                    }
                };

                std::vector<float> s_rm3, s_prf2;
                score_arm(exp_rm3, s_rm3);

                auto better = [&](size_t a, size_t b, const std::vector<float>& sc) {
                    if (sc[a] != sc[b]) return sc[a] > sc[b];
                    return doc_ids[pass1[a].first] < doc_ids[pass1[b].first];
                };

                auto fill_rank = [&](const std::vector<float>& sc, int depth,
                                     std::vector<int>& rank) {
                    std::vector<size_t> ord(n);
                    for (size_t i = 0; i < n; ++i) ord[i] = i;
                    std::sort(ord.begin(), ord.end(),
                              [&](size_t a, size_t b) { return better(a, b, sc); });
                    rank.assign(n, depth + 1);
                    const int lim = std::min(depth, static_cast<int>(n));
                    for (int r = 0; r < lim; ++r)
                        rank[ord[static_cast<size_t>(r)]] = r + 1;
                };

                if (!use_fusion || fusion_kind == 0) {
                    std::vector<size_t> ord(n);
                    for (size_t i = 0; i < n; ++i) ord[i] = i;
                    std::sort(ord.begin(), ord.end(),
                              [&](size_t a, size_t b) { return better(a, b, s_rm3); });
                    pool.clear();
                    const size_t want_n = std::min(n, static_cast<size_t>(std::max(k, 0)));
                    pool.reserve(want_n);
                    for (size_t i = 0; i < want_n; ++i)
                        pool.push_back({pass1[ord[i]].first, s_rm3[ord[i]]});
                } else if (fusion_kind == 2) {
                    std::vector<float> s_union;
                    score_arm(merge_exp(), s_union);
                    std::vector<size_t> ord(n);
                    for (size_t i = 0; i < n; ++i) ord[i] = i;
                    std::sort(ord.begin(), ord.end(),
                              [&](size_t a, size_t b) { return better(a, b, s_union); });
                    pool.clear();
                    const size_t want_n = std::min(n, static_cast<size_t>(std::max(k, 0)));
                    pool.reserve(want_n);
                    for (size_t i = 0; i < want_n; ++i)
                        pool.push_back({pass1[ord[i]].first, s_union[ord[i]]});
                } else if (fusion_kind == 3) {
                    score_arm(exp_prf2, s_prf2);
                    float mn1 = s_rm3[0], mx1 = s_rm3[0], mn2 = s_prf2[0], mx2 = s_prf2[0];
                    for (size_t i = 1; i < n; ++i) {
                        mn1 = std::min(mn1, s_rm3[i]); mx1 = std::max(mx1, s_rm3[i]);
                        mn2 = std::min(mn2, s_prf2[i]); mx2 = std::max(mx2, s_prf2[i]);
                    }
                    const float r1 = std::max(mx1 - mn1, 1e-6f);
                    const float r2 = std::max(mx2 - mn2, 1e-6f);
                    pool.clear();
                    pool.reserve(n);
                    for (size_t i = 0; i < n; ++i) {
                        float a = (s_rm3[i] - mn1) / r1;
                        float b = (s_prf2[i] - mn2) / r2;
                        pool.push_back({pass1[i].first, rrf_w * a + (1.0f - rrf_w) * b});
                    }
                    std::sort(pool.begin(), pool.end(),
                              [this](const auto& a, const auto& b) {
                                  if (a.second != b.second) return a.second > b.second;
                                  return doc_ids[a.first] < doc_ids[b.first];
                              });
                    if (static_cast<int>(pool.size()) > k)
                        pool.resize(static_cast<size_t>(std::max(k, 0)));
                } else {
                    score_arm(exp_prf2, s_prf2);
                    const int depth = std::max(fusion_depth, k);
                    std::vector<int> rank_rm3, rank_prf2;
                    fill_rank(s_rm3, depth, rank_rm3);
                    fill_rank(s_prf2, depth, rank_prf2);
                    pool.clear();
                    pool.reserve(n);
                    const float w2 = 1.0f - rrf_w;
                    const float c = std::max(rrf_c, 1.0f);
                    for (size_t i = 0; i < n; ++i) {
                        float s = 0.0f;
                        if (rank_rm3[i] <= depth)
                            s += rrf_w / (c + static_cast<float>(rank_rm3[i]));
                        if (rank_prf2[i] <= depth)
                            s += w2 / (c + static_cast<float>(rank_prf2[i]));
                        if (s > 0.0f) pool.push_back({pass1[i].first, s});
                    }
                    std::sort(pool.begin(), pool.end(),
                              [this](const auto& a, const auto& b) {
                                  if (a.second != b.second) return a.second > b.second;
                                  return doc_ids[a.first] < doc_ids[b.first];
                              });
                    if (static_cast<int>(pool.size()) > k)
                        pool.resize(static_cast<size_t>(std::max(k, 0)));
                }
            }
        }

        std::vector<std::pair<std::string, float>> results;
        size_t want = std::min(static_cast<size_t>(std::max(k, 0)), pool.size());
        results.reserve(want);
        for (size_t i = 0; i < want; ++i) results.push_back({doc_ids[pool[i].first], pool[i].second});
        return results;
    }

    // -----------------------------------------------------------------------
    // First-pass scoring only: no relevance feedback. Needed for the report's
    // ablation table and to sweep k1/b in isolation from RM3.
    // -----------------------------------------------------------------------
    std::vector<std::pair<std::string, float>> retrieve_bm25(
        const std::string& query, int k, float k1, float b, float delta,
        float w_title, float title_bonus)
    {
        return retrieve(query, k, k1, b, delta, w_title, title_bonus,
                        false, 0, 0, 1.0f, 1000.0f, RM3_POOL, 20.0f);
    }
};

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "From-scratch inverted index with BM25F+ scoring and RM3 query expansion";
    m.def("check_codecs", &check_codecs,
          "S9Reader/EFReader/s9_skip must match simple9_unpack/ef_unpack_docs");
    py::class_<CPPEngine>(m, "CPPEngine")
        .def(py::init<>())
        .def("build_index", &CPPEngine::build_index, py::arg("corpus"), py::arg("index_dir"))
        .def("build_index_from_file", &CPPEngine::build_index_from_file,
             py::arg("corpus_path"), py::arg("index_dir"))
        .def("load_index", &CPPEngine::load_index, py::arg("index_dir"))
        .def("retrieve", &CPPEngine::retrieve,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("w_title") = 0.0f, py::arg("title_bonus") = 0.0f,
             py::arg("use_rm3") = true, py::arg("fb_docs") = 10, py::arg("fb_terms") = 25,
             py::arg("alpha") = 0.6f, py::arg("mu") = 1000.0f,
             py::arg("rm3_pool") = RM3_POOL, py::arg("fb_temp") = 20.0f,
             py::arg("fusion_depth") = 100, py::arg("rrf_c") = 60.0f,
             py::arg("rrf_w") = 0.4f, py::arg("use_fusion") = false,
             py::arg("fusion_kind") = 1)
        .def("retrieve_bm25", &CPPEngine::retrieve_bm25,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("w_title") = 0.0f, py::arg("title_bonus") = 0.0f);
}