// Shared constants and on-disk types (no allocations).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

constexpr uint32_t FWD_CANDIDATES   = 64;   // prune width before per-doc keep
constexpr int      RM3_POOL         = 250;  // abstract collections: rerank this many
constexpr uint8_t  INDEX_VERSION    = 14;  // bump on postings layout OR analysis change
constexpr uint32_t TF_MAX           = 63;   // 6 bits in pack_tf
constexpr uint32_t TITLE_TF_MAX     = 3;    // 2 bits in pack_tf
constexpr uint32_t POST_VBYTE_DF    = 8;    // df below this: always vbyte
constexpr uint8_t  CODEC_S9EF       = 0;    // Simple-9 tfs + optional title + EF docs
constexpr uint8_t  CODEC_VBYTE      = 1;    // gap+tf vbyte (LSB flags tf==1)
constexpr uint8_t  CODEC_S9EF_UNIT  = 2;    // all body tf==1: no Simple-9 payload
// Grading VM is 4 cores. Never request more, even if hardware_concurrency
// reports SMT siblings or IR_BUILD_WORKERS is set above this.
constexpr int      IR_MAX_THREADS   = 4;
// Fine log (|d|/avgdl) in one byte. Scale 64 matches ~1% relative steps
// on typical docs (CORD/FiQA). Scale 32 reached ~250× but coarsened the
// bulk of the range and dropped local nDCG. Code 255 is reserved for
constexpr float    LEN_Q_SCALE      = 64.0f;
constexpr float    LEN_Q_SATURATED  = 64.0f;

struct BuildCfg {
    uint32_t title_window   = 12;     // leading tokens treated as "title"
    uint32_t fwd_terms      = 20;     // CSR terms kept per doc (scaled by length)
    float    rm3_df_frac    = 0.13f;  // drop terms with df > this fraction of N
    uint32_t min_df         = 1;
};

inline BuildCfg read_build_cfg() {
    BuildCfg c;
    if (const char* e = std::getenv("IR_TITLE_WINDOW")) c.title_window = static_cast<uint32_t>(std::atoi(e));
    if (const char* e = std::getenv("IR_FWD_TERMS"))    c.fwd_terms    = static_cast<uint32_t>(std::atoi(e));
    if (const char* e = std::getenv("IR_RM3_DF_FRAC"))  c.rm3_df_frac  = static_cast<float>(std::atof(e));
    if (const char* e = std::getenv("IR_MIN_DF"))       c.min_df       = static_cast<uint32_t>(std::atoi(e));
    return c;
}

inline int build_workers() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    int n = (hw > 0) ? std::min(hw, IR_MAX_THREADS) : IR_MAX_THREADS;
    if (const char* e = std::getenv("IR_BUILD_WORKERS")) n = std::atoi(e);
    if (n < 1) n = 1;
    if (n > IR_MAX_THREADS) n = IR_MAX_THREADS;
    return n;
}

// Cap threads on large collections so mmap + per-worker maps stay under 8 GB.
inline int build_workers_for_docs(uint32_t n_docs) {
    int n = build_workers();
    if (n_docs >= 800000u) n = 1;
    else if (n_docs >= 400000u) n = std::min(n, 2);
    return n;
}

// W workers, caller is worker 0. Spawns at most IR_MAX_THREADS-1 extra
// threads so a 4-core box is never oversubscribed.
template <typename Fn>
inline void run_parallel(int W, Fn&& fn) {
    if (W > IR_MAX_THREADS) W = IR_MAX_THREADS;
    if (W <= 1) { fn(0); return; }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(W - 1));
    for (int w = 1; w < W; ++w) pool.emplace_back(fn, w);
    fn(0);
    for (auto& t : pool) t.join();
}

// Store |d|/avgdl as 8-bit log2 so doclens.bin stays small; BM25 only needs ~%.
inline uint8_t quantize_doc_len(float len_ratio) {
    if (len_ratio <= 0.0f) return 0;
    float val = LEN_Q_SCALE * std::log2(1.0f + len_ratio);
    if (val >= 255.0f) return 255;
    return static_cast<uint8_t>(val);
}

inline float dequantize_doc_len(uint8_t q) {
    if (q == 255) return LEN_Q_SATURATED;
    return std::pow(2.0f, static_cast<float>(q) / LEN_Q_SCALE) - 1.0f;
}

struct TermMeta {
    uint32_t df = 0;
    uint8_t  has_title = 0;
    uint8_t  codec = CODEC_S9EF;  // on-disk: packed with df (not a separate field)
    uint32_t s9_bytes = 0;      // in-memory: S9 payload length (0 if none)
    uint64_t cf = 0;
    uint64_t postings_offset = 0;
    float    idf = 0.0f;        // in-memory: RSJ idf, filled at load
};

#if defined(__GNUC__) || defined(__clang__)
#  define IR_PREFETCH(p) __builtin_prefetch((p), 1, 1)
#else
#  define IR_PREFETCH(p) ((void)0)
#endif

struct DocPosting {
    uint32_t doc_id;
    uint16_t tf;        // clamped to TF_MAX (63) at merge
    uint16_t title_tf;  // clamped to TITLE_TF_MAX (3) at merge
};

inline bool build_timing_enabled() {
    const char* e = std::getenv("IR_BUILD_TIMING");
    return e && e[0] == '1';
}
