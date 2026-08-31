// Build/query constants, index version, and the small types shared by every
// translation unit. Nothing here allocates.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include <thread>

constexpr uint32_t FWD_CANDIDATES   = 64;
constexpr uint32_t FWD_OFF_STRIDE   = 8;
constexpr int      RM3_POOL         = 250;
constexpr uint8_t  INDEX_VERSION    = 10;
constexpr uint32_t TF_MAX           = 63;
constexpr uint32_t TITLE_TF_MAX     = 3;
constexpr uint32_t POST_VBYTE_DF    = 8;

struct BuildCfg {
    uint32_t title_window   = 12;
    uint32_t fwd_terms      = 20;
    float    rm3_df_frac    = 0.13f;
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
    int n = (hw > 0) ? std::min(hw, 4) : 4;
    if (const char* e = std::getenv("IR_BUILD_WORKERS")) n = std::atoi(e);
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    return n;
}

// Cap threads on large collections so mmap + per-worker maps stay under 8 GB.
inline int build_workers_for_docs(uint32_t n_docs) {
    int n = build_workers();
    if (n_docs >= 800000u) n = 1;
    else if (n_docs >= 400000u) n = std::min(n, 2);
    return n;
}

inline uint8_t quantize_doc_len(float len_ratio) {
    if (len_ratio <= 0.0f) return 0;
    float val = 64.0f * std::log2(1.0f + len_ratio);
    if (val >= 255.0f) return 255;
    return static_cast<uint8_t>(val);
}

inline float dequantize_doc_len(uint8_t q) {
    return std::pow(2.0f, static_cast<float>(q) / 64.0f) - 1.0f;
}

struct TermMeta {
    uint32_t df = 0;
    uint8_t  has_title = 0;
    uint64_t cf = 0;
    uint64_t postings_offset = 0;
};

struct DocPosting {
    uint32_t doc_id;
    uint16_t tf;        // clamped to TF_MAX (63) at merge
    uint16_t title_tf;  // clamped to TITLE_TF_MAX (3) at merge
};

inline bool build_timing_enabled() {
    const char* e = std::getenv("IR_BUILD_TIMING");
    return e && e[0] == '1';
}
