#include "codecs.h"
#include <random>
#include <algorithm>

// Greedy: pick the smallest selector that still holds the next k values.
void simple9_pack(std::vector<uint8_t>& out, const uint32_t* vals, uint32_t n) {
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

void simple9_unpack(const uint8_t*& ptr, uint32_t* vals, uint32_t n) {
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

void bit_put(std::vector<uint8_t>& out, uint64_t& acc, int& filled,
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

void bit_flush(std::vector<uint8_t>& out, uint64_t& acc, int& filled) {
    if (filled > 0) {
        out.push_back(static_cast<uint8_t>(acc));
        acc = 0;
        filled = 0;
    }
}

uint32_t bit_get(const uint8_t*& p, uint64_t& acc, int& filled, int bits) {
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
void ef_pack(std::vector<uint8_t>& out, const uint32_t* docs, uint32_t n) {
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

void ef_unpack_docs(const uint8_t*& ptr, uint32_t* docs, uint32_t n) {
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


void check_one_list(const uint32_t* tfs, const uint32_t* docs, uint32_t n, int trial) {
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

void check_codecs() {
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
