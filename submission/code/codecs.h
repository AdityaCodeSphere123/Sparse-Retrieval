// Postings codecs used by engine.h.
//
// pack_tf: bits 0–5 body tf (1..63), bits 6–7 title tf (0..3).
// Simple-9: 4-byte words, top 4 bits = selector, rest packed integers.
// Elias-Fano: increasing doc ids (unary high bits + l low bits).
// vbyte: 7-bit payload, MSB=continue.
#pragma once

#include "ir_config.h"
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>

inline uint8_t pack_tf(uint32_t body_tf, uint32_t title_tf) {
    uint32_t b = std::min(body_tf, TF_MAX);
    uint32_t t = std::min(title_tf, TITLE_TF_MAX);
    return static_cast<uint8_t>(b | (t << 6));
}

// Selector s packs S9_N[s] values of S9_B[s] bits each (fits in 28 payload bits).
constexpr uint8_t S9_N[9] = {28, 14, 9, 7, 5, 4, 3, 2, 1};
constexpr uint8_t S9_B[9] = { 1,  2, 3, 4, 5, 7, 9,14,28};

void simple9_pack(std::vector<uint8_t>& out, const uint32_t* vals, uint32_t n);
void simple9_unpack(const uint8_t*& ptr, uint32_t* vals, uint32_t n);
void ef_pack(std::vector<uint8_t>& out, const uint32_t* docs, uint32_t n);
void ef_unpack_docs(const uint8_t*& ptr, uint32_t* docs, uint32_t n);
void check_codecs();

// Sequential Elias-Fano: unary run of zeros then a 1 for each high-bit delta.
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
            std::memcpy(&word, p, 4);
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

// Advance past n Simple-9 values without unpacking (selector is in byte 3).
inline const uint8_t* s9_skip(const uint8_t* p, uint32_t n) {
    uint32_t i = 0;
    while (i < n) { i += S9_N[static_cast<uint32_t>(p[3]) >> 4]; p += 4; }
    return p;
}

inline void write_vbyte(std::vector<uint8_t>& out, uint32_t val) {
    while (val >= 128) {
        out.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val & 0x7F));
}

inline void write_vbyte64(std::vector<uint8_t>& out, uint64_t val) {
    while (val >= 128) {
        out.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val & 0x7F));
}

inline uint32_t read_vbyte(const uint8_t*& ptr) {
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

inline uint64_t read_vbyte64(const uint8_t*& ptr) {
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
