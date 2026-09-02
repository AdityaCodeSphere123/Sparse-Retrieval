#pragma once
// Inverted index: SPIMI build, mmap load, BM25+ term-at-a-time, optional RM3.

#include "ir_config.h"
#include "codecs.h"
#include "analysis.h"
#include "corpus_io.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <exception>
#include <string_view>
#include <cstddef>
#include <atomic>
#include <chrono>

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

    // ---- dictionary (sorted arrays; term id = index into these) -------------
    std::vector<std::string> terms_sorted;
    std::vector<TermMeta> term_metas;

    MmapBuffer postings_mmap;

    // RM3 forward index, rebuilt at load from postings (not persisted).
    std::vector<uint32_t> csr_off;   // N+1
    std::vector<uint32_t> csr_tid;
    std::vector<uint8_t>  csr_tf;
    uint32_t fwd_keep = 20;

    struct BatchFwd {
        std::vector<std::pair<uint32_t, uint32_t>> arena;
        std::vector<uint32_t> start;
        std::vector<uint32_t> len;
    };

    // ---- query-time scratch -------------------------------------------------
    // acc[d] is live iff acc_stamp[d] == score_gen. Bump score_gen instead of
    // zeroing the whole acc array after each query.
    std::vector<float> acc;
    std::vector<uint32_t> acc_stamp;
    uint32_t score_gen = 1;
    std::vector<uint32_t> touched;
    std::vector<uint32_t> scratch_gaps;
    std::vector<uint32_t> scratch_tfs;
    std::vector<uint32_t> scratch_titles;
    std::vector<float> fb_wts;
    BatchFwd fwd_batch;
    float deq_lut[256]{};
    float lp_lut[256]{};
    float lp_k1 = -1.0f;
    float lp_b = -1.0f;

    void fill_deq_lut() {
        deq_lut[255] = LEN_Q_SATURATED;
        for (int i = 0; i < 255; ++i)
            deq_lut[i] = std::pow(2.0f, static_cast<float>(i) / LEN_Q_SCALE) - 1.0f;
    }

    void ensure_lp(float k1, float b) {
        if (k1 == lp_k1 && b == lp_b) return;
        lp_k1 = k1;
        lp_b = b;
        for (int q = 0; q < 256; ++q)
            lp_lut[q] = k1 * (1.0f - b + b * deq_lut[q]);
    }

    CPPEngine() = default;

    int32_t find_term(const std::string& t) const {
        auto it = std::lower_bound(terms_sorted.begin(), terms_sorted.end(), t);
        if (it == terms_sorted.end() || *it != t) return -1;
        return static_cast<int32_t>(it - terms_sorted.begin());
    }

    // Build
    // Parallel SPIMI shards → k-way merge into postings.bin + front-coded
    // dict.bin. The RM3 per-document CSR is not written; load_index rebuilds it.
    template <typename Supplier>
    void build_common(Supplier&& next_doc, const std::string& index_dir,
                      const std::vector<std::pair<const char*, size_t>>* line_spans = nullptr,
                      MmapBuffer* release_corpus = nullptr) {
        postings_mmap.unmap();
        csr_off.clear();
        csr_tid.clear();
        csr_tf.clear();

        doc_ids.clear();
        doc_lens.clear();
        total_corpus_tokens = 0;
        N = 0;

        // Stemmer / stop policy are frozen in meta.bin. Override with
        // IR_STEMMER=0|1|2 and IR_STOPWORDS=0|1 (no recompile).
        buildcfg = read_build_cfg();
        tokcfg = TokenizerCfg{};
        if (const char* e = std::getenv("IR_STEMMER"))
            tokcfg.stemmer = static_cast<uint8_t>(std::atoi(e));
        if (const char* e = std::getenv("IR_STOPWORDS"))
            tokcfg.remove_stop = static_cast<uint8_t>(std::atoi(e));

        const bool time_build = build_timing_enabled();
        auto t0 = std::chrono::steady_clock::now();
        auto stamp = [&](const char* label, std::chrono::steady_clock::time_point& mark) {
            if (!time_build) return;
            auto now = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(now - mark).count();
            std::cerr << "[build] " << label << ": " << s << "s\n";
            mark = now;
        };

        std::vector<std::string> all_ids;
        std::vector<std::string> all_texts;
        if (!line_spans) {
            std::string ext_id, text;
            while (next_doc(ext_id, text)) {
                all_ids.push_back(std::move(ext_id));
                all_texts.push_back(std::move(text));
            }
        }
        stamp("load_docs", t0);
        const uint32_t Ndocs = line_spans
            ? static_cast<uint32_t>(line_spans->size())
            : static_cast<uint32_t>(all_ids.size());
        int W = build_workers_for_docs(Ndocs);
        if (Ndocs == 0) W = 1;
        else if (static_cast<uint32_t>(W) > Ndocs) W = static_cast<int>(Ndocs);
        if (W > IR_MAX_THREADS) W = IR_MAX_THREADS;

        struct Shard {
            std::unordered_map<std::string, std::vector<DocPosting>> postings;
            std::vector<std::string> ids;
            uint64_t tokens = 0;
            std::vector<uint32_t> doc_lens;
            std::exception_ptr err;
        };
        std::vector<Shard> shards(static_cast<size_t>(W));

        auto ingest_range = [&](int w) {
            try {
                const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * Ndocs) / static_cast<uint32_t>(W));
                const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(w + 1) * Ndocs) / static_cast<uint32_t>(W));
                Shard& sh = shards[static_cast<size_t>(w)];
                sh.doc_lens.reserve(hi - lo);
                if (line_spans) sh.ids.reserve(hi - lo);

                thread_local std::string tls_id, tls_text;
                thread_local std::unordered_map<std::string, uint32_t> tls_term_data;
                thread_local std::unordered_map<std::string, uint32_t> tls_title_tf;
                // Reuse per-doc maps across documents (same thread). Reserve
                // the shard map once so ingest does not rehash on every term.
                sh.postings.reserve(static_cast<size_t>(hi - lo) * 2u + 1024u);
                for (uint32_t doc_int_id = lo; doc_int_id < hi; ++doc_int_id) {
                    std::string_view text_sv;
                    if (line_spans) {
                        const auto& sp = (*line_spans)[doc_int_id];
                        if (!parse_jsonl_sv(std::string_view(sp.first, sp.second), tls_id, tls_text)) {
                            tls_id.clear();
                            tls_text.clear();
                        }
                        sh.ids.push_back(tls_id);
                        text_sv = tls_text;
                    } else {
                        text_sv = all_texts[doc_int_id];
                    }
                    tls_term_data.clear();
                    tls_title_tf.clear();
                    uint32_t d_len = 0;
                    accumulate_tf(text_sv, tokcfg, buildcfg.title_window,
                                  tls_term_data, tls_title_tf, d_len);
                    sh.doc_lens.push_back(d_len);
                    sh.tokens += d_len;

                    for (const auto& kv : tls_term_data) {
                        auto ttf = tls_title_tf.find(kv.first);
                        sh.postings[kv.first].push_back(
                            DocPosting{doc_int_id,
                                       static_cast<uint16_t>(std::min(kv.second, 65535u)),
                                       static_cast<uint16_t>(
                                           (ttf == tls_title_tf.end()) ? 0u
                                                                      : std::min(ttf->second, 65535u))});
                    }
                }
            } catch (...) {
                shards[static_cast<size_t>(w)].err = std::current_exception();
            }
        };

        run_parallel(W, ingest_range);
        for (auto& sh : shards) {
            if (sh.err) std::rethrow_exception(sh.err);
        }

        stamp("ingest", t0);
        if (release_corpus) release_corpus->unmap();

        if (line_spans) {
            doc_ids.clear();
            doc_ids.reserve(Ndocs);
            for (int w = 0; w < W; ++w)
                for (auto& id : shards[static_cast<size_t>(w)].ids)
                    doc_ids.push_back(std::move(id));
        } else {
            doc_ids = std::move(all_ids);
        }
        all_texts.clear();
        all_texts.shrink_to_fit();

        doc_lens.resize(Ndocs);
        total_corpus_tokens = 0;
        for (int w = 0; w < W; ++w) {
            const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * Ndocs) / static_cast<uint32_t>(W));
            std::copy(shards[static_cast<size_t>(w)].doc_lens.begin(),
                      shards[static_cast<size_t>(w)].doc_lens.end(),
                      doc_lens.begin() + lo);
            total_corpus_tokens += shards[static_cast<size_t>(w)].tokens;
        }

        using TermRun = std::vector<std::pair<std::string, std::vector<DocPosting>>>;
        std::vector<TermRun> mem_runs(static_cast<size_t>(W));
        auto sort_shard = [&](int w) {
            auto& src = shards[static_cast<size_t>(w)].postings;
            auto& dst = mem_runs[static_cast<size_t>(w)];
            dst.reserve(src.size());
            for (auto& kv : src) dst.emplace_back(std::move(kv.first), std::move(kv.second));
            src.clear();
            std::sort(dst.begin(), dst.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        };
        run_parallel(W, sort_shard);
        stamp("sort_runs", t0);

        N = Ndocs;
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
            uint32_t fk = buildcfg.fwd_terms;
            f.write(reinterpret_cast<const char*>(&fk), sizeof(fk));
            f.write(reinterpret_cast<const char*>(&buildcfg.rm3_df_frac), sizeof(float));
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

            bool packed = false;
            std::string alphabet;
            int bits = 0;
            if (id_len > 0 && id_len <= 24) {
                bool seen[256] = {};
                for (const auto& id : doc_ids)
                    for (unsigned char c : id) seen[c] = true;
                for (int c = 0; c < 256; ++c)
                    if (seen[c]) alphabet.push_back(static_cast<char>(c));
                const uint64_t A = alphabet.size();
                if (A >= 2 && A <= 64) {
                    uint64_t span = 1;
                    bool overflow = false;
                    for (uint16_t i = 0; i < id_len; ++i) {
                        if (span > (~0ull) / A) { overflow = true; break; }
                        span *= A;
                    }
                    if (!overflow && span > 0) {
                        uint64_t maxc = span - 1;
                        bits = 0;
                        for (uint64_t m = maxc; m; m >>= 1) ++bits;
                        const size_t packed_bytes = alphabet.size() + 2 +
                            (static_cast<size_t>(bits) * N + 7) / 8;
                        const size_t raw_bytes = static_cast<size_t>(id_len) * N;
                        if (bits > 0 && bits <= 64 && packed_bytes < raw_bytes)
                            packed = true;
                    }
                }
            }

            if (packed) {
                uint16_t flag = static_cast<uint16_t>(id_len) | 0x8000u;
                f.write(reinterpret_cast<const char*>(&flag), sizeof(flag));
                uint8_t asz = static_cast<uint8_t>(alphabet.size());
                uint8_t bw = static_cast<uint8_t>(bits);
                f.write(reinterpret_cast<const char*>(&asz), 1);
                f.write(alphabet.data(), asz);
                f.write(reinterpret_cast<const char*>(&bw), 1);
                int rank[256];
                for (int c = 0; c < 256; ++c) rank[c] = -1;
                for (size_t i = 0; i < alphabet.size(); ++i)
                    rank[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
                const uint64_t A = alphabet.size();
                std::vector<uint8_t> body;
                body.reserve((static_cast<size_t>(bits) * N + 7) / 8 + 8);
                uint64_t buf = 0;
                int nbuf = 0;
                auto emit = [&](uint64_t v, int w) {
                    while (w > 0) {
                        int take = std::min(w, 64 - nbuf);
                        uint64_t mask = (take == 64) ? ~0ull : ((1ull << take) - 1ull);
                        buf |= (v & mask) << nbuf;
                        nbuf += take;
                        v >>= take;
                        w -= take;
                        while (nbuf >= 8) {
                            body.push_back(static_cast<uint8_t>(buf));
                            buf >>= 8;
                            nbuf -= 8;
                        }
                    }
                };
                for (const auto& id : doc_ids) {
                    uint64_t code = 0;
                    for (unsigned char c : id)
                        code = code * A + static_cast<uint64_t>(rank[c]);
                    emit(code, bits);
                }
                if (nbuf) body.push_back(static_cast<uint8_t>(buf));
                f.write(reinterpret_cast<const char*>(body.data()),
                        static_cast<std::streamsize>(body.size()));
            } else {
                f.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
                if (id_len > 0) {
                    for (const auto& id : doc_ids) f.write(id.data(), id_len);
                } else {
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
        }
        {
            std::ofstream f(index_dir + "/doclens.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            f.write(reinterpret_cast<const char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.write(reinterpret_cast<const char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            f.write(reinterpret_cast<const char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));
        }

        // -------- stage 2: k-way merge of in-memory shard runs -------------
        // Split the vocabulary into W contiguous ranges, merge+encode each
        // range on its own thread, then concatenate. Dict front-coding stays
        // serial so W=1 and W=4 write identical dict.bin / postings.bin.
        struct MemCursor {
            TermRun* lists = nullptr;
            size_t i = 0;
            bool active() const { return lists && i < lists->size(); }
            const std::string& term() const { return (*lists)[i].first; }
            std::vector<DocPosting>& posts() { return (*lists)[i].second; }
            uint32_t df() const { return static_cast<uint32_t>((*lists)[i].second.size()); }
            void next() { ++i; }
        };

        auto encode_list = [](const std::vector<DocPosting>& merged,
                              std::vector<uint8_t>& block_data,
                              uint8_t& codec,
                              uint8_t& has_title,
                              std::vector<uint32_t>& gaps,
                              std::vector<uint32_t>& tfs,
                              std::vector<uint32_t>& titles) {
            const uint32_t df = static_cast<uint32_t>(merged.size());
            block_data.clear();
            gaps.resize(df);
            tfs.resize(df);
            titles.resize(df);
            has_title = 0;
            uint32_t n_title_hits = 0;
            bool unit_tf = true;
            for (uint32_t i = 0; i < df; ++i) {
                gaps[i] = merged[i].doc_id;
                uint32_t tf = std::min<uint32_t>(merged[i].tf, TF_MAX);
                tfs[i] = tf ? tf - 1 : 0;   // Simple-9 stores tf-1
                if (tfs[i]) unit_tf = false;
                titles[i] = std::min<uint32_t>(merged[i].title_tf, TITLE_TF_MAX);
                if (titles[i]) { has_title = 1; ++n_title_hits; }
            }
            auto append_title_sidecar = [&](std::vector<uint8_t>& out) {
                if (!has_title) return;
                write_vbyte(out, n_title_hits);
                uint32_t prev = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    if (!titles[i]) continue;
                    write_vbyte(out, ((i - prev) << 2) | titles[i]);
                    prev = i;
                }
            };
            // Short lists: vbyte. Long lists always S9+EF — vbyte can be a
            // few bytes smaller but is much slower to decode on high-df terms.
            // All-ones body tf: omit the S9 payload (faster and smaller).
            codec = CODEC_VBYTE;
            if (df < POST_VBYTE_DF) {
                uint32_t prev_doc_id = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    uint32_t gap = gaps[i] - prev_doc_id;
                    uint8_t packed = pack_tf(merged[i].tf, titles[i]);
                    uint32_t plain = (packed == 1) ? 1u : 0u;
                    write_vbyte(block_data, (gap << 1) | plain);
                    if (!plain) block_data.push_back(packed);
                    prev_doc_id = gaps[i];
                }
            } else {
                if (!unit_tf) simple9_pack(block_data, tfs.data(), df);
                append_title_sidecar(block_data);
                ef_pack(block_data, gaps.data(), df);
                codec = unit_tf ? CODEC_S9EF_UNIT : CODEC_S9EF;
            }
        };

        struct RangeOut {
            std::vector<uint8_t> postings;
            std::vector<std::string> terms;
            std::vector<uint32_t> dfs;
            std::vector<uint8_t> codecs;
            std::vector<uint8_t> has_titles;
            std::vector<uint32_t> block_sizes;
            std::exception_ptr err;
        };

        int nmerge = 1;
        std::vector<const std::string*> split_lo(static_cast<size_t>(W), nullptr);
        std::vector<const std::string*> split_hi(static_cast<size_t>(W), nullptr);
        if (W > 1) {
            size_t longest = 0;
            for (int w = 1; w < W; ++w)
                if (mem_runs[static_cast<size_t>(w)].size() > mem_runs[longest].size())
                    longest = static_cast<size_t>(w);
            const auto& run = mem_runs[longest];
            if (run.size() >= static_cast<size_t>(W)) {
                nmerge = std::min(W, IR_MAX_THREADS);
                for (int w = 1; w < nmerge; ++w) {
                    size_t idx = (run.size() * static_cast<size_t>(w)) / static_cast<size_t>(nmerge);
                    split_lo[static_cast<size_t>(w)] = &run[idx].first;
                    split_hi[static_cast<size_t>(w) - 1] = &run[idx].first;
                }
            }
        }

        std::vector<RangeOut> ranges(static_cast<size_t>(nmerge));
        auto merge_range = [&](int w) {
            RangeOut& out = ranges[static_cast<size_t>(w)];
            const std::string* lo = split_lo[static_cast<size_t>(w)];
            const std::string* hi = split_hi[static_cast<size_t>(w)];
            try {
                std::vector<MemCursor> readers(static_cast<size_t>(W));
                for (int r = 0; r < W; ++r) {
                    auto& c = readers[static_cast<size_t>(r)];
                    c.lists = &mem_runs[static_cast<size_t>(r)];
                    if (lo && c.lists && !c.lists->empty()) {
                        auto it = std::lower_bound(
                            c.lists->begin(), c.lists->end(), *lo,
                            [](const std::pair<std::string, std::vector<DocPosting>>& a,
                               const std::string& k) { return a.first < k; });
                        c.i = static_cast<size_t>(it - c.lists->begin());
                    } else {
                        c.i = 0;
                    }
                }

                std::vector<DocPosting> merged;
                std::vector<uint8_t> block_data;
                std::vector<uint32_t> gaps, tfs, titles;
                out.terms.reserve(1 << 14);
                out.postings.reserve(1 << 20);

                while (true) {
                    const std::string* min_term = nullptr;
                    for (auto& rdr : readers) {
                        if (!rdr.active()) continue;
                        if (hi && !(rdr.term() < *hi)) continue;
                        if (!min_term || rdr.term() < *min_term) min_term = &rdr.term();
                    }
                    if (!min_term) break;
                    std::string term = *min_term;

                    uint32_t df = 0;
                    for (auto& rdr : readers) {
                        if (!rdr.active() || (hi && !(rdr.term() < *hi)) || rdr.term() != term)
                            continue;
                        df += rdr.df();
                    }

                    merged.clear();
                    merged.reserve(df);
                    for (auto& rdr : readers) {
                        if (!rdr.active() || (hi && !(rdr.term() < *hi)) || rdr.term() != term)
                            continue;
                        // Runs are produced in increasing doc_id order and
                        // merged in run order, so `merged` is already sorted.
                        merged.insert(merged.end(), rdr.posts().begin(), rdr.posts().end());
                        rdr.next();
                    }
                    if (df < buildcfg.min_df) continue;

                    uint8_t codec = CODEC_VBYTE;
                    uint8_t has_title = 0;
                    encode_list(merged, block_data, codec, has_title, gaps, tfs, titles);
                    out.terms.push_back(std::move(term));
                    out.dfs.push_back(df);
                    out.codecs.push_back(codec);
                    out.has_titles.push_back(has_title);
                    out.block_sizes.push_back(static_cast<uint32_t>(block_data.size()));
                    out.postings.insert(out.postings.end(), block_data.begin(), block_data.end());
                }
            } catch (...) {
                out.err = std::current_exception();
            }
        };

        run_parallel(nmerge, merge_range);

        std::ofstream f_postings(index_dir + "/postings.bin", std::ios::binary);
        std::vector<char> post_buf(1 << 20);
        f_postings.rdbuf()->pubsetbuf(post_buf.data(), static_cast<std::streamsize>(post_buf.size()));
        std::ofstream f_dict(index_dir + "/dict.bin", std::ios::binary | std::ios::out);

        uint32_t placeholder = 0;
        f_dict.write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));

        uint64_t current_postings_offset = 0;
        uint64_t prev_postings_offset = 0;
        std::string prev_term;
        uint32_t num_written_terms = 0;

        std::vector<uint8_t> dict_buf;
        dict_buf.reserve(1 << 20);

        for (int w = 0; w < nmerge; ++w) {
            RangeOut& o = ranges[static_cast<size_t>(w)];
            if (o.err) std::rethrow_exception(o.err);
            if (!o.postings.empty()) {
                f_postings.write(reinterpret_cast<const char*>(o.postings.data()),
                                 static_cast<std::streamsize>(o.postings.size()));
            }
            for (size_t i = 0; i < o.terms.size(); ++i) {
                const std::string& term = o.terms[i];
                const uint32_t df = o.dfs[i];
                const uint8_t codec = o.codecs[i];
                const uint8_t has_title = o.has_titles[i];
                const uint64_t term_postings_start = current_postings_offset;
                current_postings_offset += o.block_sizes[i];

                size_t prefix = 0;
                size_t maxp = std::min(prev_term.size(), term.size());
                while (prefix < maxp && prev_term[prefix] == term[prefix]) prefix++;
                if (prefix > 255) prefix = 255;   // keep the shared prefix in one byte

                write_vbyte(dict_buf, static_cast<uint32_t>(prefix));
                write_vbyte(dict_buf, static_cast<uint32_t>(term.size() - prefix));
                dict_buf.insert(dict_buf.end(), term.begin() + prefix, term.end());
                write_vbyte(dict_buf, (df << 3) | (static_cast<uint32_t>(codec) << 1) | has_title);
                write_vbyte64(dict_buf, term_postings_start - prev_postings_offset);
                prev_postings_offset = term_postings_start;
                prev_term = term;
                num_written_terms++;

                if (dict_buf.size() > (1 << 20)) {
                    f_dict.write(reinterpret_cast<const char*>(dict_buf.data()),
                                 static_cast<std::streamsize>(dict_buf.size()));
                    dict_buf.clear();
                }
            }
            o.postings.clear();
            o.postings.shrink_to_fit();
        }

        if (!dict_buf.empty()) {
            f_dict.write(reinterpret_cast<const char*>(dict_buf.data()),
                         static_cast<std::streamsize>(dict_buf.size()));
            dict_buf.clear();
        }
        f_dict.seekp(0);
        f_dict.write(reinterpret_cast<const char*>(&num_written_terms), sizeof(num_written_terms));
        f_dict.close();
        // EFReader may pull extra bytes; 8 zero pad keeps the last list in-bounds.
        const uint8_t pad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        f_postings.write(reinterpret_cast<const char*>(pad), 8);
        f_postings.close();
        mem_runs.clear();
        stamp("merge", t0);
        // Forward index is rebuilt at load (harness times build, not load).
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
        MmapBuffer corpus_map;
        if (corpus_map.map_file(corpus_path) && corpus_map.data) {
            std::vector<std::pair<const char*, size_t>> spans;
            collect_jsonl_spans(reinterpret_cast<const char*>(corpus_map.data),
                                corpus_map.length, spans);
            build_common([&](std::string&, std::string&) { return false; },
                         index_dir, &spans, &corpus_map);
            return;
        }
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

    // RM3 forward index: walk postings, keep mid-df terms per doc (tf×idf),
    // cap at fwd_keep (more slots on long documents). Not stored on disk.
    template <typename Fn>
    void visit_postings(uint32_t tid, Fn&& fn) {
        if (tid >= term_metas.size() || !postings_mmap.data) return;
        const TermMeta& meta = term_metas[tid];
        if (meta.df == 0) return;
        const uint8_t* ptr = postings_mmap.data + meta.postings_offset;
        const uint32_t df = meta.df;
        if (meta.codec == CODEC_VBYTE) {
            uint32_t current_doc = 0;
            for (uint32_t i = 0; i < df; ++i) {
                uint32_t coded = read_vbyte(ptr);
                current_doc += coded >> 1;
                uint8_t packed = (coded & 1u) ? 1 : *ptr++;
                uint32_t tf = packed & 0x3F;
                fn(current_doc, tf);
            }
        } else {
            const bool unit = (meta.codec == CODEC_S9EF_UNIT);
            const uint8_t* p_ef = unit ? ptr : s9_skip(ptr, df);
            if (meta.has_title) {
                uint32_t n_title_hits = read_vbyte(p_ef);
                for (uint32_t h = 0; h < n_title_hits; ++h) (void)read_vbyte(p_ef);
            }
            if (unit) {
                EFReader ef; ef.init(p_ef);
                for (uint32_t i = 0; i < df; ++i) fn(ef.next(), 1u);
            } else {
                S9Reader s9; s9.init(ptr);
                EFReader ef; ef.init(p_ef);
                for (uint32_t i = 0; i < df; ++i) {
                    const uint32_t tf_raw = s9.next();
                    fn(ef.next(), tf_raw + 1u);
                }
            }
        }
    }

    void rebuild_forward() {
        csr_off.clear();
        csr_tid.clear();
        csr_tf.clear();
        if (N == 0 || term_metas.empty()) return;

        const bool small_col = (N < 20000u);
        const uint32_t df_band_min = small_col ? 1u : 5u;
        const uint32_t df_max = small_col ? N : std::max<uint32_t>(
            df_band_min, static_cast<uint32_t>(buildcfg.rm3_df_frac * static_cast<float>(N)));

        const uint32_t nterms = static_cast<uint32_t>(term_metas.size());
        int W = build_workers_for_docs(N);
        if (nterms < 512u || N < 512u) W = 1;
        if (W > static_cast<int>(nterms)) W = static_cast<int>(nterms);
        if (W < 1) W = 1;

        if (W > IR_MAX_THREADS) W = IR_MAX_THREADS;

        auto run_W = [&](auto&& fn) { run_parallel(W, fn); };

        // Pass 1: df-band occupancy + collection frequency, terms sharded.
        std::vector<std::vector<uint32_t>> cnt_w(static_cast<size_t>(W),
                                                 std::vector<uint32_t>(N, 0));
        run_W([&](int w) {
            const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * nterms) / W);
            const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(w + 1) * nterms) / W);
            auto& cnt = cnt_w[static_cast<size_t>(w)];
            for (uint32_t tid = lo; tid < hi; ++tid) {
                uint64_t cf = 0;
                visit_postings(tid, [&](uint32_t d, uint32_t tf) {
                    if (d < N) ++cnt[d];
                    cf += tf;
                });
                term_metas[tid].cf = cf;
            }
        });
        std::vector<uint32_t> cnt(N, 0);
        for (int w = 0; w < W; ++w) {
            const auto& src = cnt_w[static_cast<size_t>(w)];
            for (uint32_t d = 0; d < N; ++d) cnt[d] += src[d];
        }
        cnt_w.clear();
        cnt_w.shrink_to_fit();

        csr_off.assign(static_cast<size_t>(N) + 1, 0);
        for (uint32_t d = 0; d < N; ++d) csr_off[d + 1] = csr_off[d] + cnt[d];
        csr_tid.assign(csr_off[N], 0);
        csr_tf.assign(csr_off[N], 0);

        // Pass 2: scatter postings into CSR. Each term is owned by one
        // worker; docs collide, so the write cursor is atomic.
        std::vector<std::atomic<uint32_t>> cursor(N);
        for (uint32_t d = 0; d < N; ++d)
            cursor[d].store(csr_off[d], std::memory_order_relaxed);
        run_W([&](int w) {
            const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * nterms) / W);
            const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(w + 1) * nterms) / W);
            for (uint32_t tid = lo; tid < hi; ++tid) {
                visit_postings(tid, [&](uint32_t d, uint32_t tf) {
                    if (d >= N) return;
                    uint32_t at = cursor[d].fetch_add(1u, std::memory_order_relaxed);
                    csr_tid[at] = tid;
                    csr_tf[at] = static_cast<uint8_t>(tf > 255u ? 255u : tf);
                });
            }
        });

        // Pass 3: per-doc tf×idf prune — independent across documents.
        std::vector<uint32_t> keep_n(N, 0);
        std::vector<std::vector<uint32_t>> keep_tid(static_cast<size_t>(W));
        std::vector<std::vector<uint32_t>> keep_tf(static_cast<size_t>(W));
        run_W([&](int w) {
            const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * N) / W);
            const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(w + 1) * N) / W);
            auto& out_tid = keep_tid[static_cast<size_t>(w)];
            auto& out_tf = keep_tf[static_cast<size_t>(w)];
            out_tid.reserve(static_cast<size_t>(hi - lo) * std::min<uint32_t>(fwd_keep, 32));
            out_tf.reserve(out_tid.capacity());
            std::vector<std::tuple<uint32_t, float, uint32_t>> cand;
            for (uint32_t d = lo; d < hi; ++d) {
                uint32_t a = csr_off[d], b = csr_off[d + 1];
                cand.clear();
                cand.reserve(b - a);
                for (uint32_t i = a; i < b; ++i) {
                    uint32_t tid = csr_tid[i];
                    if (tid >= terms_sorted.size()) continue;
                    // Large collections: aggressive Fox-style list. Small:
                    // English+question words only (same as indexing).
                    if (small_col ? INDEX_STOPWORDS.count(terms_sorted[tid])
                                  : STOPWORDS.count(terms_sorted[tid]))
                        continue;
                    uint32_t df = term_metas[tid].df;
                    if (df < df_band_min || df > df_max) continue;
                    uint32_t tf = csr_tf[i];
                    float idf = (df > 0 && N > 0)
                        ? std::log(static_cast<float>(N) / static_cast<float>(df))
                        : 0.0f;
                    cand.push_back({tf, static_cast<float>(tf) * idf, tid});
                }
                const uint32_t cand_cap = small_col ? 256u : FWD_CANDIDATES;
                if (cand.size() > cand_cap) {
                    std::partial_sort(cand.begin(),
                                      cand.begin() + static_cast<std::ptrdiff_t>(cand_cap),
                                      cand.end(),
                                      [](const auto& x, const auto& y) {
                                          return std::get<1>(x) > std::get<1>(y);
                                      });
                    cand.resize(cand_cap);
                }
                const float lr = (d < quantized_doc_lens.size())
                    ? dequantize_doc_len(quantized_doc_lens[d]) : 1.0f;
                const uint32_t keep_d = small_col
                    ? 128u
                    : std::min<uint32_t>(
                        FWD_CANDIDATES,
                        static_cast<uint32_t>(fwd_keep * std::min(1.0f + lr, 3.0f) * 0.5f + 0.5f));
                size_t take = std::min(cand.size(), static_cast<size_t>(std::max(keep_d, 8u)));
                if (take < cand.size()) {
                    std::partial_sort(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(take),
                                      cand.end(),
                                      [](const auto& x, const auto& y) {
                                          return std::get<1>(x) > std::get<1>(y);
                                      });
                    cand.resize(take);
                }
                std::sort(cand.begin(), cand.end(),
                          [](const auto& x, const auto& y) {
                              return std::get<2>(x) < std::get<2>(y);
                          });
                keep_n[d] = static_cast<uint32_t>(cand.size());
                for (const auto& t : cand) {
                    out_tid.push_back(std::get<2>(t));
                    out_tf.push_back(std::get<0>(t));
                }
            }
        });

        std::vector<uint32_t> new_off(static_cast<size_t>(N) + 1, 0);
        for (uint32_t d = 0; d < N; ++d) new_off[d + 1] = new_off[d] + keep_n[d];
        std::vector<uint32_t> new_tid(new_off[N]);
        std::vector<uint8_t> new_tf(new_off[N]);
        for (int w = 0; w < W; ++w) {
            const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(w) * N) / W);
            const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(w + 1) * N) / W);
            const auto& src_tid = keep_tid[static_cast<size_t>(w)];
            const auto& src_tf = keep_tf[static_cast<size_t>(w)];
            size_t j = 0;
            for (uint32_t d = lo; d < hi; ++d) {
                uint32_t at = new_off[d];
                uint32_t n = keep_n[d];
                for (uint32_t k = 0; k < n; ++k, ++j) {
                    new_tid[at + k] = src_tid[j];
                    new_tf[at + k] = static_cast<uint8_t>(src_tf[j]);
                }
            }
        }
        csr_off.swap(new_off);
        csr_tid.swap(new_tid);
        csr_tf.swap(new_tf);
    }

    // Load
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
            // Front-coded: shared prefix with previous term, then the suffix.
            uint32_t prefix = read_vbyte(p);
            uint32_t suffix_len = read_vbyte(p);
            std::string term = prev_term.substr(0, prefix);
            term.append(reinterpret_cast<const char*>(p), suffix_len);
            p += suffix_len;

            TermMeta m;
            uint32_t df_coded = read_vbyte(p);
            m.has_title = static_cast<uint8_t>(df_coded & 1u);
            m.codec = static_cast<uint8_t>((df_coded >> 1) & 3u);
            m.df = df_coded >> 3;
            m.postings_offset = prev_postings_offset + read_vbyte64(p);
            prev_postings_offset = m.postings_offset;

            terms_sorted.push_back(term);
            term_metas.push_back(m);
            prev_term = std::move(term);
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
                f.read(reinterpret_cast<char*>(&fwd_keep), sizeof(fwd_keep));
                f.read(reinterpret_cast<char*>(&buildcfg.rm3_df_frac), sizeof(float));
                if (fwd_keep == 0) fwd_keep = 20;
            }
        }
        {
            std::ifstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            uint16_t id_len = 0;
            f.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
            doc_ids.assign(N, std::string());
            if (id_len & 0x8000u) {
                const uint16_t L = id_len & 0x7FFFu;
                uint8_t asz = 0, bw = 0;
                f.read(reinterpret_cast<char*>(&asz), 1);
                std::string alphabet(asz, '\0');
                if (asz) f.read(&alphabet[0], asz);
                f.read(reinterpret_cast<char*>(&bw), 1);
                std::ifstream::pos_type here = f.tellg();
                f.seekg(0, std::ios::end);
                std::streamsize rest = f.tellg() - here;
                f.seekg(here);
                std::vector<uint8_t> body(static_cast<size_t>(rest));
                if (rest) f.read(reinterpret_cast<char*>(body.data()), rest);
                const uint64_t A = asz;
                uint64_t bitpos = 0;
                auto get_bits = [&](int w) -> uint64_t {
                    uint64_t v = 0;
                    int got = 0;
                    while (got < w) {
                        uint64_t byte = body[bitpos / 8];
                        int off = static_cast<int>(bitpos % 8);
                        int take = std::min(w - got, 8 - off);
                        uint64_t chunk = (byte >> off) & ((take == 8) ? 0xFFull : ((1ull << take) - 1ull));
                        v |= chunk << got;
                        bitpos += static_cast<uint64_t>(take);
                        got += take;
                    }
                    return v;
                };
                for (uint32_t i = 0; i < N; ++i) {
                    uint64_t code = get_bits(bw);
                    std::string id(L, ' ');
                    for (int k = static_cast<int>(L) - 1; k >= 0; --k) {
                        id[static_cast<size_t>(k)] = alphabet[static_cast<size_t>(code % A)];
                        code /= A;
                    }
                    doc_ids[i] = std::move(id);
                }
            } else if (id_len > 0) {
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

            fill_deq_lut();
            // Reconstruct |d| from the 8-bit ratio; RM3 Dirichlet uses |d|+μ.
            doc_lens.resize(N);
            for (uint32_t i = 0; i < N; ++i) {
                doc_lens[i] = static_cast<uint32_t>(deq_lut[quantized_doc_lens[i]] * avg_doc_len);
                if (doc_lens[i] == 0) doc_lens[i] = 1;
            }
        }

        load_dictionary(index_dir);

        postings_mmap.map_file(index_dir + "/postings.bin");
        if (postings_mmap.data) {
            const uint8_t* base = postings_mmap.data;
            const float Nf = static_cast<float>(N);
            for (auto& m : term_metas) {
                m.idf = std::log(1.0f + (Nf - static_cast<float>(m.df) + 0.5f)
                                 / (static_cast<float>(m.df) + 0.5f));
                if (m.codec == CODEC_S9EF)
                    m.s9_bytes = static_cast<uint32_t>(
                        s9_skip(base + m.postings_offset, m.df) - (base + m.postings_offset));
                else
                    m.s9_bytes = 0;
            }
        }
        rebuild_forward();

        acc.assign(N, 0.0f);
        acc_stamp.assign(N, 0);
        score_gen = 1;
        touched.clear();
        touched.reserve(N);
    }

    // SCORING
    // Term-at-a-time: for each query term, add BM25+ to acc[doc]. Then
    // nth_element + sort the touched docs. Generation stamps skip a full zero.
    //
    // Score for a hit:  w * ( tf_norm + δ + [title_bonus if title_tf>0] )
    //   w       = query_weight * RSJ idf
    //   tf_norm = tf_eff * (k1+1) / (tf_eff + k1*(1-b+b*|d|/avgdl))
    //   tf_eff  = body_tf [+ w_title * title_tf]
    // Title is the first title_window tokens of the document (build time).
    std::vector<std::pair<uint32_t, float>> raw_retrieve(
        const std::vector<std::pair<int32_t, float>>& qterms,
        int top_k, float k1, float b, float delta, float w_title, float title_bonus)
    {
        if (qterms.empty() || N == 0 || !postings_mmap.data) return {};

        if (++score_gen == 0) {
            std::fill(acc_stamp.begin(), acc_stamp.end(), 0);
            score_gen = 1;
        }
        touched.clear();

        ensure_lp(k1, b);
        const float k1p1 = k1 + 1.0f;
        const bool use_title = (w_title != 0.0f) || (title_bonus != 0.0f);
        float* accp = acc.data();
        uint32_t* stampp = acc_stamp.data();
        const uint8_t* lens = quantized_doc_lens.data();
        const float* lp = lp_lut;
        const uint32_t gen = score_gen;

        for (const auto& qt : qterms) {
            int32_t tid = qt.first;
            if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
            const TermMeta& meta = term_metas[tid];
            if (meta.df == 0) continue;

            float w = qt.second * meta.idf;
            if (w <= 0.0f) continue;

            const uint8_t* ptr = postings_mmap.data + meta.postings_offset;
            const uint32_t df = meta.df;
            if (meta.codec == CODEC_VBYTE) {
                uint32_t current_doc = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    uint32_t coded = read_vbyte(ptr);
                    current_doc += coded >> 1;
                    uint8_t packed = (coded & 1u) ? 1 : *ptr++;
                    uint8_t ttf = use_title ? static_cast<uint8_t>(packed >> 6) : 0;
                    float tf_eff = static_cast<float>(packed & 0x3F);
                    if (w_title != 0.0f) tf_eff += w_title * static_cast<float>(ttf);
                    float tf_norm = (tf_eff * k1p1) / (tf_eff + lp[lens[current_doc]]);
                    float contrib = w * (tf_norm + delta + (ttf ? title_bonus : 0.0f));
                    if (stampp[current_doc] != gen) {
                        stampp[current_doc] = gen;
                        accp[current_doc] = contrib;
                        touched.push_back(current_doc);
                    } else {
                        accp[current_doc] += contrib;
                    }
                }
            } else if (meta.codec == CODEC_S9EF_UNIT) {
                const uint8_t* p_ef = ptr;
                uint32_t n_title_hits = 0;
                if (use_title && meta.has_title) {
                    n_title_hits = read_vbyte(p_ef);
                    scratch_titles.resize(n_title_hits);
                    uint32_t prev = 0;
                    for (uint32_t h = 0; h < n_title_hits; ++h) {
                        uint32_t coded = read_vbyte(p_ef);
                        prev += coded >> 2;
                        scratch_titles[h] = (prev << 2) | (coded & 3u);
                    }
                } else if (meta.has_title) {
                    n_title_hits = read_vbyte(p_ef);
                    for (uint32_t h = 0; h < n_title_hits; ++h) (void)read_vbyte(p_ef);
                    n_title_hits = 0;
                }
                EFReader ef; ef.init(p_ef);
                uint32_t hi = 0;
                for (uint32_t i = 0; i < df; ++i) {
                    const uint32_t current_doc = ef.next();
                    uint8_t ttf = 0;
                    if (hi < n_title_hits && (scratch_titles[hi] >> 2) == i) {
                        ttf = static_cast<uint8_t>(scratch_titles[hi] & 3u);
                        ++hi;
                    }
                    float tf_eff = 1.0f;
                    if (w_title != 0.0f) tf_eff += w_title * static_cast<float>(ttf);
                    float tf_norm = (tf_eff * k1p1) / (tf_eff + lp[lens[current_doc]]);
                    float contrib = w * (tf_norm + delta + (ttf ? title_bonus : 0.0f));
                    if (stampp[current_doc] != gen) {
                        stampp[current_doc] = gen;
                        accp[current_doc] = contrib;
                        touched.push_back(current_doc);
                    } else {
                        accp[current_doc] += contrib;
                    }
                }
            } else {
                const uint8_t* p_ef = ptr + meta.s9_bytes;
                uint32_t n_title_hits = 0;
                if (use_title && meta.has_title) {
                    n_title_hits = read_vbyte(p_ef);
                    scratch_titles.resize(n_title_hits);
                    uint32_t prev = 0;
                    for (uint32_t h = 0; h < n_title_hits; ++h) {
                        uint32_t coded = read_vbyte(p_ef);
                        prev += coded >> 2;
                        scratch_titles[h] = (prev << 2) | (coded & 3u);
                    }
                } else if (meta.has_title) {
                    n_title_hits = read_vbyte(p_ef);
                    for (uint32_t h = 0; h < n_title_hits; ++h) (void)read_vbyte(p_ef);
                    n_title_hits = 0;
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
                    float tf_eff = static_cast<float>(tf_raw + 1u);
                    if (w_title != 0.0f) tf_eff += w_title * static_cast<float>(ttf);
                    float tf_norm = (tf_eff * k1p1) / (tf_eff + lp[lens[current_doc]]);
                    float contrib = w * (tf_norm + delta + (ttf ? title_bonus : 0.0f));
                    if (stampp[current_doc] != gen) {
                        stampp[current_doc] = gen;
                        accp[current_doc] = contrib;
                        touched.push_back(current_doc);
                    } else {
                        accp[current_doc] += contrib;
                    }
                }
            }
        }

        size_t want = std::min(static_cast<size_t>(std::max(top_k, 0)), touched.size());
        auto better = [this](uint32_t a, uint32_t x) {
            if (acc[a] != acc[x]) return acc[a] > acc[x];
            return doc_ids[a] < doc_ids[x];
        };
        if (want < touched.size()) {
            std::nth_element(touched.begin(), touched.begin() + want, touched.end(), better);
        }
        std::sort(touched.begin(), touched.begin() + want, better);

        std::vector<std::pair<uint32_t, float>> results;
        results.reserve(want);
        for (size_t i = 0; i < want; ++i) results.push_back({touched[i], acc[touched[i]]});
        return results;
    }

    // Copy (term_id, tf) rows for the given docs from the CSR.
    void get_batch_doc_term_freqs(const std::vector<uint32_t>& target_docs, BatchFwd& out) {
        const size_t n = target_docs.size();
        out.start.assign(n, 0);
        out.len.assign(n, 0);
        out.arena.clear();
        if (csr_off.size() != static_cast<size_t>(N) + 1) return;
        out.arena.reserve(n * 24);
        for (size_t pos = 0; pos < n; ++pos) {
            const uint32_t d = target_docs[pos];
            if (d >= N) continue;
            const uint32_t a = csr_off[d], b = csr_off[d + 1];
            out.start[pos] = static_cast<uint32_t>(out.arena.size());
            for (uint32_t i = a; i < b; ++i)
                out.arena.push_back({csr_tid[i], csr_tf[i]});
            out.len[pos] = b - a;
        }
    }

    //Entry point
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
        bool apply_qlen_heuristics = true)
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

        const int qn = static_cast<int>(qw.size());
        const int ntok = static_cast<int>(tokens.size());
        // Same cut as retrieve.adapt_params newswire: classic avgdl≥180 at any
        // N≥500, plus large collections with analysed avgdl in [160, 180).
        const bool news_rm3 = (avg_doc_len >= 180.0f && N >= 500u)
            || (avg_doc_len >= 160.0f && N >= 100000u);
        // Query-length overlay on top of Python adapt_params (collection shape).
        // Composed table, competition path only (apply_qlen_heuristics=true):
        //   news + (ntok≥90 | qn≥40)     → RM3 off, δ=0, title_bonus=0
        //   !news + (qn≥20 | ntok≥45)    → RM3 off, δ=0, title_bonus=0
        //   news + RM3 + qn≤2            → α = min(α, 0.45)
        //   !news + RM3 + qn≤4           → α = min(α, 0.42+0.20*(qn-2)/2), pool≥250
        //   !news + RM3 + qn≥12 + N≥20k  → α = max(α, 0.75), fb_terms≤25
        //   !news + RM3                  → pool = min(pool, 250)
        // News descriptions keep Anserini α=0.5 / title_bonus (no extra cut).
        // retrieve_bm25 skips this block so ablations stay BM25+.
        if (apply_qlen_heuristics) {
            // Long queries: RM3 on a document-as-query pulls same-stance
            // neighbours. News descriptions keep RM3; dump-length news does not.
            if (news_rm3) {
                if (ntok >= 90 || qn >= 40) {
                    use_rm3 = false;
                    title_bonus = 0.0f;
                    delta = 0.0f;
                }
            } else if (qn >= 20 || ntok >= 45) {
                use_rm3 = false;
                title_bonus = 0.0f;
                delta = 0.0f;
            }

            // Mix is α·original + (1-α)·expansion; scores are not probabilities.
            if (use_rm3) {
                if (news_rm3) {
                    // Short Robust04 titles: a little more expansion than α=0.5.
                    if (qn <= 2) alpha = std::min(alpha, 0.45f);
                } else if (qn <= 4) {
                    float t = (static_cast<float>(qn) - 2.0f) / 2.0f;
                    if (t < 0.0f) t = 0.0f;
                    if (t > 1.0f) t = 1.0f;
                    // qn=2 → α≤0.42; qn=4 → α≤0.62 (SHORT default).
                    const float short_a = 0.42f + 0.20f * t;
                    alpha = std::min(alpha, short_a);
                    rm3_pool = std::max(rm3_pool, 250);
                } else if (qn >= 12 && N >= 20000u) {
                    alpha = std::max(alpha, 0.75f);
                    fb_terms = std::min(fb_terms, 25);
                }
                if (!news_rm3 && rm3_pool > 250) rm3_pool = 250;
            }
        }

        std::vector<std::pair<int32_t, float>> q_vec(qw.begin(), qw.end());

        int pass1_depth = std::max(k, 0);
        if (use_rm3) {
            pass1_depth = std::max(pass1_depth, fb_docs);
            if (!news_rm3) pass1_depth = std::max(pass1_depth, rm3_pool);
        }
        auto pass1 = raw_retrieve(q_vec, pass1_depth, k1, b, delta, w_title, title_bonus);
        if (pass1.empty()) return {};

        std::vector<std::pair<uint32_t, float>> pool;

        if (!use_rm3 || pass1.size() < 2) {
            pool = pass1;
        } else {
            int num_fb = std::min(static_cast<int>(pass1.size()), fb_docs);

            float mean = 0.0f;
            for (int i = 0; i < num_fb; ++i) mean += pass1[i].second;
            mean /= std::max(num_fb, 1);
            float var = 0.0f;
            for (int i = 0; i < num_fb; ++i) {
                float d = pass1[i].second - mean;
                var += d * d;
            }
            var /= std::max(num_fb, 1);

            if (var < 1e-6f) {
                // First-pass scores are identical: expansion would be uniform noise.
                pool = pass1;
            } else {
                // P(d|R): λ·uniform + (1-λ)·log1p(score)/Σ. Anserini RM3 is
                // uniform on the feedback set (news). Abstracts keep a score
                // mix so the first-pass ranking still matters.
                fb_wts.resize(num_fb);
                float wsum = 0.0f;
                for (int i = 0; i < num_fb; ++i) {
                    float s = pass1[i].second;
                    if (s < 0.0f) s = 0.0f;
                    fb_wts[i] = std::log1p(s);
                    wsum += fb_wts[i];
                }
                const float lam_u = news_rm3 ? 1.00f : 0.25f;
                const float invk = 1.0f / static_cast<float>(std::max(num_fb, 1));
                if (wsum <= 1e-12f) {
                    for (int i = 0; i < num_fb; ++i) fb_wts[i] = invk;
                } else {
                    for (int i = 0; i < num_fb; ++i)
                        fb_wts[i] = lam_u * invk + (1.0f - lam_u) * (fb_wts[i] / wsum);
                }

                std::vector<uint32_t> pool_ids;
                pool_ids.reserve(pass1.size());
                for (const auto& p : pass1) pool_ids.push_back(p.first);
                get_batch_doc_term_freqs(pool_ids, fwd_batch);
                BatchFwd& batch = fwd_batch;

                const float alpha_rm3 = alpha;
                const int keep_rm3 = fb_terms;
                // Large feedback sets: drop terms that appear in only one fb doc.
                const uint32_t min_sup = (num_fb >= 12) ? 2u : 1u;

                std::unordered_map<int32_t, float> p_w_r;
                std::unordered_map<int32_t, uint32_t> fb_df;
                for (int i = 0; i < num_fb; ++i) {
                    uint32_t doc_id = pass1[i].first;
                    float d_len = static_cast<float>(doc_lens[doc_id]);
                    const float denom = d_len + mu;
                    const uint32_t s0 = batch.start[i], sn = batch.len[i];
                    for (uint32_t j = 0; j < sn; ++j) {
                        const auto& tf_kv = batch.arena[s0 + j];
                        int32_t tid = static_cast<int32_t>(tf_kv.first);
                        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
                        const float tf = static_cast<float>(tf_kv.second);
                        // Dirichlet: P(w|d) = (tf + μ P(w|C)) / (|d| + μ)
                        float p_w_c = (total_corpus_tokens > 0)
                            ? (static_cast<float>(term_metas[tid].cf) / total_corpus_tokens)
                            : 1e-4f;
                        p_w_r[tid] += fb_wts[i] * (tf + mu * p_w_c) / denom;
                        fb_df[tid] += 1u;
                    }
                }

                auto idf_of = [&](int32_t tid) {
                    return term_metas[tid].idf;
                };

                std::vector<std::pair<int32_t, float>> expansion;
                expansion.reserve(p_w_r.size());
                for (const auto& e : p_w_r) {
                    auto sit = fb_df.find(e.first);
                    if (sit == fb_df.end() || sit->second < min_sup) continue;
                    expansion.push_back({e.first, e.second * idf_of(e.first)});
                }
                size_t keep = std::min(static_cast<size_t>(std::max(keep_rm3, 0)), expansion.size());
                std::unordered_map<uint32_t, float> exp_rm3;
                if (keep > 0) {
                    std::partial_sort(expansion.begin(), expansion.begin() + keep, expansion.end(),
                                      [](const auto& a, const auto& b) { return a.second > b.second; });
                    expansion.resize(keep);
                    float esum = 0.0f;
                    for (const auto& e : expansion) esum += p_w_r[e.first];
                    if (esum <= 0.0f) esum = 1.0f;
                    for (const auto& e : expansion) {
                        exp_rm3[static_cast<uint32_t>(e.first)] =
                            (1.0f - alpha_rm3) * (p_w_r[e.first] / esum) * idf_of(e.first);
                    }
                }

                if (news_rm3) {
                    // Full second BM25 pass. Reranking the first-pass pool
                    // cannot recover a news doc that never entered the top-250.
                    float esum = 0.0f;
                    for (const auto& kv : exp_rm3) {
                        auto it = p_w_r.find(static_cast<int32_t>(kv.first));
                        if (it != p_w_r.end()) esum += it->second;
                    }
                    if (esum <= 0.0f) esum = 1.0f;
                    std::unordered_map<int32_t, float> mixed;
                    for (const auto& kv : qw)
                        mixed[kv.first] += alpha_rm3 * kv.second;
                    for (const auto& kv : exp_rm3) {
                        int32_t tid = static_cast<int32_t>(kv.first);
                        auto it = p_w_r.find(tid);
                        const float p = (it == p_w_r.end()) ? 0.0f : it->second;
                        mixed[tid] += (1.0f - alpha_rm3) * (p / esum);
                    }
                    std::vector<std::pair<int32_t, float>> q2(mixed.begin(), mixed.end());
                    pool = raw_retrieve(q2, k, k1, b, delta, w_title, title_bonus);
                    if (pool.empty()) pool = pass1;
                } else {
                    // Abstracts: mix expansion onto the first-pass pool only.
                    ensure_lp(k1, b);
                    const float k1p1 = k1 + 1.0f;
                    const size_t n = pass1.size();
                    std::vector<float> s_rm3(n, 0.0f);
                    for (size_t pi = 0; pi < n; ++pi) {
                        uint32_t d = pass1[pi].first;
                        float s = alpha_rm3 * pass1[pi].second;
                        const uint32_t s0 = batch.start[pi], sn = batch.len[pi];
                        if (sn && !exp_rm3.empty()) {
                            float lp = lp_lut[quantized_doc_lens[d]];
                            for (uint32_t j = 0; j < sn; ++j) {
                                const auto& tf_kv = batch.arena[s0 + j];
                                auto e = exp_rm3.find(tf_kv.first);
                                if (e == exp_rm3.end()) continue;
                                float tf_eff = static_cast<float>(tf_kv.second);
                                float tf_norm = (tf_eff * k1p1) / (tf_eff + lp);
                                s += e->second * (tf_norm + delta);
                            }
                        }
                        s_rm3[pi] = s;
                    }
                    std::vector<size_t> ord(n);
                    for (size_t i = 0; i < n; ++i) ord[i] = i;
                    const size_t want_n = std::min(n, static_cast<size_t>(std::max(k, 0)));
                    auto cmp = [&](size_t a, size_t b) {
                        if (s_rm3[a] != s_rm3[b]) return s_rm3[a] > s_rm3[b];
                        return doc_ids[pass1[a].first] < doc_ids[pass1[b].first];
                    };
                    if (want_n < n)
                        std::nth_element(ord.begin(), ord.begin() + want_n, ord.end(), cmp);
                    std::sort(ord.begin(), ord.begin() + want_n, cmp);
                    pool.clear();
                    pool.reserve(want_n);
                    for (size_t i = 0; i < want_n; ++i)
                        pool.push_back({pass1[ord[i]].first, s_rm3[ord[i]]});
                }
            }
        }

        std::vector<std::pair<std::string, float>> results;
        size_t want = std::min(static_cast<size_t>(std::max(k, 0)), pool.size());
        results.reserve(want);
        for (size_t i = 0; i < want; ++i) results.push_back({doc_ids[pool[i].first], pool[i].second});
        return results;
    }

    // BM25+ only: no RM3 and no query-length cuts on δ / title.
    std::vector<std::pair<std::string, float>> retrieve_bm25(
        const std::string& query, int k, float k1, float b, float delta,
        float w_title, float title_bonus)
    {
        return retrieve(query, k, k1, b, delta, w_title, title_bonus,
                        false, 10, 25, 0.6f, 1000.0f, RM3_POOL, false);
    }
};

