#pragma once
// Index build, load, and BM25F+/RM3 retrieve — the CPPEngine class.

#include "ir_config.h"
#include "codecs.h"
#include "analysis.h"
#include "corpus_io.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <exception>
#include <string_view>
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
    void build_common(Supplier&& next_doc, const std::string& index_dir,
                      const std::vector<std::pair<const char*, size_t>>* line_spans = nullptr,
                      MmapBuffer* release_corpus = nullptr) {
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
        int W = build_workers();
        if (Ndocs == 0) W = 1;
        else if (static_cast<uint32_t>(W) > Ndocs) W = static_cast<int>(Ndocs);

        struct Shard {
            std::unordered_map<std::string, std::vector<DocPosting>> postings;
            std::vector<std::string> ids;
            std::string fwdtmp_path;
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
                sh.fwdtmp_path = index_dir + "/fwdtmp_" + std::to_string(w) + ".bin";
                std::ofstream f_fwdtmp(sh.fwdtmp_path, std::ios::binary);

                thread_local std::string tls_id, tls_text;
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
                    std::unordered_map<std::string, uint32_t> term_data;
                    std::unordered_map<std::string, uint32_t> title_tf;
                    uint32_t d_len = 0;
                    accumulate_tf(text_sv, tokcfg, buildcfg.title_window,
                                  term_data, title_tf, d_len);
                    sh.doc_lens.push_back(d_len);
                    sh.tokens += d_len;

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
                        sh.postings[kv.first].push_back(
                            DocPosting{doc_int_id, kv.second,
                                       (ttf == title_tf.end()) ? 0u : ttf->second});
                    }
                }
                f_fwdtmp.close();
            } catch (...) {
                shards[static_cast<size_t>(w)].err = std::current_exception();
            }
        };

        if (W <= 1) {
            ingest_range(0);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(static_cast<size_t>(W));
            for (int w = 0; w < W; ++w) pool.emplace_back(ingest_range, w);
            for (auto& t : pool) t.join();
        }
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

        const std::string fwdtmp_path = index_dir + "/fwdtmp.bin";
        {
            std::ofstream f_fwdtmp(fwdtmp_path, std::ios::binary);
            for (int w = 0; w < W; ++w) {
                const std::string& p = shards[static_cast<size_t>(w)].fwdtmp_path;
                std::ifstream in(p, std::ios::binary);
                if (in) f_fwdtmp << in.rdbuf();
                in.close();
                std::remove(p.c_str());
            }
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
        if (W <= 1) {
            sort_shard(0);
        } else {
            std::vector<std::thread> pool;
            for (int w = 0; w < W; ++w) pool.emplace_back(sort_shard, w);
            for (auto& t : pool) t.join();
        }
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

        // -------- stage 2: k-way merge of in-memory shard runs -------------
        struct MemCursor {
            TermRun* lists = nullptr;
            size_t i = 0;
            bool active() const { return lists && i < lists->size(); }
            const std::string& term() const { return (*lists)[i].first; }
            std::vector<DocPosting>& posts() { return (*lists)[i].second; }
            uint32_t df() const { return static_cast<uint32_t>((*lists)[i].second.size()); }
            void next() { ++i; }
        };

        std::vector<MemCursor> readers(static_cast<size_t>(W));
        for (int w = 0; w < W; ++w) {
            readers[static_cast<size_t>(w)].lists = &mem_runs[static_cast<size_t>(w)];
            readers[static_cast<size_t>(w)].i = 0;
        }

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

        std::vector<DocPosting> merged;
        std::vector<uint8_t> block_data;
        std::vector<uint64_t> cfs_in_order;

        while (true) {
            const std::string* min_term = nullptr;
            for (auto& r : readers) {
                if (!r.active()) continue;
                if (!min_term || r.term() < *min_term) min_term = &r.term();
            }
            if (!min_term) break;
            std::string term = *min_term;

            uint32_t df = 0;
            for (auto& r : readers) if (r.active() && r.term() == term) df += r.df();

            merged.clear();
            merged.reserve(df);
            uint64_t merged_cf = 0;

            for (auto& r : readers) {
                if (!r.active() || r.term() != term) continue;
                for (const auto& dp : r.posts()) {
                    merged_cf += dp.tf;
                    merged.push_back(dp);
                }
                r.next();
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
        mem_runs.clear();
        stamp("merge", t0);

        // -------- stage 3: forward index -----------------------------------
        // We need the dictionary in memory to resolve term ids and apply the
        // df filter, so load what we just wrote and then stream fwdtmp.bin.
        load_dictionary(index_dir);
        for (size_t i = 0; i < term_metas.size() && i < cfs_in_order.size(); ++i)
            term_metas[i].cf = cfs_in_order[i];
        write_forward_index(index_dir, fwdtmp_path);
        std::remove(fwdtmp_path.c_str());
        stamp("forward", t0);
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

        std::unordered_map<std::string, uint32_t> term_to_tid;
        term_to_tid.reserve(terms_sorted.size() * 2 + 1);
        for (uint32_t i = 0; i < terms_sorted.size(); ++i)
            term_to_tid.emplace(terms_sorted[i], i);

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
                auto it = term_to_tid.find(term);
                if (it == term_to_tid.end()) continue;
                int32_t tid = static_cast<int32_t>(it->second);
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
        int fusion_kind = 1,
        int prf2_docs = 30,
        int prf2_terms = 40,
        float prf2_w = 0.25f,
        float prf2_support = 0.0f,
        float prf2_rank_c = 10.0f,
        int rm3_fb_docs = 50,
        int rm3_fb_terms = 15,
        float rm3_lambda = 0.5f,
        float rm3_mu = 1000.0f,
        float rm3_support = 0.15f,
        float rm3_idf_w = 0.5f)
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

        const int f_prf2 = prf2_docs;
        const int f_rm3  = fb_docs;
        int pass1_depth = std::max(std::max(std::max(f_prf2, f_rm3), k),
                                   use_rm3 ? std::max(rm3_pool, k) : k);
        auto pass1 = raw_retrieve(q_vec, pass1_depth, k1, b, delta, w_title, title_bonus);
        if (pass1.empty()) return {};

        std::vector<std::pair<uint32_t, float>> pool;

        if (!use_rm3 || pass1.size() < 2) {
            pool = pass1;
        } else {
            int num_fb = std::min(static_cast<int>(pass1.size()),
                                  use_fusion ? std::max(f_prf2, f_rm3) : fb_docs);

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
                pool = pass1;
            } else {
                const int n_rm3 = std::min(static_cast<int>(pass1.size()), fb_docs);
                const int n_prf2 = std::min(static_cast<int>(pass1.size()), prf2_docs);

                float norm = 0.0f;
                for (int i = 0; i < n_rm3; ++i) norm += pass1[i].second;
                if (norm <= 0.0f) norm = 1.0f;
                if (fb_temp < 19.5f) {
                    const float sd = std::sqrt(var);
                    const float tau = std::max(sd * std::max(fb_temp, 0.05f), 1e-6f);
                    const float smax = pass1[0].second;
                    fb_wts.resize(n_rm3);
                    norm = 0.0f;
                    for (int i = 0; i < n_rm3; ++i) {
                        fb_wts[i] = std::exp((pass1[i].second - smax) / tau);
                        norm += fb_wts[i];
                    }
                    if (norm <= 0.0f) norm = 1.0f;
                    for (int i = 0; i < n_rm3; ++i) fb_wts[i] /= norm;
                } else {
                    fb_wts.resize(n_rm3);
                    for (int i = 0; i < n_rm3; ++i) fb_wts[i] = pass1[i].second / norm;
                }

                std::vector<uint32_t> pool_ids;
                pool_ids.reserve(pass1.size());
                for (const auto& p : pass1) pool_ids.push_back(p.first);
                BatchFwd batch;
                get_batch_doc_term_freqs(pool_ids, batch);

                const float alpha_rm3 = alpha;
                const int keep_rm3 = fb_terms;
                const int keep_prf2 = prf2_terms;

                std::unordered_map<int32_t, float> p_w_r;
                std::unordered_map<int32_t, float> prf2_acc;
                std::unordered_map<int32_t, int> df_prf2;

                for (int i = 0; i < num_fb; ++i) {
                    uint32_t doc_id = pass1[i].first;
                    float d_len = static_cast<float>(doc_lens[doc_id]);
                    const float denom = d_len + mu;
                    const uint32_t s0 = batch.start[i], sn = batch.len[i];
                    const float prf2_dw = 1.0f / (prf2_rank_c + static_cast<float>(i + 1));

                    for (uint32_t j = 0; j < sn; ++j) {
                        const auto& tf_kv = batch.arena[s0 + j];
                        int32_t tid = static_cast<int32_t>(tf_kv.first);
                        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
                        const float tf = static_cast<float>(tf_kv.second);

                        if (i < n_rm3) {
                            float p_w_c = (total_corpus_tokens > 0)
                                ? (static_cast<float>(term_metas[tid].cf) / total_corpus_tokens)
                                : 1e-4f;
                            p_w_r[tid] += fb_wts[i] * (tf + mu * p_w_c) / denom;
                        }
                        if (use_fusion && i < n_prf2 && !qw.count(tid)) {
                            prf2_acc[tid] += prf2_dw * (1.0f + std::log(std::max(tf, 1.0f)));
                            df_prf2[tid] += 1;
                        }
                    }
                }

                auto idf_of = [&](int32_t tid) {
                    return std::log(1.0f + (static_cast<float>(N) - term_metas[tid].df + 0.5f)
                                    / (term_metas[tid].df + 0.5f));
                };

                auto top_rm3 = [&]() -> std::unordered_map<uint32_t, float> {
                    std::vector<std::pair<int32_t, float>> expansion;
                    expansion.reserve(p_w_r.size());
                    for (const auto& e : p_w_r) {
                        expansion.push_back({e.first, e.second * idf_of(e.first)});
                    }
                    size_t keep = std::min(static_cast<size_t>(std::max(keep_rm3, 0)), expansion.size());
                    if (keep == 0) return {};
                    std::partial_sort(expansion.begin(), expansion.begin() + keep, expansion.end(),
                                      [](const auto& a, const auto& b) { return a.second > b.second; });
                    expansion.resize(keep);
                    float esum = 0.0f;
                    for (const auto& e : expansion) esum += p_w_r[e.first];
                    if (esum <= 0.0f) esum = 1.0f;
                    std::unordered_map<uint32_t, float> exp_w;
                    for (const auto& e : expansion) {
                        float idf = idf_of(e.first);
                        exp_w[static_cast<uint32_t>(e.first)] =
                            (1.0f - alpha_rm3) * (p_w_r[e.first] / esum) * idf;
                    }
                    return exp_w;
                };

                auto top_prf2 = [&]() -> std::unordered_map<uint32_t, float> {
                    const int min_df = (prf2_support <= 0.0f) ? 1
                        : std::max(1, static_cast<int>(
                              std::ceil(prf2_support * static_cast<float>(n_prf2))));
                    std::vector<std::pair<int32_t, float>> expansion;
                    for (const auto& e : prf2_acc) {
                        if (df_prf2[e.first] < min_df) continue;
                        expansion.push_back({e.first, e.second * idf_of(e.first)});
                    }
                    size_t keep = std::min(static_cast<size_t>(std::max(keep_prf2, 0)), expansion.size());
                    if (keep == 0) return {};
                    std::partial_sort(expansion.begin(), expansion.begin() + keep, expansion.end(),
                                      [](const auto& a, const auto& b) { return a.second > b.second; });
                    expansion.resize(keep);
                    std::unordered_map<uint32_t, float> exp_w;
                    for (const auto& e : expansion)
                        exp_w[static_cast<uint32_t>(e.first)] = prf2_w * idf_of(e.first);
                    return exp_w;
                };

                auto exp_rm3 = top_rm3();
                auto exp_prf2 = use_fusion ? top_prf2() : std::unordered_map<uint32_t, float>{};

                auto merge_exp = [&]() {
                    auto out = exp_rm3;
                    for (const auto& kv : exp_prf2) out[kv.first] += kv.second;
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
                                     float orig_mix, std::vector<float>& out) {
                    out.assign(n, 0.0f);
                    for (size_t pi = 0; pi < n; ++pi) {
                        uint32_t d = pass1[pi].first;
                        float s = orig_mix * pass1[pi].second;
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
                score_arm(exp_rm3, alpha_rm3, s_rm3);

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
                    score_arm(merge_exp(), alpha_rm3, s_union);
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
                    score_arm(exp_prf2, 1.0f, s_prf2);
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
                    score_arm(exp_prf2, 1.0f, s_prf2);
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

