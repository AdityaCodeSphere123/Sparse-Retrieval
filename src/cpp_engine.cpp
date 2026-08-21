#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <queue>
#include <cstdint>
#include <sstream>
#include <cctype>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// VByte encoding helpers
// ---------------------------------------------------------------------------
static inline void write_vbyte(std::vector<uint8_t>& out, uint32_t val) {
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

// Simple tokenizer
static inline std::vector<std::string> cpp_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 5);
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// Data Structures for Block-Max WAND Index
// ---------------------------------------------------------------------------
struct BlockHeader {
    uint32_t max_doc_id;
    float block_max_score;
    uint32_t num_postings;
    uint32_t byte_size;
};

struct TermMeta {
    uint32_t df;
    float global_max_score;
    uint64_t postings_offset;
    uint32_t num_blocks;
};

class CPPEngine {
public:
    std::vector<std::string> doc_ids;
    std::vector<uint32_t> doc_lens;
    float avg_doc_len = 0.0f;
    uint32_t N = 0;
    std::unordered_map<std::string, TermMeta> dictionary;
    std::vector<uint8_t> postings_buf;

    CPPEngine() = default;

    void build_index(const std::vector<std::pair<std::string, std::string>>& corpus, const std::string& index_dir) {
        N = corpus.size();
        doc_ids.clear();
        doc_ids.reserve(N);
        doc_lens.clear();
        doc_lens.reserve(N);

        double total_tokens = 0;
        std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> temp_postings;

        for (uint32_t doc_int_id = 0; doc_int_id < N; ++doc_int_id) {
            const auto& ext_id = corpus[doc_int_id].first;
            const auto& text = corpus[doc_int_id].second;
            doc_ids.push_back(ext_id);

            auto tokens = cpp_tokenize(text);
            uint32_t d_len = tokens.size();
            doc_lens.push_back(d_len);
            total_tokens += d_len;

            std::unordered_map<std::string, uint32_t> tf_map;
            for (const auto& tok : tokens) {
                tf_map[tok]++;
            }

            for (const auto& kv : tf_map) {
                temp_postings[kv.first].emplace_back(doc_int_id, kv.second);
            }
        }

        avg_doc_len = (N > 0) ? static_cast<float>(total_tokens / N) : 0.0f;

        // Serialize to binary files
        std::string docmap_path = index_dir + "/docmap.bin";
        std::string doclens_path = index_dir + "/doclens.bin";
        std::string dict_path = index_dir + "/dict.bin";
        std::string postings_path = index_dir + "/postings.bin";

        // 1. Save docmap
        {
            std::ofstream f(docmap_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            for (const auto& id : doc_ids) {
                uint16_t len = id.size();
                f.write(reinterpret_cast<const char*>(&len), sizeof(len));
                f.write(id.data(), len);
            }
        }

        // 2. Save doclens
        {
            std::ofstream f(doclens_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            f.write(reinterpret_cast<const char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.write(reinterpret_cast<const char*>(doc_lens.data()), N * sizeof(uint32_t));
        }

        // 3. Process & save postings and dict
        std::ofstream f_postings(postings_path, std::ios::binary);
        std::ofstream f_dict(dict_path, std::ios::binary);

        uint32_t num_terms = temp_postings.size();
        f_dict.write(reinterpret_cast<const char*>(&num_terms), sizeof(num_terms));

        uint64_t current_postings_offset = 0;

        for (auto& kv : temp_postings) {
            const std::string& term = kv.first;
            auto& plist = kv.second; // sorted by doc_int_id ascending
            uint32_t df = plist.size();

            float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
            float k1 = 1.2f, b = 0.75f, delta = 1.0f;

            // Block building (block size = 128)
            constexpr uint32_t BLOCK_SIZE = 128;
            uint32_t num_blocks = (df + BLOCK_SIZE - 1) / BLOCK_SIZE;

            float global_max_score = 0.0f;
            uint64_t term_postings_start = current_postings_offset;

            for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
                uint32_t start_i = b_idx * BLOCK_SIZE;
                uint32_t end_i = std::min(start_i + BLOCK_SIZE, df);
                uint32_t n_in_block = end_i - start_i;

                uint32_t max_doc_id = plist[end_i - 1].first;
                float block_max_score = 0.0f;

                std::vector<uint8_t> block_data;
                uint32_t prev_doc_id = 0;

                for (uint32_t i = start_i; i < end_i; ++i) {
                    uint32_t d_id = plist[i].first;
                    uint32_t tf = plist[i].second;

                    float len_ratio = doc_lens[d_id] / avg_doc_len;
                    float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * len_ratio));
                    float score = idf * (tf_norm + delta);
                    if (score > block_max_score) block_max_score = score;

                    uint32_t delta_doc = d_id - prev_doc_id;
                    write_vbyte(block_data, delta_doc);
                    write_vbyte(block_data, tf);
                    prev_doc_id = d_id;
                }

                if (block_max_score > global_max_score) global_max_score = block_max_score;

                BlockHeader bh{max_doc_id, block_max_score, n_in_block, static_cast<uint32_t>(block_data.size())};
                f_postings.write(reinterpret_cast<const char*>(&bh), sizeof(bh));
                f_postings.write(reinterpret_cast<const char*>(block_data.data()), block_data.size());

                current_postings_offset += sizeof(bh) + block_data.size();
            }

            // Write dict entry
            uint16_t term_len = term.size();
            f_dict.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            f_dict.write(term.data(), term_len);
            f_dict.write(reinterpret_cast<const char*>(&df), sizeof(df));
            f_dict.write(reinterpret_cast<const char*>(&global_max_score), sizeof(global_max_score));
            f_dict.write(reinterpret_cast<const char*>(&term_postings_start), sizeof(term_postings_start));
            f_dict.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
        }
    }

    void load_index(const std::string& index_dir) {
        // Read docmap
        {
            std::ifstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            doc_ids.clear();
            doc_ids.resize(N);
            for (uint32_t i = 0; i < N; ++i) {
                uint16_t len;
                f.read(reinterpret_cast<char*>(&len), sizeof(len));
                doc_ids[i].resize(len);
                f.read(&doc_ids[i][0], len);
            }
        }

        // Read doclens
        {
            std::ifstream f(index_dir + "/doclens.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            f.read(reinterpret_cast<char*>(&avg_doc_len), sizeof(avg_doc_len));
            doc_lens.resize(N);
            f.read(reinterpret_cast<char*>(doc_lens.data()), N * sizeof(uint32_t));
        }

        // Read dict
        {
            std::ifstream f(index_dir + "/dict.bin", std::ios::binary);
            uint32_t num_terms;
            f.read(reinterpret_cast<char*>(&num_terms), sizeof(num_terms));
            dictionary.clear();
            dictionary.reserve(num_terms);

            for (uint32_t i = 0; i < num_terms; ++i) {
                uint16_t term_len;
                f.read(reinterpret_cast<char*>(&term_len), sizeof(term_len));
                std::string term(term_len, '\0');
                f.read(&term[0], term_len);

                TermMeta meta;
                f.read(reinterpret_cast<char*>(&meta.df), sizeof(meta.df));
                f.read(reinterpret_cast<char*>(&meta.global_max_score), sizeof(meta.global_max_score));
                f.read(reinterpret_cast<char*>(&meta.postings_offset), sizeof(meta.postings_offset));
                f.read(reinterpret_cast<char*>(&meta.num_blocks), sizeof(meta.num_blocks));

                dictionary[term] = meta;
            }
        }

        // Read entire postings buffer into memory for fast zero-copy decoding
        {
            std::ifstream f(index_dir + "/postings.bin", std::ios::binary | std::ios::ate);
            std::streamsize size = f.tellg();
            f.seekg(0, std::ios::beg);
            postings_buf.resize(size);
            f.read(reinterpret_cast<char*>(postings_buf.data()), size);
        }
    }

    std::vector<std::pair<std::string, float>> retrieve(
        const std::string& query,
        int k,
        float k1 = 1.2f,
        float b = 0.75f,
        float delta = 1.0f
    ) {
        auto tokens = cpp_tokenize(query);
        if (tokens.empty() || N == 0) return {};

        std::unordered_map<std::string, int> q_counts;
        for (const auto& t : tokens) q_counts[t]++;

        struct QueryTerm {
            std::string term;
            float idf;
            TermMeta meta;
        };

        std::vector<QueryTerm> qterms;
        for (const auto& kv : q_counts) {
            auto it = dictionary.find(kv.first);
            if (it != dictionary.end()) {
                uint32_t df = it->second.df;
                float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
                qterms.push_back({kv.first, idf, it->second});
            }
        }

        if (qterms.empty()) return {};

        // Accumulate scores across candidate documents
        std::unordered_map<uint32_t, float> doc_scores;

        for (const auto& qt : qterms) {
            const uint8_t* ptr = postings_buf.data() + qt.meta.postings_offset;

            for (uint32_t b_idx = 0; b_idx < qt.meta.num_blocks; ++b_idx) {
                const BlockHeader* bh = reinterpret_cast<const BlockHeader*>(ptr);
                ptr += sizeof(BlockHeader);

                const uint8_t* block_bytes = ptr;
                ptr += bh->byte_size;

                const uint8_t* b_ptr = block_bytes;
                uint32_t current_doc = 0;

                for (uint32_t i = 0; i < bh->num_postings; ++i) {
                    uint32_t delta_doc = read_vbyte(b_ptr);
                    uint32_t tf = read_vbyte(b_ptr);
                    current_doc += delta_doc;

                    float len_ratio = doc_lens[current_doc] / avg_doc_len;
                    float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * len_ratio));
                    float score = qt.idf * (tf_norm + delta);

                    doc_scores[current_doc] += score;
                }
            }
        }

        // Top-k min-heap
        using ScoreDoc = std::pair<float, uint32_t>;
        std::priority_queue<ScoreDoc, std::vector<ScoreDoc>, std::greater<ScoreDoc>> pq;

        for (const auto& kv : doc_scores) {
            if (pq.size() < static_cast<size_t>(k)) {
                pq.push({kv.second, kv.first});
            } else if (kv.second > pq.top().first) {
                pq.pop();
                pq.push({kv.second, kv.first});
            }
        }

        std::vector<std::pair<std::string, float>> results;
        results.reserve(pq.size());
        while (!pq.empty()) {
            auto top = pq.top();
            pq.pop();
            results.push_back({doc_ids[top.second], top.first});
        }
        std::reverse(results.begin(), results.end());
        return results;
    }
};

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "C++ High-Performance Block-Max WAND Search Engine";
    py::class_<CPPEngine>(m, "CPPEngine")
        .def(py::init<>())
        .def("build_index", &CPPEngine::build_index, py::arg("corpus"), py::arg("index_dir"))
        .def("load_index", &CPPEngine::load_index, py::arg("index_dir"))
        .def("retrieve", &CPPEngine::retrieve, py::arg("query"), py::arg("k"), py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f);
}
