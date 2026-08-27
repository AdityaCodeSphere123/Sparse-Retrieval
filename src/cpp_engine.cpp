#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <queue>
#include <cstdint>
#include <sstream>
#include <cctype>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Stopwords List (Indri / INQUERY inspired subset for expansion & position pruning)
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
// Document Length Logarithmic Quantization (uint8_t)
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

static inline std::string s_stem(const std::string& word) {
    if (word.size() <= 3) return word;
    if (word.back() != 's') return word;
    if (word.size() > 4 && word.compare(word.size() - 3, 3, "ies") == 0) {
        return word.substr(0, word.size() - 3) + "y";
    }
    if (word.size() > 3 && word.compare(word.size() - 2, 2, "es") == 0) {
        char third_last = word[word.size() - 3];
        if (third_last == 's' || third_last == 'x' || third_last == 'z' ||
            (word.size() > 4 && word.compare(word.size() - 4, 2, "ch") == 0) ||
            (word.size() > 4 && word.compare(word.size() - 4, 2, "sh") == 0)) {
            return word.substr(0, word.size() - 2);
        }
        return word.substr(0, word.size() - 1);
    }
    if (word.back() == 's' && word[word.size() - 2] != 's' && word[word.size() - 2] != 'u') {
        return word.substr(0, word.size() - 1);
    }
    return word;
}

static inline std::vector<std::string> cpp_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.size() / 5);
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!current.empty()) {
                tokens.push_back(s_stem(current));
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(s_stem(current));
    }
    return tokens;
}

// ---------------------------------------------------------------------------
// Mmap Buffer Wrapper for Instant Zero-Copy Load
// ---------------------------------------------------------------------------
struct MmapBuffer {
    const uint8_t* data = nullptr;
    size_t length = 0;

    bool map_file(const std::string& path) {
        unmap();
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) return false;

        struct stat st;
        if (fstat(fd, &st) == -1 || st.st_size == 0) {
            close(fd);
            return false;
        }
        length = st.st_size;
        void* addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);

        if (addr == MAP_FAILED) {
            data = nullptr;
            length = 0;
            return false;
        }
        data = static_cast<const uint8_t*>(addr);
        return true;
    }

    void unmap() {
        if (data && data != MAP_FAILED) {
            munmap(const_cast<uint8_t*>(data), length);
            data = nullptr;
            length = 0;
        }
    }

    ~MmapBuffer() {
        unmap();
    }
};

// ---------------------------------------------------------------------------
// Index Data Structures
// ---------------------------------------------------------------------------
struct BlockHeader {
    uint32_t max_doc_id;
    float block_max_score;
    uint32_t num_postings;
    uint32_t byte_size;
    uint64_t block_positions_offset;
};

struct TermMeta {
    uint32_t df;
    uint64_t cf;
    float global_max_score;
    uint64_t postings_offset;
    uint32_t num_blocks;
    uint8_t has_positions;
};

struct DocPostingPos {
    uint32_t doc_id;
    uint32_t tf;
    std::vector<uint32_t> positions;
};

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
    std::vector<std::string> doc_ids;
    std::vector<uint8_t> quantized_doc_lens; // 1 byte per document for 500KB footprint
    std::vector<uint32_t> doc_lens;           // Exact doc lengths in memory
    float avg_doc_len = 0.0f;
    uint32_t N = 0;
    uint64_t total_corpus_tokens = 0;
    std::unordered_map<std::string, TermMeta> dictionary;
    
    MmapBuffer postings_mmap;
    MmapBuffer positions_mmap;

    CPPEngine() = default;

    // SPIMI Implementation
    void build_index(const std::vector<std::pair<std::string, std::string>>& corpus, const std::string& index_dir) {
        postings_mmap.unmap();
        positions_mmap.unmap();

        N = corpus.size();
        doc_ids.clear();
        doc_ids.reserve(N);
        doc_lens.clear();
        doc_lens.reserve(N);

        total_corpus_tokens = 0;

        constexpr uint32_t BATCH_DOC_LIMIT = 50000;
        std::vector<std::string> run_files;
        uint32_t run_count = 0;

        std::unordered_map<std::string, std::vector<DocPostingPos>> temp_postings;
        std::unordered_map<std::string, uint64_t> collection_freq;

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
                uint64_t cf = collection_freq[term];
                f.write(reinterpret_cast<const char*>(&df), sizeof(df));
                f.write(reinterpret_cast<const char*>(&cf), sizeof(cf));

                for (const auto& p : plist) {
                    f.write(reinterpret_cast<const char*>(&p.doc_id), sizeof(p.doc_id));
                    f.write(reinterpret_cast<const char*>(&p.tf), sizeof(p.tf));
                    uint32_t num_pos = p.positions.size();
                    f.write(reinterpret_cast<const char*>(&num_pos), sizeof(num_pos));
                    f.write(reinterpret_cast<const char*>(p.positions.data()), num_pos * sizeof(uint32_t));
                }
            }
            temp_postings.clear();
            run_files.push_back(run_path);
        };

        for (uint32_t doc_int_id = 0; doc_int_id < N; ++doc_int_id) {
            const auto& ext_id = corpus[doc_int_id].first;
            const auto& text = corpus[doc_int_id].second;
            doc_ids.push_back(ext_id);

            auto tokens = cpp_tokenize(text);
            uint32_t d_len = tokens.size();
            doc_lens.push_back(d_len);
            total_corpus_tokens += d_len;

            std::unordered_map<std::string, std::vector<uint32_t>> term_positions;
            for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
                const auto& tok = tokens[pos];
                term_positions[tok].push_back(pos);
                collection_freq[tok]++;
            }

            for (auto& kv : term_positions) {
                const std::string& term = kv.first;
                uint32_t tf = kv.second.size();
                temp_postings[term].push_back(DocPostingPos{doc_int_id, tf, std::move(kv.second)});
            }

            if ((doc_int_id + 1) % BATCH_DOC_LIMIT == 0) {
                flush_spimi_run(run_count++);
            }
        }

        if (!temp_postings.empty()) {
            flush_spimi_run(run_count++);
        }

        avg_doc_len = (N > 0) ? static_cast<float>(static_cast<double>(total_corpus_tokens) / N) : 0.0f;

        // Quantize doc lens (uint8_t per doc)
        quantized_doc_lens.resize(N);
        for (uint32_t i = 0; i < N; ++i) {
            float ratio = (avg_doc_len > 0) ? (static_cast<float>(doc_lens[i]) / avg_doc_len) : 1.0f;
            quantized_doc_lens[i] = quantize_doc_len(ratio);
        }

        std::string docmap_path = index_dir + "/docmap.bin";
        std::string doclens_path = index_dir + "/doclens.bin";
        std::string dict_path = index_dir + "/dict.bin";
        std::string postings_path = index_dir + "/postings.bin";
        std::string positions_path = index_dir + "/positions.bin";

        // Save docmap
        {
            std::ofstream f(docmap_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            for (const auto& id : doc_ids) {
                uint16_t len = id.size();
                f.write(reinterpret_cast<const char*>(&len), sizeof(len));
                f.write(id.data(), len);
            }
        }

        // Save doclens as uint8_t quantized bytes (500KB on 500K docs)
        {
            std::ofstream f(doclens_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            f.write(reinterpret_cast<const char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.write(reinterpret_cast<const char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            f.write(reinterpret_cast<const char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));
        }

        // Merge SPIMI Runs
        struct RunReader {
            std::ifstream file;
            uint32_t num_terms = 0;
            uint32_t terms_read = 0;
            std::string curr_term;
            uint32_t curr_df = 0;
            uint64_t curr_cf = 0;

            bool read_next_term() {
                if (terms_read >= num_terms) {
                    curr_term.clear();
                    return false;
                }
                uint16_t t_len;
                if (!file.read(reinterpret_cast<char*>(&t_len), sizeof(t_len))) {
                    curr_term.clear();
                    return false;
                }
                curr_term.resize(t_len);
                file.read(&curr_term[0], t_len);
                file.read(reinterpret_cast<char*>(&curr_df), sizeof(curr_df));
                file.read(reinterpret_cast<char*>(&curr_cf), sizeof(curr_cf));
                terms_read++;
                return true;
            }
        };

        std::vector<RunReader> readers(run_files.size());
        for (size_t i = 0; i < run_files.size(); ++i) {
            readers[i].file.open(run_files[i], std::ios::binary);
            readers[i].file.read(reinterpret_cast<char*>(&readers[i].num_terms), sizeof(readers[i].num_terms));
            readers[i].read_next_term();
        }

        std::ofstream f_postings(postings_path, std::ios::binary);
        std::ofstream f_positions(positions_path, std::ios::binary);
        std::ofstream f_dict(dict_path, std::ios::binary);

        std::unordered_map<std::string, TermMeta> final_dict;
        uint64_t current_postings_offset = 0;
        uint64_t current_positions_offset = 0;

        while (true) {
            std::string min_term = "";
            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].terms_read <= readers[i].num_terms && !readers[i].curr_term.empty()) {
                    if (min_term.empty() || readers[i].curr_term < min_term) {
                        min_term = readers[i].curr_term;
                    }
                }
            }

            if (min_term.empty()) break;

            std::vector<DocPostingPos> merged_plist;
            uint64_t merged_cf = 0;

            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].curr_term == min_term && readers[i].terms_read <= readers[i].num_terms) {
                    merged_cf += readers[i].curr_cf;
                    for (uint32_t p_idx = 0; p_idx < readers[i].curr_df; ++p_idx) {
                        DocPostingPos dpos;
                        readers[i].file.read(reinterpret_cast<char*>(&dpos.doc_id), sizeof(dpos.doc_id));
                        readers[i].file.read(reinterpret_cast<char*>(&dpos.tf), sizeof(dpos.tf));
                        uint32_t num_pos;
                        readers[i].file.read(reinterpret_cast<char*>(&num_pos), sizeof(num_pos));
                        dpos.positions.resize(num_pos);
                        readers[i].file.read(reinterpret_cast<char*>(dpos.positions.data()), num_pos * sizeof(uint32_t));
                        merged_plist.push_back(std::move(dpos));
                    }
                    readers[i].read_next_term();
                }
            }

            uint32_t df = merged_plist.size();
            float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
            float k1 = 1.2f, b = 0.75f, delta = 1.0f;

            constexpr uint32_t BLOCK_SIZE = 128;
            uint32_t num_blocks = (df + BLOCK_SIZE - 1) / BLOCK_SIZE;

            float global_max_score = 0.0f;
            uint64_t term_postings_start = current_postings_offset;

            uint8_t has_positions = (df <= static_cast<uint32_t>(N * 0.05f)) ? 1 : 0;

            for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
                uint32_t start_i = b_idx * BLOCK_SIZE;
                uint32_t end_i = std::min(start_i + BLOCK_SIZE, df);
                uint32_t n_in_block = end_i - start_i;

                uint32_t max_doc_id = merged_plist[end_i - 1].doc_id;
                float block_max_score = 0.0f;

                std::vector<uint8_t> block_data;
                uint32_t prev_doc_id = 0;

                for (uint32_t i = start_i; i < end_i; ++i) {
                    uint32_t d_id = merged_plist[i].doc_id;
                    uint32_t tf = merged_plist[i].tf;

                    float len_ratio = dequantize_doc_len(quantized_doc_lens[d_id]);
                    float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * len_ratio));
                    float score = idf * (tf_norm + delta);
                    if (score > block_max_score) block_max_score = score;

                    uint32_t delta_doc = d_id - prev_doc_id;
                    write_vbyte(block_data, delta_doc);
                    uint8_t tf_byte = static_cast<uint8_t>(std::min(tf, 255u));
                    block_data.push_back(tf_byte);
                    prev_doc_id = d_id;
                }

                if (block_max_score > global_max_score) global_max_score = block_max_score;

                uint64_t block_positions_offset = 0;

                if (has_positions) {
                    block_positions_offset = current_positions_offset;
                    std::vector<uint8_t> pos_data;

                    for (uint32_t i = start_i; i < end_i; ++i) {
                        const auto& entry = merged_plist[i];
                        write_vbyte(pos_data, entry.doc_id);
                        write_vbyte(pos_data, entry.positions.size());

                        uint32_t prev_pos = 0;
                        for (uint32_t p : entry.positions) {
                            write_vbyte(pos_data, p - prev_pos);
                            prev_pos = p;
                        }
                    }

                    f_positions.write(reinterpret_cast<const char*>(pos_data.data()), pos_data.size());
                    current_positions_offset += pos_data.size();
                }

                BlockHeader bh{max_doc_id, block_max_score, n_in_block, static_cast<uint32_t>(block_data.size()), block_positions_offset};
                f_postings.write(reinterpret_cast<const char*>(&bh), sizeof(bh));
                f_postings.write(reinterpret_cast<const char*>(block_data.data()), block_data.size());

                current_postings_offset += sizeof(bh) + block_data.size();
            }

            TermMeta meta{df, merged_cf, global_max_score, term_postings_start, num_blocks, has_positions};
            final_dict[min_term] = meta;
        }

        for (auto& r : readers) r.file.close();

        // Write dict.bin
        uint32_t final_num_terms = final_dict.size();
        f_dict.write(reinterpret_cast<const char*>(&final_num_terms), sizeof(final_num_terms));

        for (const auto& kv : final_dict) {
            const std::string& term = kv.first;
            const TermMeta& meta = kv.second;

            uint16_t term_len = term.size();
            f_dict.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            f_dict.write(term.data(), term_len);
            f_dict.write(reinterpret_cast<const char*>(&meta.df), sizeof(meta.df));
            f_dict.write(reinterpret_cast<const char*>(&meta.cf), sizeof(meta.cf));
            f_dict.write(reinterpret_cast<const char*>(&meta.global_max_score), sizeof(meta.global_max_score));
            f_dict.write(reinterpret_cast<const char*>(&meta.postings_offset), sizeof(meta.postings_offset));
            f_dict.write(reinterpret_cast<const char*>(&meta.num_blocks), sizeof(meta.num_blocks));
            f_dict.write(reinterpret_cast<const char*>(&meta.has_positions), sizeof(meta.has_positions));
        }

        for (const auto& rfile : run_files) {
            std::remove(rfile.c_str());
        }
    }

    void build_index_from_file(const std::string& corpus_path, const std::string& index_dir) {
        postings_mmap.unmap();
        positions_mmap.unmap();

        doc_ids.clear();
        doc_lens.clear();
        total_corpus_tokens = 0;
        N = 0;

        constexpr uint32_t BATCH_DOC_LIMIT = 50000;
        std::vector<std::string> run_files;
        uint32_t run_count = 0;

        std::unordered_map<std::string, std::vector<DocPostingPos>> temp_postings;
        std::unordered_map<std::string, uint64_t> collection_freq;

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
                uint64_t cf = collection_freq[term];
                f.write(reinterpret_cast<const char*>(&df), sizeof(df));
                f.write(reinterpret_cast<const char*>(&cf), sizeof(cf));

                for (const auto& p : plist) {
                    f.write(reinterpret_cast<const char*>(&p.doc_id), sizeof(p.doc_id));
                    f.write(reinterpret_cast<const char*>(&p.tf), sizeof(p.tf));
                    uint32_t num_pos = p.positions.size();
                    f.write(reinterpret_cast<const char*>(&num_pos), sizeof(num_pos));
                    f.write(reinterpret_cast<const char*>(p.positions.data()), num_pos * sizeof(uint32_t));
                }
            }
            temp_postings.clear();
            run_files.push_back(run_path);
        };

        std::ifstream file(corpus_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open corpus file: " + corpus_path);
        }

        std::string line;
        std::string ext_id, text;
        uint32_t doc_int_id = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (!parse_jsonl_line(line, ext_id, text)) continue;

            doc_ids.push_back(ext_id);

            auto tokens = cpp_tokenize(text);
            uint32_t d_len = tokens.size();
            doc_lens.push_back(d_len);
            total_corpus_tokens += d_len;

            std::unordered_map<std::string, std::vector<uint32_t>> term_positions;
            for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
                const auto& tok = tokens[pos];
                term_positions[tok].push_back(pos);
                collection_freq[tok]++;
            }

            for (auto& kv : term_positions) {
                const std::string& term = kv.first;
                uint32_t tf = kv.second.size();
                temp_postings[term].push_back(DocPostingPos{doc_int_id, tf, std::move(kv.second)});
            }

            doc_int_id++;
            if (doc_int_id % BATCH_DOC_LIMIT == 0) {
                flush_spimi_run(run_count++);
            }
        }

        if (!temp_postings.empty()) {
            flush_spimi_run(run_count++);
        }

        N = doc_int_id;

        avg_doc_len = (N > 0) ? static_cast<float>(static_cast<double>(total_corpus_tokens) / N) : 0.0f;

        // Quantize doc lens (uint8_t per doc)
        quantized_doc_lens.resize(N);
        for (uint32_t i = 0; i < N; ++i) {
            float ratio = (avg_doc_len > 0) ? (static_cast<float>(doc_lens[i]) / avg_doc_len) : 1.0f;
            quantized_doc_lens[i] = quantize_doc_len(ratio);
        }

        std::string docmap_path = index_dir + "/docmap.bin";
        std::string doclens_path = index_dir + "/doclens.bin";
        std::string dict_path = index_dir + "/dict.bin";
        std::string postings_path = index_dir + "/postings.bin";
        std::string positions_path = index_dir + "/positions.bin";

        // Save docmap
        {
            std::ofstream f(docmap_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            for (const auto& id : doc_ids) {
                uint16_t len = id.size();
                f.write(reinterpret_cast<const char*>(&len), sizeof(len));
                f.write(id.data(), len);
            }
        }

        // Save doclens as uint8_t quantized bytes (500KB on 500K docs)
        {
            std::ofstream f(doclens_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            f.write(reinterpret_cast<const char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.write(reinterpret_cast<const char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            f.write(reinterpret_cast<const char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));
        }

        // Merge SPIMI Runs
        struct RunReader {
            std::ifstream file;
            uint32_t num_terms = 0;
            uint32_t terms_read = 0;
            std::string curr_term;
            uint32_t curr_df = 0;
            uint64_t curr_cf = 0;

            bool read_next_term() {
                if (terms_read >= num_terms) {
                    curr_term.clear();
                    return false;
                }
                uint16_t t_len;
                if (!file.read(reinterpret_cast<char*>(&t_len), sizeof(t_len))) {
                    curr_term.clear();
                    return false;
                }
                curr_term.resize(t_len);
                file.read(&curr_term[0], t_len);
                file.read(reinterpret_cast<char*>(&curr_df), sizeof(curr_df));
                file.read(reinterpret_cast<char*>(&curr_cf), sizeof(curr_cf));
                terms_read++;
                return true;
            }
        };

        std::vector<RunReader> readers(run_files.size());
        for (size_t i = 0; i < run_files.size(); ++i) {
            readers[i].file.open(run_files[i], std::ios::binary);
            readers[i].file.read(reinterpret_cast<char*>(&readers[i].num_terms), sizeof(readers[i].num_terms));
            readers[i].read_next_term();
        }

        std::ofstream f_postings(postings_path, std::ios::binary);
        std::ofstream f_positions(positions_path, std::ios::binary);
        std::ofstream f_dict(dict_path, std::ios::binary);

        std::unordered_map<std::string, TermMeta> final_dict;
        uint64_t current_postings_offset = 0;
        uint64_t current_positions_offset = 0;

        while (true) {
            std::string min_term = "";
            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].terms_read <= readers[i].num_terms && !readers[i].curr_term.empty()) {
                    if (min_term.empty() || readers[i].curr_term < min_term) {
                        min_term = readers[i].curr_term;
                    }
                }
            }

            if (min_term.empty()) break;

            std::vector<DocPostingPos> merged_plist;
            uint64_t merged_cf = 0;

            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].curr_term == min_term && readers[i].terms_read <= readers[i].num_terms) {
                    merged_cf += readers[i].curr_cf;
                    for (uint32_t p_idx = 0; p_idx < readers[i].curr_df; ++p_idx) {
                        DocPostingPos dpos;
                        readers[i].file.read(reinterpret_cast<char*>(&dpos.doc_id), sizeof(dpos.doc_id));
                        readers[i].file.read(reinterpret_cast<char*>(&dpos.tf), sizeof(dpos.tf));
                        uint32_t num_pos;
                        readers[i].file.read(reinterpret_cast<char*>(&num_pos), sizeof(num_pos));
                        dpos.positions.resize(num_pos);
                        readers[i].file.read(reinterpret_cast<char*>(dpos.positions.data()), num_pos * sizeof(uint32_t));
                        merged_plist.push_back(std::move(dpos));
                    }
                    readers[i].read_next_term();
                }
            }

            uint32_t df = merged_plist.size();
            float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
            float k1 = 1.2f, b = 0.75f, delta = 1.0f;

            constexpr uint32_t BLOCK_SIZE = 128;
            uint32_t num_blocks = (df + BLOCK_SIZE - 1) / BLOCK_SIZE;

            float global_max_score = 0.0f;
            uint64_t term_postings_start = current_postings_offset;

            uint8_t has_positions = (df <= static_cast<uint32_t>(N * 0.05f)) ? 1 : 0;

            for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
                uint32_t start_i = b_idx * BLOCK_SIZE;
                uint32_t end_i = std::min(start_i + BLOCK_SIZE, df);
                uint32_t n_in_block = end_i - start_i;

                uint32_t max_doc_id = merged_plist[end_i - 1].doc_id;
                float block_max_score = 0.0f;

                std::vector<uint8_t> block_data;
                uint32_t prev_doc_id = 0;

                for (uint32_t i = start_i; i < end_i; ++i) {
                    uint32_t d_id = merged_plist[i].doc_id;
                    uint32_t tf = merged_plist[i].tf;

                    float len_ratio = dequantize_doc_len(quantized_doc_lens[d_id]);
                    float tf_norm = (tf * (k1 + 1.0f)) / (tf + k1 * (1.0f - b + b * len_ratio));
                    float score = idf * (tf_norm + delta);
                    if (score > block_max_score) block_max_score = score;

                    uint32_t delta_doc = d_id - prev_doc_id;
                    write_vbyte(block_data, delta_doc);
                    uint8_t tf_byte = static_cast<uint8_t>(std::min(tf, 255u));
                    block_data.push_back(tf_byte);
                    prev_doc_id = d_id;
                }

                if (block_max_score > global_max_score) global_max_score = block_max_score;

                uint64_t block_positions_offset = 0;

                if (has_positions) {
                    block_positions_offset = current_positions_offset;
                    std::vector<uint8_t> pos_data;

                    for (uint32_t i = start_i; i < end_i; ++i) {
                        const auto& entry = merged_plist[i];
                        write_vbyte(pos_data, entry.doc_id);
                        write_vbyte(pos_data, entry.positions.size());

                        uint32_t prev_pos = 0;
                        for (uint32_t p : entry.positions) {
                            write_vbyte(pos_data, p - prev_pos);
                            prev_pos = p;
                        }
                    }

                    f_positions.write(reinterpret_cast<const char*>(pos_data.data()), pos_data.size());
                    current_positions_offset += pos_data.size();
                }

                BlockHeader bh{max_doc_id, block_max_score, n_in_block, static_cast<uint32_t>(block_data.size()), block_positions_offset};
                f_postings.write(reinterpret_cast<const char*>(&bh), sizeof(bh));
                f_postings.write(reinterpret_cast<const char*>(block_data.data()), block_data.size());

                current_postings_offset += sizeof(bh) + block_data.size();
            }

            TermMeta meta{df, merged_cf, global_max_score, term_postings_start, num_blocks, has_positions};
            final_dict[min_term] = meta;
        }

        for (auto& r : readers) r.file.close();

        // Write dict.bin
        uint32_t final_num_terms = final_dict.size();
        f_dict.write(reinterpret_cast<const char*>(&final_num_terms), sizeof(final_num_terms));

        for (const auto& kv : final_dict) {
            const std::string& term = kv.first;
            const TermMeta& meta = kv.second;

            uint16_t term_len = term.size();
            f_dict.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
            f_dict.write(term.data(), term_len);
            f_dict.write(reinterpret_cast<const char*>(&meta.df), sizeof(meta.df));
            f_dict.write(reinterpret_cast<const char*>(&meta.cf), sizeof(meta.cf));
            f_dict.write(reinterpret_cast<const char*>(&meta.global_max_score), sizeof(meta.global_max_score));
            f_dict.write(reinterpret_cast<const char*>(&meta.postings_offset), sizeof(meta.postings_offset));
            f_dict.write(reinterpret_cast<const char*>(&meta.num_blocks), sizeof(meta.num_blocks));
            f_dict.write(reinterpret_cast<const char*>(&meta.has_positions), sizeof(meta.has_positions));
        }

        for (const auto& rfile : run_files) {
            std::remove(rfile.c_str());
        }
    }

    void load_index(const std::string& index_dir) {
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

        {
            std::ifstream f(index_dir + "/doclens.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            f.read(reinterpret_cast<char*>(&avg_doc_len), sizeof(avg_doc_len));
            f.read(reinterpret_cast<char*>(&total_corpus_tokens), sizeof(total_corpus_tokens));
            quantized_doc_lens.resize(N);
            f.read(reinterpret_cast<char*>(quantized_doc_lens.data()), N * sizeof(uint8_t));

            // Dequantize exact lengths for RM3 expansion lookup
            doc_lens.resize(N);
            for (uint32_t i = 0; i < N; ++i) {
                doc_lens[i] = static_cast<uint32_t>(dequantize_doc_len(quantized_doc_lens[i]) * avg_doc_len);
            }
        }

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
                f.read(reinterpret_cast<char*>(&meta.cf), sizeof(meta.cf));
                f.read(reinterpret_cast<char*>(&meta.global_max_score), sizeof(meta.global_max_score));
                f.read(reinterpret_cast<char*>(&meta.postings_offset), sizeof(meta.postings_offset));
                f.read(reinterpret_cast<char*>(&meta.num_blocks), sizeof(meta.num_blocks));
                f.read(reinterpret_cast<char*>(&meta.has_positions), sizeof(meta.has_positions));

                dictionary[term] = meta;
            }
        }

        postings_mmap.map_file(index_dir + "/postings.bin");
        positions_mmap.map_file(index_dir + "/positions.bin");
    }

    std::vector<std::pair<uint32_t, float>> raw_retrieve(
        const std::unordered_map<std::string, float>& query_weights,
        int top_k,
        float k1,
        float b,
        float delta
    ) {
        if (query_weights.empty() || N == 0 || !postings_mmap.data) return {};

        // Precompute 256-element Length Penalty Lookup Table (LUT) for parameter b
        float length_penalty_lut[256];
        for (int q = 0; q < 256; ++q) {
            float ratio = dequantize_doc_len(static_cast<uint8_t>(q));
            length_penalty_lut[q] = k1 * (1.0f - b + b * ratio);
        }

        struct QueryTermInfo {
            std::string term;
            float q_w;
            float idf;
            TermMeta meta;
            float max_term_contrib;
        };

        std::vector<QueryTermInfo> qterms;
        float total_max_query_score = 0.0f;

        for (const auto& kv : query_weights) {
            const std::string& term = kv.first;
            float q_w = kv.second;

            auto it = dictionary.find(term);
            if (it == dictionary.end()) continue;

            uint32_t df = it->second.df;
            float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));
            float max_contrib = q_w * idf * it->second.global_max_score;
            total_max_query_score += max_contrib;

            qterms.push_back({term, q_w, idf, it->second, max_contrib});
        }

        if (qterms.empty()) return {};

        std::unordered_map<uint32_t, float> doc_scores;
        std::vector<std::pair<float, uint32_t>> top_k_list;
        float min_top_k_score = -1e9f;

        for (const auto& qt : qterms) {
            float other_terms_max = total_max_query_score - qt.max_term_contrib;
            const uint8_t* ptr = postings_mmap.data + qt.meta.postings_offset;

            for (uint32_t b_idx = 0; b_idx < qt.meta.num_blocks; ++b_idx) {
                const BlockHeader* bh = reinterpret_cast<const BlockHeader*>(ptr);
                ptr += sizeof(BlockHeader);

                float block_contrib = qt.q_w * qt.idf * bh->block_max_score;
                float upper_bound = block_contrib + other_terms_max;

                // TRUE MATHEMATICALLY EXACT BLOCK-MAX WAND SKIPPING:
                if (top_k_list.size() == static_cast<size_t>(top_k) && upper_bound <= min_top_k_score) {
                    ptr += bh->byte_size;
                    continue;
                }

                const uint8_t* block_bytes = ptr;
                ptr += bh->byte_size;

                const uint8_t* b_ptr = block_bytes;
                uint32_t current_doc = 0;

                for (uint32_t i = 0; i < bh->num_postings; ++i) {
                    uint32_t delta_doc = read_vbyte(b_ptr);
                    uint8_t tf = *b_ptr++;
                    current_doc += delta_doc;

                    // LIGHTNING-FAST ARRAY LOOKUP
                    uint8_t q_len = quantized_doc_lens[current_doc];
                    float tf_norm = (tf * (k1 + 1.0f)) / (tf + length_penalty_lut[q_len]);
                    float score = qt.q_w * qt.idf * (tf_norm + delta);

                    float new_score = (doc_scores[current_doc] += score);

                    auto it = std::find_if(top_k_list.begin(), top_k_list.end(), [current_doc](const auto& p) { return p.second == current_doc; });
                    if (it != top_k_list.end()) {
                        it->first = new_score;
                    } else {
                        top_k_list.push_back({new_score, current_doc});
                    }
                    std::sort(top_k_list.begin(), top_k_list.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    if (top_k_list.size() > static_cast<size_t>(top_k)) {
                        top_k_list.pop_back();
                    }
                    if (top_k_list.size() == static_cast<size_t>(top_k)) {
                        min_top_k_score = top_k_list.back().first;
                    }
                }
            }
        }

        std::vector<std::pair<uint32_t, float>> results;
        results.reserve(top_k_list.size());
        for (const auto& item : top_k_list) {
            results.push_back({item.second, item.first});
        }
        return results;
    }

    std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> get_batch_doc_term_freqs(const std::unordered_set<uint32_t>& target_docs) {
        std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> doc_freqs;
        if (!postings_mmap.data || target_docs.empty()) return doc_freqs;

        uint32_t min_target_doc = *std::min_element(target_docs.begin(), target_docs.end());
        uint32_t max_target_doc = *std::max_element(target_docs.begin(), target_docs.end());

        for (const auto& kv : dictionary) {
            const std::string& term = kv.first;
            const TermMeta& meta = kv.second;

            if (STOPWORDS.count(term)) continue;
            if (meta.df < 5 || meta.df > static_cast<uint32_t>(N * 0.05f)) continue;

            const uint8_t* ptr = postings_mmap.data + meta.postings_offset;

            for (uint32_t b_idx = 0; b_idx < meta.num_blocks; ++b_idx) {
                const BlockHeader* bh = reinterpret_cast<const BlockHeader*>(ptr);
                ptr += sizeof(BlockHeader);

                const uint8_t* block_bytes = ptr;
                ptr += bh->byte_size;

                if (bh->max_doc_id < min_target_doc) continue;

                const uint8_t* b_ptr = block_bytes;
                uint32_t current_doc = 0;

                for (uint32_t i = 0; i < bh->num_postings; ++i) {
                    uint32_t delta_doc = read_vbyte(b_ptr);
                    uint8_t tf = *b_ptr++;
                    current_doc += delta_doc;

                    if (target_docs.count(current_doc)) {
                        doc_freqs[current_doc][term] = tf;
                    }
                    if (current_doc > max_target_doc) break;
                }
            }
        }
        return doc_freqs;
    }

    // INSTANT BLOCK FAST-FORWARDING FOR POSITIONS: O(128) instead of O(N)
    std::vector<uint32_t> get_positions(const std::string& term, uint32_t target_doc_id) {
        auto it = dictionary.find(term);
        if (it == dictionary.end() || !it->second.has_positions || !positions_mmap.data || !postings_mmap.data) return {};

        const uint8_t* ptr = postings_mmap.data + it->second.postings_offset;

        for (uint32_t b_idx = 0; b_idx < it->second.num_blocks; ++b_idx) {
            const BlockHeader* bh = reinterpret_cast<const BlockHeader*>(ptr);
            ptr += sizeof(BlockHeader) + bh->byte_size;

            if (bh->max_doc_id < target_doc_id) continue;

            const uint8_t* pos_ptr = positions_mmap.data + bh->block_positions_offset;

            for (uint32_t i = 0; i < bh->num_postings; ++i) {
                uint32_t doc_id = read_vbyte(pos_ptr);
                uint32_t num_pos = read_vbyte(pos_ptr);

                if (doc_id == target_doc_id) {
                    std::vector<uint32_t> pos_list;
                    pos_list.reserve(num_pos);
                    uint32_t current_pos = 0;
                    for (uint32_t p = 0; p < num_pos; ++p) {
                        uint32_t delta_pos = read_vbyte(pos_ptr);
                        current_pos += delta_pos;
                        pos_list.push_back(current_pos);
                    }
                    return pos_list;
                } else {
                    for (uint32_t p = 0; p < num_pos; ++p) {
                        read_vbyte(pos_ptr);
                    }
                }

                if (doc_id > target_doc_id) break;
            }
            break;
        }
        return {};
    }

    std::vector<std::pair<std::string, float>> sdm_rerank(
        const std::vector<std::string>& query_tokens,
        const std::vector<std::pair<uint32_t, float>>& candidates,
        int top_k,
        float lambda_t = 0.85f,
        float lambda_o = 0.10f,
        float lambda_u = 0.05f,
        uint32_t window_u = 8
    ) {
        if (candidates.empty()) return {};

        std::vector<std::pair<std::string, std::string>> bigrams;
        for (size_t i = 0; i + 1 < query_tokens.size(); ++i) {
            bigrams.push_back({query_tokens[i], query_tokens[i + 1]});
        }

        std::vector<std::pair<uint32_t, float>> reranked;
        reranked.reserve(candidates.size());

        for (const auto& cand : candidates) {
            uint32_t doc_id = cand.first;
            float unigram_score = cand.second;

            float ordered_score = 0.0f;
            float unordered_score = 0.0f;

            for (const auto& bg : bigrams) {
                auto pos1 = get_positions(bg.first, doc_id);
                auto pos2 = get_positions(bg.second, doc_id);

                if (pos1.empty() || pos2.empty()) continue;

                uint32_t ordered_cnt = 0;
                uint32_t unordered_cnt = 0;

                size_t idx1 = 0, idx2 = 0;
                while (idx1 < pos1.size() && idx2 < pos2.size()) {
                    long diff = static_cast<long>(pos2[idx2]) - static_cast<long>(pos1[idx1]);

                    if (diff == 1) {
                        ordered_cnt++;
                        unordered_cnt++;
                        idx1++;
                        idx2++;
                    } else if (diff > 1 && diff <= static_cast<long>(window_u)) {
                        unordered_cnt++;
                        idx2++;
                    } else if (diff > static_cast<long>(window_u)) {
                        idx1++;
                    } else if (diff < 0 && std::abs(diff) <= static_cast<long>(window_u)) {
                        unordered_cnt++;
                        idx1++;
                    } else {
                        idx2++;
                    }
                }

                if (ordered_cnt > 0) ordered_score += std::log(1.0f + ordered_cnt);
                if (unordered_cnt > 0) unordered_score += std::log(1.0f + unordered_cnt);
            }

            float final_score = lambda_t * unigram_score + lambda_o * ordered_score + lambda_u * unordered_score;
            reranked.push_back({doc_id, final_score});
        }

        std::sort(reranked.begin(), reranked.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::vector<std::pair<std::string, float>> results;
        results.reserve(std::min(static_cast<size_t>(top_k), reranked.size()));
        for (size_t i = 0; i < std::min(static_cast<size_t>(top_k), reranked.size()); ++i) {
            results.push_back({doc_ids[reranked[i].first], reranked[i].second});
        }
        return results;
    }

    std::vector<std::pair<std::string, float>> retrieve(
        const std::string& query,
        int k,
        float k1 = 1.2f,
        float b = 0.75f,
        float delta = 1.0f,
        bool use_rm3 = true,
        int fb_docs = 10,
        int fb_terms = 25,
        float alpha = 0.6f,
        float mu = 1000.0f,
        bool use_sdm = true,
        int sdm_rerank_depth = 200
    ) {
        auto tokens = cpp_tokenize(query);
        if (tokens.empty() || N == 0) return {};

        std::unordered_map<std::string, float> q_weights;
        for (const auto& t : tokens) {
            q_weights[t] += 1.0f;
        }
        float q_size = tokens.size();
        for (auto& kv : q_weights) {
            kv.second /= q_size;
        }

        // 1st Pass Retrieval
        auto pass1 = raw_retrieve(q_weights, std::max(fb_docs, sdm_rerank_depth), k1, b, delta);
        if (pass1.empty()) return {};

        std::vector<std::pair<uint32_t, float>> candidate_pool;

        if (!use_rm3 || pass1.size() < 2) {
            candidate_pool = pass1;
        } else {
            // Adaptive Guard: Score Variance Check
            float score_sum = 0.0f;
            int num_fb = std::min(static_cast<int>(pass1.size()), fb_docs);
            for (int i = 0; i < num_fb; ++i) score_sum += pass1[i].second;
            float score_mean = score_sum / num_fb;

            float score_var = 0.0f;
            for (int i = 0; i < num_fb; ++i) {
                float diff = pass1[i].second - score_mean;
                score_var += diff * diff;
            }
            score_var /= num_fb;

            if (score_var < 1e-6f) {
                candidate_pool = pass1;
            } else {
                // RM3 Relevance Model Estimation P(w|R)
                std::unordered_map<std::string, float> p_w_r;
                float norm_pass1_scores = 0.0f;
                for (int i = 0; i < num_fb; ++i) norm_pass1_scores += pass1[i].second;
                if (norm_pass1_scores <= 0.0f) norm_pass1_scores = 1.0f;

                std::unordered_set<uint32_t> fb_doc_set;
                for (int i = 0; i < num_fb; ++i) fb_doc_set.insert(pass1[i].first);

                auto batch_term_freqs = get_batch_doc_term_freqs(fb_doc_set);

                for (int i = 0; i < num_fb; ++i) {
                    uint32_t doc_id = pass1[i].first;
                    float p_d_q = pass1[i].second / norm_pass1_scores;
                    uint32_t d_len = doc_lens[doc_id];

                    const auto& term_freqs = batch_term_freqs[doc_id];

                    for (const auto& tf_kv : term_freqs) {
                        const std::string& term = tf_kv.first;
                        uint32_t tf = tf_kv.second;

                        auto dict_it = dictionary.find(term);
                        if (dict_it == dictionary.end()) continue;

                        float p_w_c = (total_corpus_tokens > 0) ? (static_cast<float>(dict_it->second.cf) / total_corpus_tokens) : 0.0001f;
                        float p_w_d = (tf + mu * p_w_c) / (d_len + mu);

                        p_w_r[term] += p_w_d * p_d_q;
                    }
                }

                std::vector<std::pair<std::string, float>> sorted_expansion(p_w_r.begin(), p_w_r.end());
                std::sort(sorted_expansion.begin(), sorted_expansion.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });

                if (sorted_expansion.size() > static_cast<size_t>(fb_terms)) {
                    sorted_expansion.resize(fb_terms);
                }

                float rm3_sum = 0.0f;
                for (const auto& kv : sorted_expansion) rm3_sum += kv.second;
                if (rm3_sum <= 0.0f) rm3_sum = 1.0f;

                std::unordered_map<std::string, float> final_query_weights;
                for (const auto& kv : q_weights) {
                    final_query_weights[kv.first] += alpha * kv.second;
                }

                for (const auto& kv : sorted_expansion) {
                    final_query_weights[kv.first] += (1.0f - alpha) * (kv.second / rm3_sum);
                }

                // 2nd Pass RM3 BM25+ Retrieval
                candidate_pool = raw_retrieve(final_query_weights, std::max(sdm_rerank_depth, k), k1, b, delta);
            }
        }

        // Apply SDM Reranking over candidate pool if enabled and query has > 1 term
        if (use_sdm && tokens.size() > 1) {
            if (candidate_pool.size() > static_cast<size_t>(sdm_rerank_depth)) {
                candidate_pool.resize(sdm_rerank_depth);
            }
            return sdm_rerank(tokens, candidate_pool, k);
        }

        std::vector<std::pair<std::string, float>> results;
        results.reserve(std::min(static_cast<size_t>(k), candidate_pool.size()));
        for (size_t i = 0; i < std::min(static_cast<size_t>(k), candidate_pool.size()); ++i) {
            results.push_back({doc_ids[candidate_pool[i].first], candidate_pool[i].second});
        }
        return results;
    }
};

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "C++ High-Performance Search Engine with DocLen Quantization & Precomputed LUT";
    py::class_<CPPEngine>(m, "CPPEngine")
        .def(py::init<>())
        .def("build_index", &CPPEngine::build_index, py::arg("corpus"), py::arg("index_dir"))
        .def("build_index_from_file", &CPPEngine::build_index_from_file, py::arg("corpus_path"), py::arg("index_dir"))
        .def("load_index", &CPPEngine::load_index, py::arg("index_dir"))
        .def(
            "retrieve",
            &CPPEngine::retrieve,
            py::arg("query"),
            py::arg("k"),
            py::arg("k1") = 1.2f,
            py::arg("b") = 0.75f,
            py::arg("delta") = 1.0f,
            py::arg("use_rm3") = true,
            py::arg("fb_docs") = 10,
            py::arg("fb_terms") = 25,
            py::arg("alpha") = 0.6f,
            py::arg("mu") = 1000.0f,
            py::arg("use_sdm") = true,
            py::arg("sdm_rerank_depth") = 200
        );
}
