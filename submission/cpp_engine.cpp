// ===========================================================================
// cpp_engine.cpp — from-scratch inverted index + BM25+/RM3/SDM retrieval.
//
// On-disk layout (v2). Everything is little-endian, written by build_*() and
// read back by load_index(); nothing survives between the two processes
// except these files.
//
//   docmap.bin    N, then per doc: uint16 len + external doc_id bytes
//   doclens.bin   N, avg_doc_len, total_tokens, then N x uint8 quantised
//                 length ratios (log-quantised, see quantize_doc_len)
//   dict.bin      terms in sorted order, front-coded, varint fields
//   postings.bin  per term: optional block skip table + gap/varint postings
//   positions.bin per term: gap/varint position lists (selected terms only)
//   fwd.bin       pruned forward index (doc -> top terms), for RM3
//   fwdoff.bin    N+1 x uint32 offsets into fwd.bin
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
#include <cmath>
#include <queue>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cctype>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Tunable build-time constants. These are the knobs worth sweeping for the
// report: each trades index size against ranking quality or latency.
// ---------------------------------------------------------------------------
static constexpr uint32_t BLOCK_SIZE            = 128;   // postings per block
static constexpr uint32_t FWD_TERMS_PER_DOC     = 32;    // RM3 forward index width
static constexpr uint32_t FWD_CANDIDATES        = 64;    // pre-df-filter margin
static constexpr uint32_t MAX_POSITIONS_PER_DOC = 64;    // truncate long position lists
static constexpr float    POSITIONS_DF_FRACTION = 0.05f; // store positions if df <= this * N
static constexpr uint32_t SPIMI_BATCH_DOCS      = 50000;

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
// Tokenisation: lowercase, alphanumeric runs, light S-stemmer (plural strip).
// Must stay identical between index build and query time.
// ---------------------------------------------------------------------------
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
    if (!current.empty()) tokens.push_back(s_stem(current));
    return tokens;
}

// ---------------------------------------------------------------------------
// mmap wrapper: postings/positions/forward index are read straight out of the
// page cache, so load_index() does no parsing of the bulk data at all.
// ---------------------------------------------------------------------------
struct MmapBuffer {
    const uint8_t* data = nullptr;
    size_t length = 0;

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

    ~MmapBuffer() { unmap(); }
};

// ---------------------------------------------------------------------------
// Per-term dictionary entry.
//
// Note there is no stored max *score*: we store the raw statistics that a
// score bound needs (highest tf seen, smallest quantised length seen) so the
// bound can be recomputed for whatever k1/b are passed at query time. Storing
// a precomputed max score bakes one parameter setting into the index and
// silently invalidates pruning as soon as you sweep k1 or b.
// ---------------------------------------------------------------------------
struct TermMeta {
    uint32_t df = 0;
    uint64_t cf = 0;
    uint64_t postings_offset = 0;
    uint64_t positions_offset = 0;
    uint8_t  max_tf = 0;
    uint8_t  min_qlen = 0;
    uint8_t  has_positions = 0;
};

struct DocPosting {
    uint32_t doc_id;
    uint32_t tf;
    std::vector<uint32_t> positions;
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

    // ---- dictionary, held as parallel sorted arrays -------------------------
    // Sorted term array + binary search rather than a hash map: on a corpus
    // with millions of distinct terms this loads faster and uses far less
    // memory, and the array index doubles as the term id used by fwd.bin.
    std::vector<std::string> terms_sorted;
    std::vector<TermMeta> term_metas;

    MmapBuffer postings_mmap;
    MmapBuffer positions_mmap;
    MmapBuffer fwd_mmap;
    MmapBuffer fwdoff_mmap;

    // ---- query-time scratch -------------------------------------------------
    // A dense accumulator plus a dirty list. Scoring touches acc[doc] directly
    // (no hashing) and only the touched entries are reset after each query.
    std::vector<float> acc;
    std::vector<uint32_t> touched;

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
    //   2. MERGE  — k-way merge the runs into postings.bin / positions.bin,
    //               writing dict.bin front-coded in the same sorted pass.
    //   3. FWD    — map the buffered candidate terms to term ids, drop the
    //               ones the df filter rejects, write fwd.bin / fwdoff.bin.
    //
    template <typename Supplier>
    void build_common(Supplier&& next_doc, const std::string& index_dir) {
        postings_mmap.unmap();
        positions_mmap.unmap();
        fwd_mmap.unmap();
        fwdoff_mmap.unmap();

        doc_ids.clear();
        doc_lens.clear();
        total_corpus_tokens = 0;
        N = 0;

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
                    uint32_t num_pos = p.positions.size();
                    f.write(reinterpret_cast<const char*>(&num_pos), sizeof(num_pos));
                    f.write(reinterpret_cast<const char*>(p.positions.data()), num_pos * sizeof(uint32_t));
                }
            }
            temp_postings.clear();
            run_files.push_back(run_path);
        };

        std::string ext_id, text;
        uint32_t doc_int_id = 0;

        while (next_doc(ext_id, text)) {
            doc_ids.push_back(ext_id);

            auto tokens = cpp_tokenize(text);
            uint32_t d_len = tokens.size();
            doc_lens.push_back(d_len);
            total_corpus_tokens += d_len;

            // tf is counted in full; the position list is capped separately,
            // so truncating positions never corrupts the term frequency.
            std::unordered_map<std::string, std::pair<uint32_t, std::vector<uint32_t>>> term_data;
            for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
                const auto& tok = tokens[pos];
                auto& entry = term_data[tok];
                entry.first++;
                if (entry.second.size() < MAX_POSITIONS_PER_DOC) entry.second.push_back(pos);
                collection_freq[tok]++;
            }

            // Forward-index candidates: the highest-tf non-stopword terms in
            // this document. Written now, resolved to term ids after merge.
            std::vector<std::pair<uint32_t, const std::string*>> cands;
            cands.reserve(term_data.size());
            for (const auto& kv : term_data) {
                if (STOPWORDS.count(kv.first)) continue;
                cands.push_back({kv.second.first, &kv.first});
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

            for (auto& kv : term_data) {
                temp_postings[kv.first].push_back(
                    DocPosting{doc_int_id, kv.second.first, std::move(kv.second.second)});
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

        // -------- docmap.bin / doclens.bin ---------------------------------
        {
            std::ofstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.write(reinterpret_cast<const char*>(&N), sizeof(N));
            for (const auto& id : doc_ids) {
                uint16_t len = id.size();
                f.write(reinterpret_cast<const char*>(&len), sizeof(len));
                f.write(id.data(), len);
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
        std::ofstream f_positions(index_dir + "/positions.bin", std::ios::binary);
        std::ofstream f_dict(index_dir + "/dict.bin", std::ios::binary | std::ios::out);

        uint32_t placeholder = 0;
        f_dict.write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));

        uint64_t current_postings_offset = 0;
        uint64_t current_positions_offset = 0;
        uint64_t prev_postings_offset = 0;
        uint64_t prev_positions_offset = 0;
        std::string prev_term;
        uint32_t num_written_terms = 0;

        const uint32_t positions_df_cap =
            static_cast<uint32_t>(POSITIONS_DF_FRACTION * static_cast<float>(N));

        std::vector<uint8_t> dict_buf;
        dict_buf.reserve(1 << 20);

        std::vector<DocPosting> merged;
        std::vector<uint8_t> block_data, pos_data, hdr_buf;

        while (true) {
            const std::string* min_term = nullptr;
            for (auto& r : readers) {
                if (!r.active) continue;
                if (!min_term || r.curr_term < *min_term) min_term = &r.curr_term;
            }
            if (!min_term) break;
            std::string term = *min_term;

            // df is known before we read any postings, so we can decide up
            // front whether positions are worth keeping and skip reading them
            // entirely for very common terms.
            uint32_t df = 0;
            for (auto& r : readers) if (r.active && r.curr_term == term) df += r.curr_df;
            uint8_t has_positions = (df <= positions_df_cap) ? 1 : 0;

            merged.clear();
            merged.reserve(df);
            uint64_t merged_cf = 0;

            for (auto& r : readers) {
                if (!r.active || r.curr_term != term) continue;
                for (uint32_t p_idx = 0; p_idx < r.curr_df; ++p_idx) {
                    DocPosting dp;
                    r.file.read(reinterpret_cast<char*>(&dp.doc_id), sizeof(dp.doc_id));
                    r.file.read(reinterpret_cast<char*>(&dp.tf), sizeof(dp.tf));
                    uint32_t num_pos;
                    r.file.read(reinterpret_cast<char*>(&num_pos), sizeof(num_pos));
                    if (has_positions) {
                        dp.positions.resize(num_pos);
                        r.file.read(reinterpret_cast<char*>(dp.positions.data()),
                                    num_pos * sizeof(uint32_t));
                    } else {
                        r.file.seekg(static_cast<std::streamoff>(num_pos) * sizeof(uint32_t),
                                     std::ios::cur);
                    }
                    merged_cf += dp.tf;
                    merged.push_back(std::move(dp));
                }
                r.read_next_term();
            }

            // Runs are produced in increasing doc_id order and merged in run
            // order, so `merged` is already sorted by doc_id.
            uint32_t num_blocks = (df + BLOCK_SIZE - 1) / BLOCK_SIZE;
            uint64_t term_postings_start = current_postings_offset;
            uint64_t term_positions_start = current_positions_offset;

            uint8_t global_max_tf = 0;
            uint8_t global_min_qlen = 255;
            uint32_t prev_block_last_doc = 0;

            for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
                uint32_t start_i = b_idx * BLOCK_SIZE;
                uint32_t end_i = std::min(start_i + BLOCK_SIZE, df);

                block_data.clear();
                pos_data.clear();

                uint8_t block_max_tf = 0;
                uint8_t block_min_qlen = 255;
                uint32_t prev_doc_id = (b_idx == 0) ? 0 : prev_block_last_doc;
                uint32_t pos_prev_doc = prev_doc_id;

                for (uint32_t i = start_i; i < end_i; ++i) {
                    uint32_t d_id = merged[i].doc_id;
                    uint32_t tf = merged[i].tf;

                    uint8_t tf_byte = static_cast<uint8_t>(std::min(tf, 255u));
                    if (tf_byte > block_max_tf) block_max_tf = tf_byte;
                    uint8_t ql = quantized_doc_lens[d_id];
                    if (ql < block_min_qlen) block_min_qlen = ql;

                    write_vbyte(block_data, d_id - prev_doc_id);
                    block_data.push_back(tf_byte);
                    prev_doc_id = d_id;

                    if (has_positions) {
                        write_vbyte(pos_data, d_id - pos_prev_doc);
                        pos_prev_doc = d_id;
                        write_vbyte(pos_data, static_cast<uint32_t>(merged[i].positions.size()));
                        uint32_t prev_pos = 0;
                        for (uint32_t p : merged[i].positions) {
                            write_vbyte(pos_data, p - prev_pos);
                            prev_pos = p;
                        }
                    }
                }

                if (block_max_tf > global_max_tf) global_max_tf = block_max_tf;
                if (block_min_qlen < global_min_qlen) global_min_qlen = block_min_qlen;

                // Skip-table entry, written only when the term actually spans
                // more than one block. A single-block term (the overwhelming
                // majority, by Zipf) pays zero header bytes: df in the
                // dictionary already tells us how many postings to decode.
                if (num_blocks > 1) {
                    hdr_buf.clear();
                    write_vbyte(hdr_buf, merged[end_i - 1].doc_id - prev_block_last_doc);
                    hdr_buf.push_back(block_max_tf);
                    hdr_buf.push_back(block_min_qlen);
                    write_vbyte(hdr_buf, static_cast<uint32_t>(block_data.size()));
                    if (has_positions) write_vbyte(hdr_buf, static_cast<uint32_t>(pos_data.size()));
                    f_postings.write(reinterpret_cast<const char*>(hdr_buf.data()), hdr_buf.size());
                    current_postings_offset += hdr_buf.size();
                }

                prev_block_last_doc = merged[end_i - 1].doc_id;

                f_postings.write(reinterpret_cast<const char*>(block_data.data()), block_data.size());
                current_postings_offset += block_data.size();

                if (has_positions && !pos_data.empty()) {
                    f_positions.write(reinterpret_cast<const char*>(pos_data.data()), pos_data.size());
                    current_positions_offset += pos_data.size();
                }
            }

            // -------- dict entry, front-coded against the previous term ----
            size_t prefix = 0;
            size_t maxp = std::min(prev_term.size(), term.size());
            while (prefix < maxp && prev_term[prefix] == term[prefix]) prefix++;
            if (prefix > 255) prefix = 255;   // keep the shared prefix in one byte

            write_vbyte(dict_buf, static_cast<uint32_t>(prefix));
            write_vbyte(dict_buf, static_cast<uint32_t>(term.size() - prefix));
            dict_buf.insert(dict_buf.end(), term.begin() + prefix, term.end());
            write_vbyte(dict_buf, df);
            write_vbyte64(dict_buf, merged_cf);
            dict_buf.push_back(global_max_tf);
            dict_buf.push_back(global_min_qlen);
            write_vbyte64(dict_buf, term_postings_start - prev_postings_offset);
            dict_buf.push_back(has_positions);
            if (has_positions) {
                write_vbyte64(dict_buf, term_positions_start - prev_positions_offset);
                prev_positions_offset = term_positions_start;
            }
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
        f_postings.close();
        f_positions.close();

        for (auto& r : readers) r.file.close();
        for (const auto& rfile : run_files) std::remove(rfile.c_str());

        // -------- stage 3: forward index -----------------------------------
        // We need the dictionary in memory to resolve term ids and apply the
        // df filter, so load what we just wrote and then stream fwdtmp.bin.
        load_dictionary(index_dir);
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
    // Forward index: doc -> its top FWD_TERMS_PER_DOC terms (term id + tf),
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
            df_min, static_cast<uint32_t>(POSITIONS_DF_FRACTION * static_cast<float>(N)));

        uint32_t offset = 0;
        f_off.write(reinterpret_cast<const char*>(&offset), sizeof(offset));

        std::vector<uint8_t> out;
        std::vector<std::pair<uint32_t, uint32_t>> kept;   // (term_id, tf)
        std::string term;

        for (uint32_t d = 0; d < N; ++d) {
            uint32_t n_cand = 0;
            f_in.read(reinterpret_cast<char*>(&n_cand), sizeof(n_cand));
            kept.clear();

            for (uint32_t i = 0; i < n_cand; ++i) {
                uint16_t tl, tf16;
                f_in.read(reinterpret_cast<char*>(&tl), sizeof(tl));
                term.resize(tl);
                f_in.read(&term[0], tl);
                f_in.read(reinterpret_cast<char*>(&tf16), sizeof(tf16));

                if (kept.size() >= FWD_TERMS_PER_DOC) continue;
                int32_t tid = find_term(term);
                if (tid < 0) continue;
                uint32_t df = term_metas[tid].df;
                if (df < df_min || df > df_max) continue;
                kept.push_back({static_cast<uint32_t>(tid), tf16});
            }

            std::sort(kept.begin(), kept.end());

            out.clear();
            write_vbyte(out, static_cast<uint32_t>(kept.size()));
            uint32_t prev_tid = 0;
            for (const auto& kv : kept) {
                write_vbyte(out, kv.first - prev_tid);
                write_vbyte(out, kv.second);
                prev_tid = kv.first;
            }
            f_fwd.write(reinterpret_cast<const char*>(out.data()), out.size());
            offset += static_cast<uint32_t>(out.size());
            f_off.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        }
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
        uint64_t prev_positions_offset = 0;

        for (uint32_t i = 0; i < num_terms; ++i) {
            uint32_t prefix = read_vbyte(p);
            uint32_t suffix_len = read_vbyte(p);
            std::string term = prev_term.substr(0, prefix);
            term.append(reinterpret_cast<const char*>(p), suffix_len);
            p += suffix_len;

            TermMeta m;
            m.df = read_vbyte(p);
            m.cf = read_vbyte64(p);
            m.max_tf = *p++;
            m.min_qlen = *p++;
            m.postings_offset = prev_postings_offset + read_vbyte64(p);
            prev_postings_offset = m.postings_offset;
            m.has_positions = *p++;
            if (m.has_positions) {
                m.positions_offset = prev_positions_offset + read_vbyte64(p);
                prev_positions_offset = m.positions_offset;
            }

            terms_sorted.push_back(term);
            term_metas.push_back(m);
            prev_term = std::move(term);
        }
    }

    void load_index(const std::string& index_dir) {
        {
            std::ifstream f(index_dir + "/docmap.bin", std::ios::binary);
            f.read(reinterpret_cast<char*>(&N), sizeof(N));
            doc_ids.assign(N, std::string());
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

            // Approximate exact lengths back out of the quantised ratios; RM3's
            // Dirichlet smoothing only needs them to within a few percent.
            doc_lens.resize(N);
            for (uint32_t i = 0; i < N; ++i) {
                doc_lens[i] = static_cast<uint32_t>(dequantize_doc_len(quantized_doc_lens[i]) * avg_doc_len);
                if (doc_lens[i] == 0) doc_lens[i] = 1;
            }
        }

        load_dictionary(index_dir);

        postings_mmap.map_file(index_dir + "/postings.bin");
        positions_mmap.map_file(index_dir + "/positions.bin");
        fwd_mmap.map_file(index_dir + "/fwd.bin");
        fwdoff_mmap.map_file(index_dir + "/fwdoff.bin");

        acc.assign(N, 0.0f);
        touched.clear();
        touched.reserve(1 << 16);
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
        int top_k, float k1, float b, float delta)
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

            uint32_t num_blocks = (meta.df + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const uint8_t* ptr = postings_mmap.data + meta.postings_offset;
            uint32_t current_doc = 0;
            uint32_t remaining = meta.df;

            for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
                uint32_t n_in_block = std::min(BLOCK_SIZE, remaining);
                remaining -= n_in_block;

                if (num_blocks > 1) {          // consume the skip-table entry
                    read_vbyte(ptr);           // last doc id gap
                    ptr += 2;                  // max_tf, min_qlen
                    read_vbyte(ptr);           // payload byte size
                    if (meta.has_positions) read_vbyte(ptr);
                }

                for (uint32_t i = 0; i < n_in_block; ++i) {
                    current_doc += read_vbyte(ptr);
                    uint8_t tf = *ptr++;

                    float tf_norm = (tf * (k1 + 1.0f)) /
                                    (tf + length_penalty_lut[quantized_doc_lens[current_doc]]);
                    float contrib = w * (tf_norm + delta);

                    if (acc[current_doc] == 0.0f) touched.push_back(current_doc);
                    acc[current_doc] += contrib;
                }
            }
        }

        size_t want = std::min(static_cast<size_t>(std::max(top_k, 0)), touched.size());
        std::partial_sort(touched.begin(), touched.begin() + want, touched.end(),
                          [this](uint32_t a, uint32_t x) {
                              if (acc[a] != acc[x]) return acc[a] > acc[x];
                              return a < x;   // deterministic tie-break
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
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>
    get_batch_doc_term_freqs(const std::vector<uint32_t>& target_docs) {
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> out;
        if (!fwd_mmap.data || !fwdoff_mmap.data) return out;

        const uint32_t* offs = reinterpret_cast<const uint32_t*>(fwdoff_mmap.data);
        for (uint32_t d : target_docs) {
            if (d >= N) continue;
            const uint8_t* p = fwd_mmap.data + offs[d];
            uint32_t cnt = read_vbyte(p);
            std::vector<std::pair<uint32_t, uint32_t>> v;
            v.reserve(cnt);
            uint32_t tid = 0;
            for (uint32_t i = 0; i < cnt; ++i) {
                tid += read_vbyte(p);
                uint32_t tf = read_vbyte(p);
                v.push_back({tid, tf});
            }
            out[d] = std::move(v);
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Positions for one (term, doc), used by the SDM proximity rerank.
    // Walks the skip table to find the block containing target_doc_id, then
    // decodes only that block's position payload.
    // -----------------------------------------------------------------------
    std::vector<uint32_t> get_positions(int32_t tid, uint32_t target_doc_id) {
        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) return {};
        const TermMeta& meta = term_metas[tid];
        if (!meta.has_positions || !positions_mmap.data || !postings_mmap.data) return {};

        uint32_t num_blocks = (meta.df + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint8_t* ptr = postings_mmap.data + meta.postings_offset;
        uint64_t pos_off = meta.positions_offset;
        uint32_t block_first_doc = 0;   // running doc id at the start of a block
        uint32_t last_doc = 0;
        uint32_t remaining = meta.df;

        for (uint32_t b_idx = 0; b_idx < num_blocks; ++b_idx) {
            uint32_t n_in_block = std::min(BLOCK_SIZE, remaining);
            remaining -= n_in_block;

            uint32_t payload_bytes = 0, pos_bytes = 0;
            if (num_blocks > 1) {
                last_doc += read_vbyte(ptr);
                ptr += 2;
                payload_bytes = read_vbyte(ptr);
                pos_bytes = read_vbyte(ptr);
            } else {
                last_doc = 0xFFFFFFFFu;   // single block necessarily contains it
            }

            if (last_doc >= target_doc_id) {
                const uint8_t* pp = positions_mmap.data + pos_off;
                uint32_t doc = block_first_doc;
                for (uint32_t i = 0; i < n_in_block; ++i) {
                    doc += read_vbyte(pp);
                    uint32_t num_pos = read_vbyte(pp);
                    if (doc == target_doc_id) {
                        std::vector<uint32_t> res;
                        res.reserve(num_pos);
                        uint32_t cur = 0;
                        for (uint32_t j = 0; j < num_pos; ++j) {
                            cur += read_vbyte(pp);
                            res.push_back(cur);
                        }
                        return res;
                    }
                    if (doc > target_doc_id) return {};
                    for (uint32_t j = 0; j < num_pos; ++j) read_vbyte(pp);
                }
                return {};
            }

            ptr += payload_bytes;
            pos_off += pos_bytes;
            block_first_doc = last_doc;
        }
        return {};
    }

    // -----------------------------------------------------------------------
    // Sequential Dependence Model rerank: unigram score plus ordered-bigram
    // and unordered-window proximity evidence over the candidate pool.
    // -----------------------------------------------------------------------
    std::vector<std::pair<std::string, float>> sdm_rerank(
        const std::vector<int32_t>& query_tids,
        const std::vector<std::pair<uint32_t, float>>& candidates,
        int top_k,
        float lambda_t, float lambda_o, float lambda_u, uint32_t window_u)
    {
        if (candidates.empty()) return {};

        std::vector<std::pair<int32_t, int32_t>> bigrams;
        for (size_t i = 0; i + 1 < query_tids.size(); ++i) {
            if (query_tids[i] >= 0 && query_tids[i + 1] >= 0)
                bigrams.push_back({query_tids[i], query_tids[i + 1]});
        }

        std::vector<std::pair<uint32_t, float>> reranked;
        reranked.reserve(candidates.size());

        for (const auto& cand : candidates) {
            uint32_t doc_id = cand.first;
            float unigram_score = cand.second;
            float ordered_score = 0.0f, unordered_score = 0.0f;

            for (const auto& bg : bigrams) {
                auto pos1 = get_positions(bg.first, doc_id);
                if (pos1.empty()) continue;
                auto pos2 = get_positions(bg.second, doc_id);
                if (pos2.empty()) continue;

                uint32_t ordered_cnt = 0, unordered_cnt = 0;
                size_t i1 = 0, i2 = 0;
                while (i1 < pos1.size() && i2 < pos2.size()) {
                    long diff = static_cast<long>(pos2[i2]) - static_cast<long>(pos1[i1]);
                    if (diff == 1) { ordered_cnt++; unordered_cnt++; i1++; i2++; }
                    else if (diff > 1 && diff <= static_cast<long>(window_u)) { unordered_cnt++; i2++; }
                    else if (diff > static_cast<long>(window_u)) { i1++; }
                    else if (diff < 0 && -diff <= static_cast<long>(window_u)) { unordered_cnt++; i1++; }
                    else { i2++; }
                }
                if (ordered_cnt > 0) ordered_score += std::log(1.0f + ordered_cnt);
                if (unordered_cnt > 0) unordered_score += std::log(1.0f + unordered_cnt);
            }

            reranked.push_back({doc_id,
                lambda_t * unigram_score + lambda_o * ordered_score + lambda_u * unordered_score});
        }

        size_t want = std::min(static_cast<size_t>(std::max(top_k, 0)), reranked.size());
        std::partial_sort(reranked.begin(), reranked.begin() + want, reranked.end(),
                          [](const auto& a, const auto& b) {
                              if (a.second != b.second) return a.second > b.second;
                              return a.first < b.first;
                          });

        std::vector<std::pair<std::string, float>> results;
        results.reserve(want);
        for (size_t i = 0; i < want; ++i)
            results.push_back({doc_ids[reranked[i].first], reranked[i].second});
        return results;
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
        bool use_rm3 = true,
        int fb_docs = 10,
        int fb_terms = 25,
        float alpha = 0.6f,
        float mu = 1000.0f,
        bool use_sdm = true,
        int sdm_rerank_depth = 200,
        float lambda_o = 0.10f,
        float lambda_u = 0.05f,
        uint32_t window_u = 8)
    {
        auto tokens = cpp_tokenize(query);
        if (tokens.empty() || N == 0) return {};

        std::vector<int32_t> query_tids;
        query_tids.reserve(tokens.size());
        std::unordered_map<int32_t, float> qw;
        for (const auto& t : tokens) {
            int32_t tid = find_term(t);
            query_tids.push_back(tid);
            if (tid >= 0) qw[tid] += 1.0f;
        }
        if (qw.empty()) return {};
        for (auto& kv : qw) kv.second /= static_cast<float>(tokens.size());

        std::vector<std::pair<int32_t, float>> q_vec(qw.begin(), qw.end());

        int pass1_depth = std::max(fb_docs, use_sdm ? sdm_rerank_depth : k);
        auto pass1 = raw_retrieve(q_vec, pass1_depth, k1, b, delta);
        if (pass1.empty()) return {};

        std::vector<std::pair<uint32_t, float>> pool;

        if (!use_rm3 || pass1.size() < 2) {
            pool = pass1;
        } else {
            int num_fb = std::min(static_cast<int>(pass1.size()), fb_docs);

            // If the top results are indistinguishable, feedback has nothing
            // to learn from; fall back to the first-pass ranking.
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

                std::vector<uint32_t> fb_docs_vec;
                fb_docs_vec.reserve(num_fb);
                for (int i = 0; i < num_fb; ++i) fb_docs_vec.push_back(pass1[i].first);
                auto batch = get_batch_doc_term_freqs(fb_docs_vec);

                // RM3 relevance model: P(w|R) = sum_D P(w|D) P(D|Q), with
                // P(w|D) Dirichlet-smoothed against the collection model.
                std::unordered_map<int32_t, float> p_w_r;
                for (int i = 0; i < num_fb; ++i) {
                    uint32_t doc_id = pass1[i].first;
                    float p_d_q = pass1[i].second / norm;
                    float d_len = static_cast<float>(doc_lens[doc_id]);

                    auto it = batch.find(doc_id);
                    if (it == batch.end()) continue;
                    for (const auto& tf_kv : it->second) {
                        int32_t tid = static_cast<int32_t>(tf_kv.first);
                        if (tid < 0 || tid >= static_cast<int32_t>(term_metas.size())) continue;
                        float p_w_c = (total_corpus_tokens > 0)
                            ? (static_cast<float>(term_metas[tid].cf) / total_corpus_tokens)
                            : 1e-4f;
                        float p_w_d = (tf_kv.second + mu * p_w_c) / (d_len + mu);
                        p_w_r[tid] += p_w_d * p_d_q;
                    }
                }

                std::vector<std::pair<int32_t, float>> expansion(p_w_r.begin(), p_w_r.end());
                size_t keep = std::min(static_cast<size_t>(std::max(fb_terms, 0)), expansion.size());
                std::partial_sort(expansion.begin(), expansion.begin() + keep, expansion.end(),
                                  [](const auto& a, const auto& b) { return a.second > b.second; });
                expansion.resize(keep);

                float esum = 0.0f;
                for (const auto& kv : expansion) esum += kv.second;
                if (esum <= 0.0f) esum = 1.0f;

                std::unordered_map<int32_t, float> final_w;
                for (const auto& kv : qw) final_w[kv.first] += alpha * kv.second;
                for (const auto& kv : expansion) final_w[kv.first] += (1.0f - alpha) * (kv.second / esum);

                std::vector<std::pair<int32_t, float>> final_vec(final_w.begin(), final_w.end());
                pool = raw_retrieve(final_vec, std::max(use_sdm ? sdm_rerank_depth : k, k),
                                    k1, b, delta);
            }
        }

        if (use_sdm && query_tids.size() > 1) {
            if (pool.size() > static_cast<size_t>(sdm_rerank_depth)) pool.resize(sdm_rerank_depth);
            float lambda_t = 1.0f - lambda_o - lambda_u;
            return sdm_rerank(query_tids, pool, k, lambda_t, lambda_o, lambda_u, window_u);
        }

        std::vector<std::pair<std::string, float>> results;
        size_t want = std::min(static_cast<size_t>(std::max(k, 0)), pool.size());
        results.reserve(want);
        for (size_t i = 0; i < want; ++i) results.push_back({doc_ids[pool[i].first], pool[i].second});
        return results;
    }

    // -----------------------------------------------------------------------
    // Plain BM25+ with no RM3 and no SDM. Needed for the report's ablation
    // table and for the unit tests that check BM25 in isolation.
    // -----------------------------------------------------------------------
    std::vector<std::pair<std::string, float>> retrieve_bm25(
        const std::string& query, int k, float k1, float b, float delta)
    {
        return retrieve(query, k, k1, b, delta, false, 0, 0, 1.0f, 1000.0f, false, 0);
    }
};

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "From-scratch inverted index with BM25+, RM3 expansion and SDM proximity rerank";
    py::class_<CPPEngine>(m, "CPPEngine")
        .def(py::init<>())
        .def("build_index", &CPPEngine::build_index, py::arg("corpus"), py::arg("index_dir"))
        .def("build_index_from_file", &CPPEngine::build_index_from_file,
             py::arg("corpus_path"), py::arg("index_dir"))
        .def("load_index", &CPPEngine::load_index, py::arg("index_dir"))
        .def("retrieve", &CPPEngine::retrieve,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("use_rm3") = true, py::arg("fb_docs") = 10, py::arg("fb_terms") = 25,
             py::arg("alpha") = 0.6f, py::arg("mu") = 1000.0f,
             py::arg("use_sdm") = true, py::arg("sdm_rerank_depth") = 200,
             py::arg("lambda_o") = 0.10f, py::arg("lambda_u") = 0.05f, py::arg("window_u") = 8)
        .def("retrieve_bm25", &CPPEngine::retrieve_bm25,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f);
}