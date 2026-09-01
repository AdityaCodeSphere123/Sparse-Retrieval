#include "corpus_io.h"

// Fast path: find "doc_id" / "text" keys; unescape only if a backslash is present.
bool parse_jsonl_sv(std::string_view line, std::string& doc_id, std::string& text) {
    if (line.empty()) return false;
    size_t id_pos = line.find("\"doc_id\"");
    if (id_pos == std::string_view::npos) return false;
    size_t id_colon = line.find(':', id_pos);
    if (id_colon == std::string_view::npos) return false;
    size_t id_quote1 = line.find('"', id_colon);
    if (id_quote1 == std::string_view::npos) return false;
    size_t id_quote2 = line.find('"', id_quote1 + 1);
    while (id_quote2 != std::string_view::npos && id_quote2 > 0 && line[id_quote2 - 1] == '\\') {
        id_quote2 = line.find('"', id_quote2 + 1);
    }
    if (id_quote2 == std::string_view::npos) return false;
    doc_id.assign(line.data() + id_quote1 + 1, id_quote2 - id_quote1 - 1);

    size_t text_pos = line.find("\"text\"");
    if (text_pos == std::string_view::npos) return false;
    size_t text_colon = line.find(':', text_pos);
    if (text_colon == std::string_view::npos) return false;
    size_t text_quote1 = line.find('"', text_colon);
    if (text_quote1 == std::string_view::npos) return false;
    size_t text_quote2 = line.rfind('"');
    if (text_quote2 <= text_quote1) return false;
    while (text_quote2 > text_quote1 && line[text_quote2 - 1] == '\\') {
        size_t prev = line.rfind('"', text_quote2 - 1);
        if (prev <= text_quote1) break;
        text_quote2 = prev;
    }

    std::string_view raw_text(line.data() + text_quote1 + 1, text_quote2 - text_quote1 - 1);
    if (raw_text.find('\\') == std::string_view::npos) {
        text.assign(raw_text.data(), raw_text.size());
        return true;
    }
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

bool parse_jsonl_line(const std::string& line, std::string& doc_id, std::string& text) {
    return parse_jsonl_sv(line, doc_id, text);
}

void collect_jsonl_spans(const char* p, size_t n,
                                std::vector<std::pair<const char*, size_t>>& spans) {
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && p[j] != '\n') ++j;
        size_t len = j - i;
        if (len > 0 && p[i + len - 1] == '\r') --len;
        if (len > 0) spans.emplace_back(p + i, len);
        if (j >= n) break;
        i = j + 1;
    }
}
