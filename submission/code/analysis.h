// Query/index analysis: lowercase, split on non-alnum, fold dotted
// initials (U.S. → us), drop leftover length-1, optional stops, stem.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct TokenizerCfg {
    uint8_t stemmer = 2;     // 0 = none, 1 = S-stemmer, 2 = Porter
    uint8_t remove_stop = 1;
};

extern const std::unordered_set<std::string> STOPWORDS;
extern const std::unordered_set<std::string> INDEX_STOPWORDS;

const std::string& apply_stem(const std::string& w, uint8_t stemmer);

std::vector<std::pair<std::string, uint32_t>>
cpp_tokenize(const std::string& text, const TokenizerCfg& cfg);

void accumulate_tf(
    std::string_view text, const TokenizerCfg& cfg, uint32_t title_window,
    std::unordered_map<std::string, uint32_t>& term_data,
    std::unordered_map<std::string, uint32_t>& title_tf,
    uint32_t& d_len);
