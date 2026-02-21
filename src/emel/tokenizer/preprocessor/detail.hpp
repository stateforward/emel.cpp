#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::preprocessor::detail {

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;

inline bool token_type_is_special(const int32_t type) noexcept {
  return type == k_token_type_control || type == k_token_type_user_defined ||
         type == k_token_type_unknown;
}

inline bool token_type_skip_when_no_parse(const int32_t type) noexcept {
  return type == k_token_type_control || type == k_token_type_unknown;
}

inline std::string_view token_text(const emel::model::data::vocab & vocab,
                                   const uint32_t id) {
  if (id >= vocab.n_tokens) {
    return {};
  }
  const auto & entry = vocab.entries[id];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset,
                          entry.text_length);
}

inline bool flag_set(
    const emel::model::data::vocab & vocab,
    const std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const uint32_t id) noexcept {
  if (id >= vocab.n_tokens) {
    return false;
  }
  const uint32_t byte = id >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (id & 7u));
  return (flags[byte] & mask) != 0;
}

inline bool has_lstrip(const emel::model::data::vocab & vocab,
                       const uint32_t id) noexcept {
  return flag_set(vocab, vocab.lstrip_flags, id);
}

inline bool has_rstrip(const emel::model::data::vocab & vocab,
                       const uint32_t id) noexcept {
  return flag_set(vocab, vocab.rstrip_flags, id);
}

inline bool is_special_type(const emel::model::data::vocab & vocab,
                            const uint32_t id) noexcept {
  if (id >= vocab.n_tokens) {
    return false;
  }
  return token_type_is_special(vocab.entries[id].type);
}

inline bool build_special_tokens(special_token_cache & cache,
                                 const emel::model::data::vocab & vocab) {
  if (cache.vocab == &vocab && cache.count > 0) {
    return true;
  }
  cache.vocab = &vocab;
  cache.count = 0;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    if (!is_special_type(vocab, i)) {
      continue;
    }
    const std::string_view text = token_text(vocab, i);
    if (text.empty()) {
      continue;
    }
    if (cache.count >= cache.tokens.size()) {
      return false;
    }
    special_token & entry = cache.tokens[cache.count];
    entry.text = text;
    entry.token = static_cast<int32_t>(i);
    entry.type = vocab.entries[i].type;
    entry.lstrip = has_lstrip(vocab, i);
    entry.rstrip = has_rstrip(vocab, i);
    cache.count += 1;
  }
  std::sort(cache.tokens.begin(),
            cache.tokens.begin() + static_cast<std::ptrdiff_t>(cache.count),
            [](const special_token & a, const special_token & b) {
              return a.text.size() > b.text.size();
            });
  return true;
}

inline bool push_raw_fragment(fragment * out, const size_t capacity,
                              size_t & count, const std::string_view text) {
  if (text.empty()) {
    return true;
  }
  if (count >= capacity) {
    return false;
  }
  fragment & entry = out[count];
  entry.kind = fragment_kind::raw_text;
  entry.text = text;
  entry.token = -1;
  count += 1;
  return true;
}

inline bool push_token_fragment(fragment * out, const size_t capacity,
                                size_t & count, const int32_t token) {
  if (token < 0) {
    return false;
  }
  if (count >= capacity) {
    return false;
  }
  fragment & entry = out[count];
  entry.kind = fragment_kind::token;
  entry.text = {};
  entry.token = token;
  count += 1;
  return true;
}

inline bool partition_with_specials(const std::string_view text,
                                    const special_token_cache & cache,
                                    const bool parse_special,
                                    fragment * fragments_out,
                                    const size_t fragment_capacity,
                                    size_t * fragment_count_out) {
  if (fragments_out == nullptr || fragment_count_out == nullptr) {
    return false;
  }
  *fragment_count_out = 0;
  if (fragment_capacity == 0 || fragment_capacity > k_max_fragments) {
    return false;
  }

  std::array<fragment, k_max_fragments> current_fragments = {};
  size_t current_count = 0;
  if (!push_raw_fragment(current_fragments.data(), fragment_capacity,
                         current_count, text)) {
    return false;
  }

  std::array<fragment, k_max_fragments> next_fragments = {};
  for (size_t token_idx = 0; token_idx < cache.count; ++token_idx) {
    const special_token & token = cache.tokens[token_idx];
    if (token.text.empty()) {
      continue;
    }
    if (!parse_special && token_type_skip_when_no_parse(token.type)) {
      continue;
    }

    size_t next_count = 0;
    for (size_t frag_idx = 0; frag_idx < current_count; ++frag_idx) {
      const fragment & frag = current_fragments[frag_idx];
      if (frag.kind != fragment_kind::raw_text) {
        if (!push_token_fragment(next_fragments.data(), fragment_capacity,
                                 next_count, frag.token)) {
          return false;
        }
        continue;
      }

      const std::string_view raw = frag.text;
      size_t base_offset = 0;
      while (base_offset < raw.size()) {
        const size_t match = raw.find(token.text, base_offset);
        if (match == std::string_view::npos) {
          if (!push_raw_fragment(next_fragments.data(), fragment_capacity,
                                 next_count, raw.substr(base_offset))) {
            return false;
          }
          break;
        }

        size_t left_len = match - base_offset;
        if (token.lstrip) {
          while (left_len > 0 &&
                 std::isspace(static_cast<unsigned char>(
                     raw[base_offset + left_len - 1])) != 0) {
            left_len -= 1;
          }
        }
        if (left_len > 0) {
          if (!push_raw_fragment(next_fragments.data(), fragment_capacity,
                                 next_count, raw.substr(base_offset, left_len))) {
            return false;
          }
        }

        if (!push_token_fragment(next_fragments.data(), fragment_capacity,
                                 next_count, token.token)) {
          return false;
        }

        size_t right_offset = match + token.text.size();
        if (token.rstrip) {
          while (right_offset < raw.size() &&
                 std::isspace(static_cast<unsigned char>(raw[right_offset])) != 0) {
            right_offset += 1;
          }
        }
        base_offset = right_offset;
      }
    }

    current_fragments = next_fragments;
    current_count = next_count;
  }

  for (size_t i = 0; i < current_count; ++i) {
    fragments_out[i] = current_fragments[i];
  }
  *fragment_count_out = current_count;
  return true;
}

}  // namespace emel::tokenizer::preprocessor::detail
