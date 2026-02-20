#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <string>

#include "emel/encoder/ugm/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"

namespace emel::encoder::ugm::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline bool init_xcda_tables(emel::encoder::ugm::action::context &ctx) {
  ctx.xcda_table = nullptr;
  ctx.xcda_table_size = 0;
  ctx.prefix_replacements = nullptr;
  ctx.prefix_replacements_size = 0;
  if (ctx.vocab == nullptr || ctx.vocab->precompiled_charsmap_size == 0) {
    return false;
  }
  const uint8_t *data = ctx.vocab->precompiled_charsmap.data();
  const uint32_t blob_size = *reinterpret_cast<const uint32_t *>(data);
  if (blob_size + sizeof(blob_size) > ctx.vocab->precompiled_charsmap_size) {
    return false;
  }
  ctx.xcda_table = reinterpret_cast<const uint32_t *>(data + sizeof(blob_size));
  ctx.xcda_table_size = blob_size / sizeof(uint32_t);
  ctx.prefix_replacements = reinterpret_cast<const char *>(data + sizeof(blob_size) + blob_size);
  ctx.prefix_replacements_size =
      ctx.vocab->precompiled_charsmap_size - sizeof(blob_size) - blob_size;
  return true;
}

inline bool ensure_ugm_tables(emel::encoder::ugm::action::context &ctx,
                              const emel::model::data::vocab &vocab) {
  if (ctx.ugm_tables_ready && ctx.ugm_vocab == &vocab) {
    return true;
  }
  ctx.ugm_vocab = &vocab;
  ctx.ugm_tables_ready = false;
  ctx.token_matcher = emel::encoder::detail::naive_trie{};
  ctx.user_defined_token_matcher = emel::encoder::detail::naive_trie{};
  ctx.min_score = std::numeric_limits<float>::max();
  ctx.max_score = -std::numeric_limits<float>::max();

  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const auto &entry = vocab.entries[id];
    const std::string_view text = emel::encoder::detail::token_text(vocab, id);
    if (text.empty()) {
      continue;
    }
    const int32_t type = entry.type;
    const bool is_normal = (type == 1);
    const bool is_user_defined = (type == 4);
    const bool is_unused = (type == 5);
    if (is_normal) {
      ctx.min_score = std::min(ctx.min_score, entry.score);
      ctx.max_score = std::max(ctx.max_score, entry.score);
    }
    if (is_normal || is_user_defined || is_unused) {
      ctx.token_matcher.insert(text.data(), text.size(), static_cast<int32_t>(id));
    }
    if (is_user_defined) {
      ctx.user_defined_token_matcher.insert(text.data(), text.size(),
                                            static_cast<int32_t>(id));
    }
  }

  if (ctx.min_score == std::numeric_limits<float>::max()) {
    ctx.min_score = 0.0f;
  }
  ctx.unknown_token_score = ctx.min_score - ctx.unknown_token_score_penalty;
  init_xcda_tables(ctx);
  ctx.ugm_tables_ready = true;
  return true;
}

struct xcda_view {
  const uint32_t *table = nullptr;
  size_t table_size = 0;

  bool valid_index(const size_t index) const {
    return index < table_size;
  }

  uint32_t node(const size_t index) const {
    if (!valid_index(index)) {
      return 0;
    }
    return table[index];
  }

  uint32_t get_base(const size_t index) const {
    const uint32_t packed = node(index);
    return (packed >> 10) << ((packed & (1U << 9)) >> 6);
  }

  uint32_t get_lcheck(const size_t index) const {
    return node(index) & ((1U << 31) | 0xFFu);
  }

  bool get_leaf(const size_t index) const {
    return (node(index) >> 8) & 1U;
  }

  uint32_t get_value(const size_t index) const {
    return node(index) & ((1U << 31) - 1U);
  }
};

struct normalization_result {
  const char *normalized = nullptr;
  size_t normalized_len = 0;
  size_t consumed_input = 0;
};

inline size_t trie_longest_prefix(const emel::encoder::detail::naive_trie &trie,
                                  const char *text,
                                  const size_t len) {
  if (len == 0) {
    return 0;
  }
  const auto *node = trie.traverse(text[0]);
  if (node == nullptr) {
    return 0;
  }
  size_t matched = 0;
  size_t offset = 1;
  if (node->has_value) {
    matched = 1;
  }
  while (offset < len && node != nullptr) {
    node = node->traverse(text[offset]);
    offset += 1;
    if (node == nullptr) {
      break;
    }
    if (node->has_value) {
      matched = offset;
    }
  }
  return matched;
}

inline normalization_result normalize_prefix(
    const emel::model::data::vocab &vocab,
    emel::encoder::ugm::action::context &ctx,
    const std::string &input,
    const size_t input_offset) {
  (void)vocab;
  if (input_offset >= input.size()) {
    return {input.data() + input_offset, 0, 0};
  }

  const size_t remaining = input.size() - input_offset;
  const size_t user_len = trie_longest_prefix(
      ctx.user_defined_token_matcher, input.data() + input_offset, remaining);
  if (user_len > 0) {
    return {input.data() + input_offset, user_len, user_len};
  }

  size_t longest_prefix_length = 0;
  size_t longest_prefix_offset = 0;
  if (ctx.xcda_table != nullptr && ctx.xcda_table_size > 0) {
    xcda_view view = {ctx.xcda_table, ctx.xcda_table_size};
    uint32_t node_index = 0;
    if (!view.valid_index(node_index)) {
      longest_prefix_length = 0;
    } else {
      node_index = view.get_base(node_index);
      for (size_t prefix_offset = input_offset; prefix_offset < input.size();
           ++prefix_offset) {
        const unsigned char c = static_cast<unsigned char>(input[prefix_offset]);
        if (c == 0) {
          break;
        }
        node_index ^= c;
        if (!view.valid_index(node_index) ||
            view.get_lcheck(node_index) != c) {
          break;
        }
        const bool is_leaf = view.get_leaf(node_index);
        node_index ^= view.get_base(node_index);
        if (is_leaf) {
          longest_prefix_length = prefix_offset - input_offset + 1;
          longest_prefix_offset = view.get_value(node_index);
        }
      }
    }
  }

  if (longest_prefix_length > 0) {
    if (longest_prefix_offset >= ctx.prefix_replacements_size) {
      return {nullptr, 0, 0};
    }
    const char *replacement = ctx.prefix_replacements + longest_prefix_offset;
    const size_t replacement_len = std::strlen(replacement);
    return {replacement, replacement_len, longest_prefix_length};
  }

  size_t prefix_offset = input_offset;
  try {
    emel::text::unicode_cpt_from_utf8(input, prefix_offset);
    return {input.data() + input_offset, prefix_offset - input_offset,
            prefix_offset - input_offset};
  } catch (const std::invalid_argument &) {
    static const char k_replacement[] = "\xEF\xBF\xBD";
    return {k_replacement, 3, 1};
  }
}

inline bool normalize_ugm_into(const emel::model::data::vocab &vocab,
                               emel::encoder::ugm::action::context &ctx,
                               const std::string_view text,
                               std::string_view &out_view) {
  const std::string input(text);
  const char *space = vocab.escape_whitespaces ? "\xE2\x96\x81" : " ";
  const size_t space_len = vocab.escape_whitespaces ? 3 : 1;
  const bool shall_prepend_space =
      !vocab.treat_whitespace_as_suffix && vocab.add_space_prefix;
  const bool shall_append_space =
      vocab.treat_whitespace_as_suffix && vocab.add_space_prefix;
  const bool shall_merge_spaces = vocab.remove_extra_whitespaces;

  size_t out_len = 0;
  bool is_space_prepended = false;
  bool processing_non_ws = false;

  size_t input_offset = 0;
  while (input_offset < input.size()) {
    normalization_result norm = normalize_prefix(vocab, ctx, input, input_offset);
    if (norm.normalized == nullptr && norm.consumed_input == 0) {
      return false;
    }
    for (size_t i = 0; i < norm.normalized_len; ++i) {
      const char c = norm.normalized[i];
      if (c != ' ') {
        if (!processing_non_ws) {
          processing_non_ws = true;
          if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
            if (out_len + space_len > ctx.scratch.buffer.size()) {
              return false;
            }
            std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
            out_len += space_len;
            is_space_prepended = true;
          }
        }
        if (out_len + 1 > ctx.scratch.buffer.size()) {
          return false;
        }
        ctx.scratch.buffer[out_len++] = c;
      } else {
        if (processing_non_ws) {
          processing_non_ws = false;
        }
        if (!shall_merge_spaces) {
          if (out_len + space_len > ctx.scratch.buffer.size()) {
            return false;
          }
          std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
          out_len += space_len;
        }
      }
    }
    input_offset += norm.consumed_input;
  }

  if (shall_append_space) {
    if (out_len + space_len > ctx.scratch.buffer.size()) {
      return false;
    }
    std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
    out_len += space_len;
  }

  out_view = std::string_view(ctx.scratch.buffer.data(), out_len);
  return true;
}

inline encode_result encode_ugm(const event::encode &ev,
                                emel::encoder::ugm::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  if (!ensure_ugm_tables(ctx, vocab)) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  const int32_t unk_id = (vocab.unk_id != k_token_null)
                             ? vocab.unk_id
                             : emel::encoder::detail::lookup_token(ctx, "<unk>");

  std::string_view normalized;
  if (!normalize_ugm_into(vocab, ctx, ev.text, normalized)) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }
  const size_t input_len = normalized.size();
  if (input_len == 0) {
    result.error = EMEL_OK;
    return result;
  }
  if (input_len >= ctx.best.size()) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  for (size_t i = 0; i <= input_len; ++i) {
    ctx.best[i] = {unk_id, 0u, -std::numeric_limits<double>::max()};
  }
  ctx.best[0] = {unk_id, 0u, 0.0};

  size_t input_offset = 0;
  while (input_offset < input_len) {
    const size_t n_utf8_code_units =
        std::min(static_cast<size_t>(emel::encoder::detail::utf8_len(
                     normalized[input_offset])),
                 input_len - input_offset);
    bool single_codepoint_token_found = false;
    const auto &current_best = ctx.best[input_offset];
    size_t prefix_offset = input_offset;
    const auto *node = ctx.token_matcher.traverse(normalized[prefix_offset]);
    prefix_offset += 1;
    while (prefix_offset <= input_len && node != nullptr) {
      if (node->has_value) {
        if (prefix_offset - input_offset == n_utf8_code_units) {
          single_codepoint_token_found = true;
        }
        const int32_t token_id = node->value;
        const auto &token_data = vocab.entries[static_cast<uint32_t>(token_id)];
        const bool is_user_defined = (token_data.type == 4);
        const double token_score = is_user_defined ? 0.0 : token_data.score;
        const double challenger_score = current_best.score_sum + token_score;
        auto &current_champ = ctx.best[prefix_offset];
        if (challenger_score > current_champ.score_sum) {
          current_champ = {token_id, static_cast<uint32_t>(input_offset), challenger_score};
        }
      }
      if (prefix_offset >= input_len) {
        break;
      }
      node = node->traverse(normalized[prefix_offset]);
      prefix_offset += 1;
    }

    if (!single_codepoint_token_found && unk_id != k_token_null) {
      const double challenger_score = current_best.score_sum + ctx.unknown_token_score;
      const size_t next_offset = input_offset + n_utf8_code_units;
      auto &current_champ = ctx.best[next_offset];
      if (challenger_score > current_champ.score_sum) {
        current_champ = {unk_id, static_cast<uint32_t>(input_offset), challenger_score};
      }
    }

    input_offset += n_utf8_code_units;
  }

  size_t out_count = 0;
  bool is_prev_unknown = false;
  for (auto tokenization = ctx.best[input_len]; ; tokenization = ctx.best[tokenization.input_offset]) {
    const bool is_unknown = tokenization.token_id == unk_id;
    if (!(is_prev_unknown && is_unknown)) {
      if (out_count >= ctx.token_buffer.size()) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      ctx.token_buffer[out_count++] = tokenization.token_id;
    }
    if (tokenization.input_offset == 0) {
      break;
    }
    is_prev_unknown = is_unknown;
  }

  int32_t count = 0;
  for (size_t i = 0; i < out_count; ++i) {
    const int32_t token = ctx.token_buffer[out_count - 1 - i];
    if (!emel::encoder::detail::push_token(ev, token, count)) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::ugm::detail
