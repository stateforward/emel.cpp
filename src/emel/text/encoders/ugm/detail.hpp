#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "emel/text/encoders/ugm/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::ugm::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

inline int32_t select_i32(const bool choose_true,
                          const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true,
                           const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask = static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline float select_f32(const bool choose_true,
                        const float true_value,
                        const float false_value) noexcept {
  const std::array<float, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline size_t ugm_utf8_len(const char byte) noexcept {
  constexpr std::array<size_t, 16> lookup{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  const uint8_t highbits = static_cast<uint8_t>(byte) >> 4u;
  return lookup[highbits];
}

inline std::string_view ugm_token_text(const emel::model::data::vocab &vocab,
                                       const int32_t id) noexcept {
  std::string_view text{};
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  for (bool read_entry = valid_id; read_entry; read_entry = false) {
    const auto &entry = vocab.entries[static_cast<uint32_t>(id)];
    const bool has_text = entry.text_length > 0;
    for (bool assign = has_text; assign; assign = false) {
      text = std::string_view(vocab.token_storage.data() + entry.text_offset, entry.text_length);
    }
  }
  return text;
}

inline bool ugm_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
  const bool token_ok = token >= 0;
  const bool count_ok = count >= 0;
  const size_t slot = select_size(count_ok, static_cast<size_t>(count), static_cast<size_t>(0));
  const bool output_ok = !ev.token_ids.empty();
  const bool room_ok = slot < ev.token_ids.size();
  const bool can_write = token_ok && count_ok && output_ok && room_ok;
  for (bool write = can_write; write; write = false) {
    ev.token_ids[slot] = token;
    count += 1;
  }
  return can_write;
}

inline void ugm_trie_insert(emel::text::encoders::detail::naive_trie &trie,
                            const char *text,
                            const size_t len,
                            const int32_t value) noexcept {
  size_t idx = 0;
  for (size_t i = 0; i < len; ++i) {
    auto &node = trie.nodes[idx];
    const uint8_t byte = static_cast<uint8_t>(text[i]);
    const bool missing = node.next[byte] < 0;
    for (bool grow = missing; grow; grow = false) {
      node.next[byte] = static_cast<int32_t>(trie.nodes.size());
      trie.nodes.emplace_back();
      trie.nodes.back().nodes_ref = &trie.nodes;
    }
    idx = static_cast<size_t>(node.next[byte]);
  }
  trie.nodes[idx].has_value = true;
  trie.nodes[idx].value = value;
}

inline const emel::text::encoders::detail::naive_trie::node *ugm_trie_root(
  const emel::text::encoders::detail::naive_trie &trie,
  const char c) noexcept {
  const int32_t idx = trie.nodes[0].next[static_cast<uint8_t>(c)];
  const bool valid = idx >= 0 && static_cast<size_t>(idx) < trie.nodes.size();
  const size_t safe_idx = select_size(valid, static_cast<size_t>(idx), static_cast<size_t>(0));
  const auto *candidate = &trie.nodes[safe_idx];
  const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
    nullptr,
    candidate,
  };
  return options[static_cast<size_t>(valid)];
}

inline const emel::text::encoders::detail::naive_trie::node *ugm_trie_step(
  const emel::text::encoders::detail::naive_trie::node &node,
  const char c) noexcept {
  const int32_t idx = node.next[static_cast<uint8_t>(c)];
  const auto &nodes = *node.nodes_ref;
  const bool valid = idx >= 0 && static_cast<size_t>(idx) < nodes.size();
  const size_t safe_idx = select_size(valid, static_cast<size_t>(idx), static_cast<size_t>(0));
  const auto *candidate = &nodes[safe_idx];
  const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
    nullptr,
    candidate,
  };
  return options[static_cast<size_t>(valid)];
}

inline int32_t ugm_lookup_token_exact(const emel::model::data::vocab &vocab,
                                      const std::string_view target) noexcept {
  int32_t resolved = k_token_null;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view token = ugm_token_text(vocab, static_cast<int32_t>(id));
    const bool exact = token == target;
    resolved = select_i32(exact, static_cast<int32_t>(id), resolved);
  }
  return resolved;
}

inline bool init_xcda_tables(emel::text::encoders::ugm::action::context &ctx) noexcept {
  ctx.xcda_table = nullptr;
  ctx.xcda_table_size = 0;
  ctx.prefix_replacements = nullptr;
  ctx.prefix_replacements_size = 0;

  const bool has_vocab = ctx.vocab != nullptr;
  const bool has_blob = has_vocab && ctx.vocab->precompiled_charsmap_size > 0u;
  for (bool missing_blob = !has_blob; missing_blob; missing_blob = false) {
    return false;
  }

  const uint8_t *data = ctx.vocab->precompiled_charsmap.data();
  const uint32_t blob_size = *reinterpret_cast<const uint32_t *>(data);
  const bool bounded = blob_size + static_cast<uint32_t>(sizeof(blob_size)) <=
                       static_cast<uint32_t>(ctx.vocab->precompiled_charsmap_size);
  for (bool invalid_blob = !bounded; invalid_blob; invalid_blob = false) {
    return false;
  }

  ctx.xcda_table = reinterpret_cast<const uint32_t *>(data + sizeof(blob_size));
  ctx.xcda_table_size = blob_size / sizeof(uint32_t);
  ctx.prefix_replacements = reinterpret_cast<const char *>(data + sizeof(blob_size) + blob_size);
  ctx.prefix_replacements_size =
    ctx.vocab->precompiled_charsmap_size - sizeof(blob_size) - blob_size;
  return true;
}

inline bool ugm_tables_ready(const emel::text::encoders::ugm::action::context &ctx,
                             const emel::model::data::vocab &vocab) noexcept {
  return ctx.ugm_tables_ready && ctx.ugm_vocab == &vocab;
}

inline bool ensure_ugm_tables(emel::text::encoders::ugm::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  for (bool already_ready = ugm_tables_ready(ctx, vocab); already_ready; already_ready = false) {
    return true;
  }

  ctx.ugm_vocab = &vocab;
  ctx.ugm_tables_ready = false;
  ctx.token_matcher = emel::text::encoders::detail::naive_trie{};
  ctx.user_defined_token_matcher = emel::text::encoders::detail::naive_trie{};
  ctx.min_score = std::numeric_limits<float>::max();
  ctx.max_score = -std::numeric_limits<float>::max();

  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const auto &entry = vocab.entries[id];
    const std::string_view text = ugm_token_text(vocab, static_cast<int32_t>(id));
    const bool has_text = !text.empty();
    const int32_t type = entry.type;
    const bool is_normal = type == 1;
    const bool is_user_defined = type == 4;
    const bool is_unused = type == 5;
    const bool insert_general = has_text && (is_normal || is_user_defined || is_unused);
    const bool insert_user_defined = has_text && is_user_defined;
    const bool update_min = has_text && is_normal;

    for (bool update = update_min; update; update = false) {
      ctx.min_score = std::min(ctx.min_score, entry.score);
      ctx.max_score = std::max(ctx.max_score, entry.score);
    }
    for (bool insert = insert_general; insert; insert = false) {
      ugm_trie_insert(ctx.token_matcher, text.data(), text.size(), static_cast<int32_t>(id));
    }
    for (bool insert = insert_user_defined; insert; insert = false) {
      ugm_trie_insert(ctx.user_defined_token_matcher, text.data(), text.size(), static_cast<int32_t>(id));
    }
  }

  const bool has_normal_scores = ctx.min_score != std::numeric_limits<float>::max();
  ctx.min_score = select_f32(has_normal_scores, ctx.min_score, 0.0f);
  ctx.unknown_token_score = ctx.min_score - ctx.unknown_token_score_penalty;
  init_xcda_tables(ctx);
  ctx.ugm_tables_ready = true;
  return true;
}

struct xcda_view {
  const uint32_t *table = nullptr;
  size_t table_size = 0;

  bool valid_index(const size_t index) const noexcept {
    return table != nullptr && index < table_size;
  }

  uint32_t node(const size_t index) const noexcept {
    uint32_t zero = 0u;
    const bool has_table = table != nullptr && table_size > 0u;
    const bool valid = has_table && index < table_size;
    const size_t safe_index = select_size(valid, index, static_cast<size_t>(0));
    const uint32_t *sources[2] = {&zero, table};
    const uint32_t *source = sources[static_cast<size_t>(has_table)];
    const uint32_t value = source[safe_index];
    return select_u32(valid, value, 0u);
  }

  uint32_t get_base(const size_t index) const noexcept {
    const uint32_t packed = node(index);
    return (packed >> 10u) << ((packed & (1u << 9u)) >> 6u);
  }

  uint32_t get_lcheck(const size_t index) const noexcept {
    return node(index) & ((1u << 31u) | 0xFFu);
  }

  bool get_leaf(const size_t index) const noexcept {
    return ((node(index) >> 8u) & 1u) != 0u;
  }

  uint32_t get_value(const size_t index) const noexcept {
    return node(index) & ((1u << 31u) - 1u);
  }
};

struct normalization_result {
  const char *normalized = nullptr;
  size_t normalized_len = 0;
  size_t consumed_input = 0;
};

inline size_t trie_longest_prefix(const emel::text::encoders::detail::naive_trie &trie,
                                  const char *text,
                                  const size_t len) noexcept {
  size_t matched = 0;
  for (bool has_input = len > 0u; has_input; has_input = false) {
    const auto *node = ugm_trie_root(trie, text[0]);
    bool walking = node != nullptr;
    size_t offset = 1;
    matched = select_size(walking && node->has_value, static_cast<size_t>(1), matched);
    while (walking && offset < len) {
      node = ugm_trie_step(*node, text[offset]);
      offset += 1u;
      walking = node != nullptr;
      matched = select_size(walking && node->has_value, offset, matched);
    }
  }
  return matched;
}

inline normalization_result normalize_prefix(const emel::model::data::vocab &vocab,
                                             emel::text::encoders::ugm::action::context &ctx,
                                             const std::string_view input,
                                             const size_t input_offset) noexcept {
  (void)vocab;
  for (bool at_end = input_offset >= input.size(); at_end; at_end = false) {
    return {input.data() + input_offset, 0, 0};
  }

  const size_t remaining = input.size() - input_offset;
  const size_t user_len = trie_longest_prefix(
    ctx.user_defined_token_matcher, input.data() + input_offset, remaining);
  for (bool user_hit = user_len > 0u; user_hit; user_hit = false) {
    return {input.data() + input_offset, user_len, user_len};
  }

  size_t longest_prefix_length = 0;
  size_t longest_prefix_offset = 0;

  for (bool has_xcda = ctx.xcda_table != nullptr && ctx.xcda_table_size > 0u;
       has_xcda;
       has_xcda = false) {
    xcda_view view = {ctx.xcda_table, ctx.xcda_table_size};
    bool active = view.valid_index(0);
    uint32_t node_index = select_u32(active, view.get_base(0), 0u);

    for (size_t prefix_offset = input_offset; active && prefix_offset < input.size(); ++prefix_offset) {
      const uint32_t c = static_cast<unsigned char>(input[prefix_offset]);
      const bool non_zero = c != 0u;
      const uint32_t candidate = node_index ^ c;
      const bool valid = active && non_zero && view.valid_index(candidate)
                         && view.get_lcheck(candidate) == c;
      const bool leaf = valid && view.get_leaf(candidate);
      const uint32_t branch = candidate ^ view.get_base(candidate);
      const size_t candidate_length = prefix_offset - input_offset + 1u;
      const size_t candidate_offset = static_cast<size_t>(view.get_value(branch));
      longest_prefix_length = select_size(leaf, candidate_length, longest_prefix_length);
      longest_prefix_offset = select_size(leaf, candidate_offset, longest_prefix_offset);
      node_index = select_u32(valid, branch, node_index);
      active = valid;
    }
  }

  for (bool has_prefix = longest_prefix_length > 0u; has_prefix; has_prefix = false) {
    const bool offset_ok = longest_prefix_offset < ctx.prefix_replacements_size;
    for (bool invalid_offset = !offset_ok; invalid_offset; invalid_offset = false) {
      return {nullptr, 0, 0};
    }
    const char *replacement = ctx.prefix_replacements + longest_prefix_offset;
    const size_t replacement_len = std::strlen(replacement);
    return {replacement, replacement_len, longest_prefix_length};
  }

  constexpr std::array<char, 3> replacement = {'\xEF', '\xBF', '\xBD'};
  const uint8_t first = static_cast<uint8_t>(input[input_offset]);
  const bool continuation = (first & 0xC0u) == 0x80u;
  const size_t len_raw = ugm_utf8_len(static_cast<char>(first));
  const bool bounded = len_raw <= remaining;
  const bool invalid = continuation || !bounded;
  const size_t consumed = select_size(bounded, len_raw, static_cast<size_t>(1));
  for (bool invalid_utf8 = invalid; invalid_utf8; invalid_utf8 = false) {
    return {replacement.data(), replacement.size(), 1};
  }
  return {input.data() + input_offset, consumed, consumed};
}

inline bool normalize_ugm_into(const emel::model::data::vocab &vocab,
                               emel::text::encoders::ugm::action::context &ctx,
                               const std::string_view text,
                               std::string_view &out_view) noexcept {
  const std::string_view input = text;
  constexpr std::array<const char *, 2> spaces = {" ", "\xE2\x96\x81"};
  constexpr std::array<size_t, 2> space_lengths = {1u, 3u};
  const size_t space_idx = static_cast<size_t>(vocab.escape_whitespaces);
  const char *space = spaces[space_idx];
  const size_t space_len = space_lengths[space_idx];
  const bool shall_prepend_space = !vocab.treat_whitespace_as_suffix && vocab.add_space_prefix;
  const bool shall_append_space = vocab.treat_whitespace_as_suffix && vocab.add_space_prefix;
  const bool shall_merge_spaces = vocab.remove_extra_whitespaces;

  size_t out_len = 0;
  bool is_space_prepended = false;
  bool processing_non_ws = false;

  size_t input_offset = 0;
  while (input_offset < input.size()) {
    normalization_result norm = normalize_prefix(vocab, ctx, input, input_offset);
    const bool invalid_norm = norm.normalized == nullptr && norm.consumed_input == 0u;
    for (bool fail_norm = invalid_norm; fail_norm; fail_norm = false) {
      return false;
    }

    for (size_t i = 0; i < norm.normalized_len; ++i) {
      const char c = norm.normalized[i];
      const bool non_space = c != ' ';

      for (bool emit_non_space = non_space; emit_non_space; emit_non_space = false) {
        for (bool begin_non_ws = !processing_non_ws; begin_non_ws; begin_non_ws = false) {
          processing_non_ws = true;
          const bool emit_prefix = (shall_prepend_space && !is_space_prepended) || shall_merge_spaces;
          for (bool write_prefix = emit_prefix; write_prefix; write_prefix = false) {
            const bool has_capacity = out_len + space_len <= ctx.scratch.buffer.size();
            for (bool overflow = !has_capacity; overflow; overflow = false) {
              return false;
            }
            std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
            out_len += space_len;
            is_space_prepended = true;
          }
        }

        const bool has_capacity = out_len + 1u <= ctx.scratch.buffer.size();
        for (bool overflow = !has_capacity; overflow; overflow = false) {
          return false;
        }
        ctx.scratch.buffer[out_len] = c;
        out_len += 1u;
      }

      for (bool emit_space = !non_space; emit_space; emit_space = false) {
        processing_non_ws = false;
        for (bool keep_spaces = !shall_merge_spaces; keep_spaces; keep_spaces = false) {
          const bool has_capacity = out_len + space_len <= ctx.scratch.buffer.size();
          for (bool overflow = !has_capacity; overflow; overflow = false) {
            return false;
          }
          std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
          out_len += space_len;
        }
      }
    }

    input_offset += norm.consumed_input;
  }

  for (bool append_space = shall_append_space; append_space; append_space = false) {
    const bool has_capacity = out_len + space_len <= ctx.scratch.buffer.size();
    for (bool overflow = !has_capacity; overflow; overflow = false) {
      return false;
    }
    std::memcpy(ctx.scratch.buffer.data() + out_len, space, space_len);
    out_len += space_len;
  }

  out_view = std::string_view(ctx.scratch.buffer.data(), out_len);
  return true;
}

inline encode_result encode_ugm(const event::encode &ev,
                                emel::text::encoders::ugm::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  result.token_count = 0;

  for (bool empty_text = ev.text.empty(); empty_text; empty_text = false) {
    result.error = EMEL_OK;
    return result;
  }

  const bool tables_ready = ugm_tables_ready(ctx, vocab);
  for (bool missing_tables = !tables_ready; missing_tables; missing_tables = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  int32_t unk_id = vocab.unk_id;
  for (bool resolve_unk = unk_id == k_token_null; resolve_unk; resolve_unk = false) {
    unk_id = ugm_lookup_token_exact(vocab, "<unk>");
  }

  std::string_view normalized{};
  const bool normalized_ok = normalize_ugm_into(vocab, ctx, ev.text, normalized);
  for (bool normalize_fail = !normalized_ok; normalize_fail; normalize_fail = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  const size_t input_len = normalized.size();
  for (bool no_input = input_len == 0u; no_input; no_input = false) {
    result.error = EMEL_OK;
    return result;
  }
  for (bool overflow = input_len >= ctx.best.size(); overflow; overflow = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  for (size_t i = 0; i <= input_len; ++i) {
    ctx.best[i] = {unk_id, 0u, -std::numeric_limits<double>::max()};
  }
  ctx.best[0] = {unk_id, 0u, 0.0};

  size_t input_offset = 0;
  while (input_offset < input_len) {
    const size_t n_utf8_code_units = std::min(
      static_cast<size_t>(ugm_utf8_len(normalized[input_offset])),
      input_len - input_offset);
    bool single_codepoint_token_found = false;
    const auto current_best = ctx.best[input_offset];
    size_t prefix_offset = input_offset;
    const auto *node = ugm_trie_root(ctx.token_matcher, normalized[prefix_offset]);
    prefix_offset += 1u;
    bool walking = node != nullptr && prefix_offset <= input_len;

    while (walking) {
      for (bool has_value = node->has_value; has_value; has_value = false) {
        const bool single_codepoint = prefix_offset - input_offset == n_utf8_code_units;
        single_codepoint_token_found = single_codepoint_token_found || single_codepoint;
        const int32_t token_id = node->value;
        const auto &token_data = vocab.entries[static_cast<uint32_t>(token_id)];
        const bool is_user_defined = token_data.type == 4;
        const std::array<double, 2> score_table{
          static_cast<double>(token_data.score),
          0.0,
        };
        const double token_score = score_table[static_cast<size_t>(is_user_defined)];
        const double challenger_score = current_best.score_sum + token_score;
        auto &current_champ = ctx.best[prefix_offset];
        for (bool better = challenger_score > current_champ.score_sum; better; better = false) {
          current_champ = {token_id, static_cast<uint32_t>(input_offset), challenger_score};
        }
      }

      const bool can_advance = prefix_offset < input_len;
      const size_t safe_offset = select_size(can_advance, prefix_offset, input_offset);
      const auto *next_node = ugm_trie_step(*node, normalized[safe_offset]);
      const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
        node,
        next_node,
      };
      node = options[static_cast<size_t>(can_advance)];
      prefix_offset += static_cast<size_t>(can_advance);
      walking = can_advance && node != nullptr && prefix_offset <= input_len;
    }

    const bool use_unk = !single_codepoint_token_found && unk_id != k_token_null;
    for (bool update_unk = use_unk; update_unk; update_unk = false) {
      const double challenger_score = current_best.score_sum + static_cast<double>(ctx.unknown_token_score);
      const size_t next_offset = input_offset + n_utf8_code_units;
      auto &current_champ = ctx.best[next_offset];
      for (bool better = challenger_score > current_champ.score_sum; better; better = false) {
        current_champ = {unk_id, static_cast<uint32_t>(input_offset), challenger_score};
      }
    }

    input_offset += n_utf8_code_units;
  }

  size_t out_count = 0;
  bool is_prev_unknown = false;
  emel::text::encoders::ugm::action::best_tokenization tokenization = ctx.best[input_len];
  bool tracing = true;
  while (tracing) {
    const bool is_unknown = tokenization.token_id == unk_id;
    const bool emit_token = !(is_prev_unknown && is_unknown);
    for (bool emit = emit_token; emit; emit = false) {
      const bool has_room = out_count < ctx.token_buffer.size();
      for (bool no_room = !has_room; no_room; no_room = false) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      ctx.token_buffer[out_count] = tokenization.token_id;
      out_count += 1u;
    }

    const bool at_root = tokenization.input_offset == 0u;
    for (bool advance = !at_root; advance; advance = false) {
      is_prev_unknown = is_unknown;
      tokenization = ctx.best[tokenization.input_offset];
    }
    tracing = !at_root;
  }

  int32_t count = 0;
  for (size_t i = 0; i < out_count; ++i) {
    const int32_t token = ctx.token_buffer[out_count - 1u - i];
    const bool pushed = ugm_push_token(ev, token, count);
    for (bool push_fail = !pushed; push_fail; push_fail = false) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::text::encoders::ugm::detail
