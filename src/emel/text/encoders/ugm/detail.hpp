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

inline double select_f64(const bool choose_true,
                         const double true_value,
                         const double false_value) noexcept {
  const std::array<double, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline bool select_bool(const bool choose_true,
                        const bool true_value,
                        const bool false_value) noexcept {
  const std::array<bool, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline size_t ugm_utf8_len(const char byte) noexcept {
  constexpr std::array<size_t, 16> lookup{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  const uint8_t highbits = static_cast<uint8_t>(byte) >> 4u;
  return lookup[highbits];
}

inline std::string_view ugm_token_text(const emel::model::data::vocab &vocab,
                                       const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t safe_id = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[safe_id];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(vocab.token_storage.data() + static_cast<size_t>(offset),
                          static_cast<size_t>(length));
}

inline bool ugm_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t *base_ptrs[2] = {&sink, ev.token_ids.data()};
  int32_t *base = base_ptrs[static_cast<size_t>(has_buffer)];
  const bool non_negative_count = count >= 0;
  const int32_t safe_count = select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space = has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;
  const size_t target_index = count_index * static_cast<size_t>(write);
  int32_t *target = base + target_index;
  *target = select_i32(write, token, *target);
  count += static_cast<int32_t>(write);
  return write;
}

inline void ugm_trie_insert_none(emel::text::encoders::detail::naive_trie::node &,
                                 emel::text::encoders::detail::naive_trie &,
                                 const uint8_t) noexcept {}

inline void ugm_trie_insert_some(emel::text::encoders::detail::naive_trie::node &node,
                                 emel::text::encoders::detail::naive_trie &trie,
                                 const uint8_t byte) noexcept {
  node.next[byte] = static_cast<int32_t>(trie.nodes.size());
  trie.nodes.emplace_back();
  trie.nodes.back().nodes_ref = &trie.nodes;
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
    using trie_insert_handler_t = void (*)(emel::text::encoders::detail::naive_trie::node &,
                                           emel::text::encoders::detail::naive_trie &,
                                           uint8_t) noexcept;
    const trie_insert_handler_t trie_insert_handlers[2] = {
        ugm_trie_insert_none,
        ugm_trie_insert_some,
    };
    trie_insert_handlers[static_cast<size_t>(missing)](node, trie, byte);
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

struct xcda_blob_info {
  const uint8_t *data = nullptr;
  uint32_t blob_size = 0u;
  bool bounded = false;
};

inline void ugm_load_xcda_blob_none(const emel::text::encoders::ugm::action::context &,
                                    xcda_blob_info &) noexcept {}

inline void ugm_load_xcda_blob_some(const emel::text::encoders::ugm::action::context &ctx,
                                    xcda_blob_info &blob) noexcept {
  blob.data = ctx.vocab->precompiled_charsmap.data();
  blob.blob_size = *reinterpret_cast<const uint32_t *>(blob.data);
  blob.bounded = blob.blob_size + static_cast<uint32_t>(sizeof(blob.blob_size)) <=
                 static_cast<uint32_t>(ctx.vocab->precompiled_charsmap_size);
}

inline bool ugm_init_xcda_blob_none(emel::text::encoders::ugm::action::context &,
                                    const xcda_blob_info &) noexcept {
  return false;
}

inline bool ugm_init_xcda_blob_some(emel::text::encoders::ugm::action::context &ctx,
                                    const xcda_blob_info &blob) noexcept {
  ctx.xcda_table = reinterpret_cast<const uint32_t *>(blob.data + sizeof(blob.blob_size));
  ctx.xcda_table_size = blob.blob_size / sizeof(uint32_t);
  ctx.prefix_replacements =
      reinterpret_cast<const char *>(blob.data + sizeof(blob.blob_size) + blob.blob_size);
  ctx.prefix_replacements_size =
      ctx.vocab->precompiled_charsmap_size - sizeof(blob.blob_size) - blob.blob_size;
  return true;
}

inline bool init_xcda_tables(emel::text::encoders::ugm::action::context &ctx) noexcept {
  ctx.xcda_table = nullptr;
  ctx.xcda_table_size = 0;
  ctx.prefix_replacements = nullptr;
  ctx.prefix_replacements_size = 0;

  const bool has_vocab = ctx.vocab != nullptr;
  const bool has_blob = has_vocab && ctx.vocab->precompiled_charsmap_size > 0u;
  xcda_blob_info blob{};
  using load_handler_t = void (*)(const emel::text::encoders::ugm::action::context &,
                                  xcda_blob_info &) noexcept;
  const load_handler_t load_handlers[2] = {
      ugm_load_xcda_blob_none,
      ugm_load_xcda_blob_some,
  };
  load_handlers[static_cast<size_t>(has_blob)](ctx, blob);

  using init_handler_t =
      bool (*)(emel::text::encoders::ugm::action::context &, const xcda_blob_info &) noexcept;
  const init_handler_t init_handlers[2] = {
      ugm_init_xcda_blob_none,
      ugm_init_xcda_blob_some,
  };
  return init_handlers[static_cast<size_t>(blob.bounded)](ctx, blob);
}

inline bool ugm_tables_ready(const emel::text::encoders::ugm::action::context &ctx,
                             const emel::model::data::vocab &vocab) noexcept {
  return ctx.ugm_tables_ready && ctx.ugm_vocab == &vocab;
}

inline void ugm_insert_token_none(emel::text::encoders::detail::naive_trie &,
                                  const std::string_view,
                                  const int32_t) noexcept {}

inline void ugm_insert_token_some(emel::text::encoders::detail::naive_trie &trie,
                                  const std::string_view text,
                                  const int32_t id) noexcept {
  ugm_trie_insert(trie, text.data(), text.size(), id);
}

inline bool rebuild_ugm_tables(emel::text::encoders::ugm::action::context &ctx,
                               const emel::model::data::vocab &vocab) noexcept {
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
    const float min_candidate = std::min(ctx.min_score, entry.score);
    const float max_candidate = std::max(ctx.max_score, entry.score);
    ctx.min_score = select_f32(update_min, min_candidate, ctx.min_score);
    ctx.max_score = select_f32(update_min, max_candidate, ctx.max_score);

    using insert_handler_t =
        void (*)(emel::text::encoders::detail::naive_trie &, std::string_view, int32_t) noexcept;
    const insert_handler_t insert_handlers[2] = {
        ugm_insert_token_none,
        ugm_insert_token_some,
    };
    insert_handlers[static_cast<size_t>(insert_general)](
        ctx.token_matcher, text, static_cast<int32_t>(id));
    insert_handlers[static_cast<size_t>(insert_user_defined)](
        ctx.user_defined_token_matcher, text, static_cast<int32_t>(id));
  }

  const bool has_normal_scores = ctx.min_score != std::numeric_limits<float>::max();
  ctx.min_score = select_f32(has_normal_scores, ctx.min_score, 0.0f);
  ctx.unknown_token_score = ctx.min_score - ctx.unknown_token_score_penalty;
  init_xcda_tables(ctx);
  ctx.ugm_tables_ready = true;
  return true;
}

inline bool keep_ugm_tables(emel::text::encoders::ugm::action::context &,
                            const emel::model::data::vocab &) noexcept {
  return true;
}

inline bool ensure_ugm_tables(emel::text::encoders::ugm::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  const bool already_ready = ugm_tables_ready(ctx, vocab);
  using ensure_handler_t = bool (*)(emel::text::encoders::ugm::action::context &,
                                    const emel::model::data::vocab &) noexcept;
  const ensure_handler_t ensure_handlers[2] = {
      rebuild_ugm_tables,
      keep_ugm_tables,
  };
  return ensure_handlers[static_cast<size_t>(already_ready)](ctx, vocab);
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

inline size_t trie_longest_prefix_none(const emel::text::encoders::detail::naive_trie &,
                                       const char *,
                                       const size_t) noexcept {
  return 0u;
}

inline size_t trie_longest_prefix_some(const emel::text::encoders::detail::naive_trie &trie,
                                       const char *text,
                                       const size_t len) noexcept {
  size_t matched = 0;
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
  return matched;
}

inline size_t trie_longest_prefix(const emel::text::encoders::detail::naive_trie &trie,
                                  const char *text,
                                  const size_t len) noexcept {
  using prefix_handler_t = size_t (*)(const emel::text::encoders::detail::naive_trie &,
                                      const char *,
                                      size_t) noexcept;
  const prefix_handler_t prefix_handlers[2] = {
      trie_longest_prefix_none,
      trie_longest_prefix_some,
  };
  return prefix_handlers[static_cast<size_t>(len > 0u)](trie, text, len);
}

inline normalization_result normalize_prefix_at_end(const std::string_view input,
                                                    const size_t input_offset) noexcept {
  return {input.data() + input_offset, 0, 0};
}

inline normalization_result normalize_prefix_user_miss(const std::string_view,
                                                       const size_t,
                                                       const size_t) noexcept {
  return {};
}

inline normalization_result normalize_prefix_user_hit(const std::string_view input,
                                                      const size_t input_offset,
                                                      const size_t user_len) noexcept {
  return {input.data() + input_offset, user_len, user_len};
}

inline void normalize_prefix_scan_xcda_none(const emel::text::encoders::ugm::action::context &,
                                            const std::string_view,
                                            const size_t,
                                            size_t &,
                                            size_t &) noexcept {}

inline void normalize_prefix_scan_xcda_some(const emel::text::encoders::ugm::action::context &ctx,
                                            const std::string_view input,
                                            const size_t input_offset,
                                            size_t &longest_prefix_length,
                                            size_t &longest_prefix_offset) noexcept {
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

inline normalization_result normalize_prefix_prefix_invalid(
    const emel::text::encoders::ugm::action::context &, const size_t, const size_t) noexcept {
  return {nullptr, 0, 0};
}

inline normalization_result normalize_prefix_prefix_valid(
    const emel::text::encoders::ugm::action::context &ctx,
    const size_t longest_prefix_length,
    const size_t longest_prefix_offset) noexcept {
  const char *replacement = ctx.prefix_replacements + longest_prefix_offset;
  const size_t replacement_len = std::strlen(replacement);
  return {replacement, replacement_len, longest_prefix_length};
}

inline normalization_result normalize_prefix_invalid_utf8(const std::string_view,
                                                          const size_t,
                                                          const size_t) noexcept {
  static constexpr std::array<char, 3> replacement = {'\xEF', '\xBF', '\xBD'};
  return {replacement.data(), replacement.size(), 1};
}

inline normalization_result normalize_prefix_valid_utf8(const std::string_view input,
                                                        const size_t input_offset,
                                                        const size_t consumed) noexcept {
  return {input.data() + input_offset, consumed, consumed};
}

inline normalization_result normalize_prefix_core(emel::text::encoders::ugm::action::context &ctx,
                                                  const std::string_view input,
                                                  const size_t input_offset,
                                                  const size_t remaining) noexcept {
  size_t matched = 0;
  (void)matched;
  size_t longest_prefix_length = 0;
  size_t longest_prefix_offset = 0;

  const bool has_xcda = ctx.xcda_table != nullptr && ctx.xcda_table_size > 0u;
  using scan_xcda_handler_t = void (*)(const emel::text::encoders::ugm::action::context &,
                                       std::string_view,
                                       size_t,
                                       size_t &,
                                       size_t &) noexcept;
  const scan_xcda_handler_t scan_xcda_handlers[2] = {
      normalize_prefix_scan_xcda_none,
      normalize_prefix_scan_xcda_some,
  };
  scan_xcda_handlers[static_cast<size_t>(has_xcda)](
      ctx, input, input_offset, longest_prefix_length, longest_prefix_offset);

  const bool has_prefix = longest_prefix_length > 0u;
  const bool offset_ok = longest_prefix_offset < ctx.prefix_replacements_size;
  using prefix_handler_t = normalization_result (*)(
      const emel::text::encoders::ugm::action::context &, size_t, size_t) noexcept;
  const prefix_handler_t prefix_handlers[2] = {
      normalize_prefix_prefix_invalid,
      normalize_prefix_prefix_valid,
  };
  const normalization_result prefix_result = prefix_handlers[static_cast<size_t>(offset_ok)](
      ctx, longest_prefix_length, longest_prefix_offset);

  const uint8_t first = static_cast<uint8_t>(input[input_offset]);
  const bool continuation = (first & 0xC0u) == 0x80u;
  const size_t len_raw = ugm_utf8_len(static_cast<char>(first));
  const bool bounded = len_raw <= remaining;
  const bool invalid = continuation || !bounded;
  const size_t consumed = select_size(bounded, len_raw, static_cast<size_t>(1));
  using utf8_handler_t = normalization_result (*)(std::string_view, size_t, size_t) noexcept;
  const utf8_handler_t utf8_handlers[2] = {
      normalize_prefix_valid_utf8,
      normalize_prefix_invalid_utf8,
  };
  const normalization_result utf8_result =
      utf8_handlers[static_cast<size_t>(invalid)](input, input_offset, consumed);

  const std::array<normalization_result, 2> result_table{utf8_result, prefix_result};
  return result_table[static_cast<size_t>(has_prefix)];
}

inline normalization_result normalize_prefix_not_end(const emel::model::data::vocab &vocab,
                                                     emel::text::encoders::ugm::action::context &ctx,
                                                     const std::string_view input,
                                                     const size_t input_offset) noexcept {
  (void)vocab;
  const size_t remaining = input.size() - input_offset;
  const size_t user_len = trie_longest_prefix(
      ctx.user_defined_token_matcher, input.data() + input_offset, remaining);
  const bool user_hit = user_len > 0u;
  using user_handler_t = normalization_result (*)(std::string_view, size_t, size_t) noexcept;
  const user_handler_t user_handlers[2] = {
      normalize_prefix_user_miss,
      normalize_prefix_user_hit,
  };
  const normalization_result user_result =
      user_handlers[static_cast<size_t>(user_hit)](input, input_offset, user_len);
  const normalization_result core_result =
      normalize_prefix_core(ctx, input, input_offset, remaining);
  const std::array<normalization_result, 2> result_table{core_result, user_result};
  return result_table[static_cast<size_t>(user_hit)];
}

inline normalization_result normalize_prefix(const emel::model::data::vocab &vocab,
                                             emel::text::encoders::ugm::action::context &ctx,
                                             const std::string_view input,
                                             const size_t input_offset) noexcept {
  const bool at_end = input_offset >= input.size();
  using prefix_handler_t = normalization_result (*)(
      const emel::model::data::vocab &, emel::text::encoders::ugm::action::context &,
      std::string_view, size_t) noexcept;
  const prefix_handler_t prefix_handlers[2] = {
      normalize_prefix_not_end,
      [](const emel::model::data::vocab &,
         emel::text::encoders::ugm::action::context &,
         const std::string_view in,
         const size_t off) noexcept {
        return normalize_prefix_at_end(in, off);
      },
  };
  return prefix_handlers[static_cast<size_t>(at_end)](vocab, ctx, input, input_offset);
}

struct normalize_emit_state {
  size_t out_len = 0u;
  bool is_space_prepended = false;
  bool processing_non_ws = false;
  bool ok = true;
};

inline void ugm_append_bytes_none(emel::text::encoders::ugm::action::context &,
                                  normalize_emit_state &,
                                  const char *,
                                  const size_t) noexcept {}

inline void ugm_append_bytes_some(emel::text::encoders::ugm::action::context &ctx,
                                  normalize_emit_state &state,
                                  const char *src,
                                  const size_t len) noexcept {
  std::memcpy(ctx.scratch.buffer.data() + state.out_len, src, len);
  state.out_len += len;
}

inline bool ugm_append_bytes_if(const bool emit,
                                emel::text::encoders::ugm::action::context &ctx,
                                normalize_emit_state &state,
                                const char *src,
                                const size_t len) noexcept {
  const bool has_capacity = state.out_len + len <= ctx.scratch.buffer.size();
  const bool write = emit && has_capacity;
  using append_handler_t = void (*)(emel::text::encoders::ugm::action::context &,
                                    normalize_emit_state &, const char *, size_t) noexcept;
  const append_handler_t append_handlers[2] = {
      ugm_append_bytes_none,
      ugm_append_bytes_some,
  };
  append_handlers[static_cast<size_t>(write)](ctx, state, src, len);
  return !emit || has_capacity;
}

inline void process_normalized_space(emel::text::encoders::ugm::action::context &ctx,
                                     normalize_emit_state &state,
                                     const char,
                                     const char *space,
                                     const size_t space_len,
                                     const bool,
                                     const bool shall_merge_spaces) noexcept {
  state.processing_non_ws = false;
  const bool emit_space = state.ok && !shall_merge_spaces;
  const bool space_ok = ugm_append_bytes_if(emit_space, ctx, state, space, space_len);
  state.ok = state.ok && space_ok;
}

inline void process_normalized_non_space(emel::text::encoders::ugm::action::context &ctx,
                                         normalize_emit_state &state,
                                         const char c,
                                         const char *space,
                                         const size_t space_len,
                                         const bool shall_prepend_space,
                                         const bool shall_merge_spaces) noexcept {
  const bool begin_non_ws = !state.processing_non_ws;
  state.processing_non_ws = true;
  const bool emit_prefix = begin_non_ws &&
                           ((shall_prepend_space && !state.is_space_prepended) ||
                            shall_merge_spaces);
  const bool prefix_ok = ugm_append_bytes_if(state.ok && emit_prefix, ctx, state, space, space_len);
  const bool prefix_written = state.ok && emit_prefix && prefix_ok;
  state.ok = state.ok && prefix_ok;
  state.is_space_prepended = state.is_space_prepended || prefix_written;

  const bool emit_char = state.ok;
  const bool char_ok = ugm_append_bytes_if(emit_char, ctx, state, &c, 1u);
  state.ok = state.ok && char_ok;
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

  normalize_emit_state state{};

  size_t input_offset = 0;
  while (input_offset < input.size() && state.ok) {
    normalization_result norm = normalize_prefix(vocab, ctx, input, input_offset);
    const bool invalid_norm = norm.normalized == nullptr && norm.consumed_input == 0u;
    state.ok = state.ok && !invalid_norm;

    const size_t normalized_len = norm.normalized_len * static_cast<size_t>(state.ok);
    for (size_t i = 0; i < normalized_len; ++i) {
      const char c = norm.normalized[i];
      const bool non_space = c != ' ';
      using process_char_handler_t = void (*)(emel::text::encoders::ugm::action::context &,
                                              normalize_emit_state &, char, const char *, size_t,
                                              bool, bool) noexcept;
      const process_char_handler_t process_char_handlers[2] = {
          process_normalized_space,
          process_normalized_non_space,
      };
      process_char_handlers[static_cast<size_t>(non_space)](
          ctx, state, c, space, space_len, shall_prepend_space, shall_merge_spaces);
    }

    input_offset += norm.consumed_input * static_cast<size_t>(state.ok);
  }

  const bool append_space = state.ok && shall_append_space;
  const bool append_ok = ugm_append_bytes_if(append_space, ctx, state, space, space_len);
  state.ok = state.ok && append_ok;

  out_view = std::string_view(
      ctx.scratch.buffer.data(), state.out_len * static_cast<size_t>(state.ok));
  return state.ok;
}

inline void normalize_ugm_into_none(const emel::model::data::vocab &,
                                    emel::text::encoders::ugm::action::context &,
                                    const std::string_view,
                                    std::string_view &out_view,
                                    bool &ok) noexcept {
  out_view = std::string_view{};
  ok = true;
}

inline void normalize_ugm_into_some(const emel::model::data::vocab &vocab,
                                    emel::text::encoders::ugm::action::context &ctx,
                                    const std::string_view text,
                                    std::string_view &out_view,
                                    bool &ok) noexcept {
  ok = normalize_ugm_into(vocab, ctx, text, out_view);
}

inline encode_result encode_ugm(const event::encode &ev,
                                emel::text::encoders::ugm::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  const bool has_text = !ev.text.empty();
  const bool tables_ready = ugm_tables_ready(ctx, vocab);
  int32_t err = select_i32(has_text && !tables_ready, EMEL_ERR_INVALID_ARGUMENT, EMEL_OK);
  int32_t unk_id = vocab.unk_id;
  const bool resolve_unk = has_text && err == EMEL_OK && unk_id == k_token_null;
  const int32_t looked_up_unk = ugm_lookup_token_exact(vocab, "<unk>");
  unk_id = select_i32(resolve_unk, looked_up_unk, unk_id);

  std::string_view normalized{};
  bool normalized_ok = true;
  const bool normalize_active = has_text && err == EMEL_OK;
  using normalize_handler_t = void (*)(const emel::model::data::vocab &,
                                       emel::text::encoders::ugm::action::context &,
                                       std::string_view, std::string_view &, bool &) noexcept;
  const normalize_handler_t normalize_handlers[2] = {
      normalize_ugm_into_none,
      normalize_ugm_into_some,
  };
  normalize_handlers[static_cast<size_t>(normalize_active)](vocab, ctx, ev.text, normalized,
                                                            normalized_ok);
  err = select_i32(normalize_active && !normalized_ok, EMEL_ERR_INVALID_ARGUMENT, err);

  const size_t input_len_raw = normalized.size();
  const size_t input_len = input_len_raw * static_cast<size_t>(normalize_active && normalized_ok);
  const bool no_input = normalize_active && normalized_ok && input_len == 0u;
  const bool overflow = normalize_active && normalized_ok && input_len >= ctx.best.size();
  err = select_i32(err == EMEL_OK && overflow, EMEL_ERR_INVALID_ARGUMENT, err);

  const bool run_dp = normalize_active && normalized_ok && !no_input && err == EMEL_OK;
  const size_t safe_input_len = input_len * static_cast<size_t>(run_dp);

  for (size_t i = 0; i <= safe_input_len; ++i) {
    ctx.best[i] = {unk_id, 0u, -std::numeric_limits<double>::max()};
  }
  ctx.best[0] = {unk_id, 0u, 0.0};

  size_t input_offset = 0;
  while (input_offset < safe_input_len) {
    const size_t n_utf8_code_units = std::min(
      static_cast<size_t>(ugm_utf8_len(normalized[input_offset])),
      safe_input_len - input_offset);
    bool single_codepoint_token_found = false;
    const auto current_best = ctx.best[input_offset];
    size_t prefix_offset = input_offset;
    const auto *node = ugm_trie_root(ctx.token_matcher, normalized[prefix_offset]);
    prefix_offset += 1u;
    bool walking = node != nullptr && prefix_offset <= safe_input_len;

    while (walking) {
      const bool has_value = node->has_value;
      const bool single_codepoint = prefix_offset - input_offset == n_utf8_code_units;
      single_codepoint_token_found = single_codepoint_token_found || (has_value && single_codepoint);
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
      const bool better = has_value && challenger_score > current_champ.score_sum;
      current_champ.token_id = select_i32(better, token_id, current_champ.token_id);
      current_champ.input_offset = select_u32(
          better, static_cast<uint32_t>(input_offset), current_champ.input_offset);
      current_champ.score_sum = select_f64(better, challenger_score, current_champ.score_sum);

      const bool can_advance = prefix_offset < safe_input_len;
      const size_t safe_offset = select_size(can_advance, prefix_offset, input_offset);
      const auto *next_node = ugm_trie_step(*node, normalized[safe_offset]);
      const std::array<const emel::text::encoders::detail::naive_trie::node *, 2> options{
        node,
        next_node,
      };
      node = options[static_cast<size_t>(can_advance)];
      prefix_offset += static_cast<size_t>(can_advance);
      walking = can_advance && node != nullptr && prefix_offset <= safe_input_len;
    }

    const bool use_unk = !single_codepoint_token_found && unk_id != k_token_null;
    const double challenger_score =
        current_best.score_sum + static_cast<double>(ctx.unknown_token_score);
    const size_t next_offset = input_offset + n_utf8_code_units;
    auto &current_champ = ctx.best[next_offset];
    const bool better = use_unk && challenger_score > current_champ.score_sum;
    current_champ.token_id = select_i32(better, unk_id, current_champ.token_id);
    current_champ.input_offset = select_u32(
        better, static_cast<uint32_t>(input_offset), current_champ.input_offset);
    current_champ.score_sum = select_f64(better, challenger_score, current_champ.score_sum);

    input_offset += n_utf8_code_units;
  }

  size_t out_count = 0;
  bool is_prev_unknown = false;
  emel::text::encoders::ugm::action::best_tokenization tokenization = ctx.best[safe_input_len];
  bool tracing = run_dp;
  while (tracing && err == EMEL_OK) {
    const bool is_unknown = tokenization.token_id == unk_id;
    const bool emit_token = !(is_prev_unknown && is_unknown);
    const bool has_room = out_count < ctx.token_buffer.size();
    err = select_i32(err == EMEL_OK && emit_token && !has_room, EMEL_ERR_INVALID_ARGUMENT, err);
    const bool write = emit_token && has_room;
    const size_t write_idx = out_count * static_cast<size_t>(write);
    ctx.token_buffer[write_idx] =
        select_i32(write, tokenization.token_id, ctx.token_buffer[write_idx]);
    out_count += static_cast<size_t>(write);

    const bool at_root = tokenization.input_offset == 0u;
    const auto next_tokenization = ctx.best[tokenization.input_offset];
    const bool advance = !at_root;
    is_prev_unknown = select_bool(advance, is_unknown, is_prev_unknown);
    tokenization.token_id = select_i32(advance, next_tokenization.token_id, tokenization.token_id);
    tokenization.input_offset =
        select_u32(advance, next_tokenization.input_offset, tokenization.input_offset);
    tokenization.score_sum =
        select_f64(advance, next_tokenization.score_sum, tokenization.score_sum);
    tracing = !at_root;
  }

  int32_t count = 0;
  size_t emit_limit = select_size(err == EMEL_OK, out_count, static_cast<size_t>(0));
  for (size_t i = 0; i < emit_limit; ++i) {
    const int32_t token = ctx.token_buffer[out_count - 1u - i];
    const bool pushed = ugm_push_token(ev, token, count);
    err = select_i32(err == EMEL_OK && !pushed, EMEL_ERR_INVALID_ARGUMENT, err);
    emit_limit = select_size(err == EMEL_OK, emit_limit, i + 1u);
  }

  result.token_count = count * static_cast<int32_t>(err == EMEL_OK);
  result.error = err;
  return result;
}

}  // namespace emel::text::encoders::ugm::detail
