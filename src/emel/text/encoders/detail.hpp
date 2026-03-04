#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/types.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::detail {

template <class value_type>
inline void write_optional(value_type * destination, value_type & sink,
                           const value_type value) noexcept {
  value_type * destinations[2] = {&sink, destination};
  *destinations[static_cast<size_t>(destination != nullptr)] = value;
}

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

inline uint8_t select_u8(const bool choose_true,
                         const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) |
                              (true_value & mask));
}

template <class value_type>
inline value_type * pick_ptr(const bool choose_true,
                               value_type * true_value,
                               value_type * false_value) noexcept {
  value_type * values[2] = {false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline void dispatch_done_noop(const event::encode &, const int32_t) noexcept {
}

inline void dispatch_done_call(const event::encode & request,
                               const int32_t token_count) noexcept {
  request.dispatch_done(request.owner_sm, events::encoding_done{request, token_count});
}

inline void dispatch_done_if_bound(const event::encode & request,
                                   const int32_t token_count) noexcept {
  using dispatch_fn = void (*)(const event::encode &, int32_t);
  const std::array<dispatch_fn, 2> dispatchers{&dispatch_done_noop, &dispatch_done_call};
  const bool can_dispatch = request.dispatch_done != nullptr && request.owner_sm != nullptr;
  dispatchers[static_cast<size_t>(can_dispatch)](request, token_count);
}

inline void dispatch_error_noop(const event::encode &, const int32_t) noexcept {
}

inline void dispatch_error_call(const event::encode & request, const int32_t err) noexcept {
  request.dispatch_error(request.owner_sm, events::encoding_error{request, err});
}

inline void dispatch_error_if_bound(const event::encode & request, const int32_t err) noexcept {
  using dispatch_fn = void (*)(const event::encode &, int32_t);
  const std::array<dispatch_fn, 2> dispatchers{&dispatch_error_noop, &dispatch_error_call};
  const bool can_dispatch = request.dispatch_error != nullptr && request.owner_sm != nullptr;
  dispatchers[static_cast<size_t>(can_dispatch)](request, err);
}

inline void publish_error(const event::encode & request, const event::encode_ctx & ctx) noexcept {
  dispatch_error_if_bound(request, ctx.err);
}

inline void publish_done(const event::encode & request, const event::encode_ctx & ctx) noexcept {
  dispatch_done_if_bound(request, ctx.token_count);
}

inline void publish_result(const event::encode & request,
                           const event::encode_ctx & ctx) noexcept {
  using publish_fn = void (*)(const event::encode &, const event::encode_ctx &);
  const std::array<publish_fn, 2> publishers{&publish_error, &publish_done};
  publishers[static_cast<size_t>(ctx.err == EMEL_OK)](request, ctx);
}

inline int32_t select_final_error(const bool accepted,
                                  const int32_t runtime_error) noexcept {
  const std::array<int32_t, 2> accepted_errors{EMEL_ERR_INVALID_ARGUMENT, runtime_error};
  const std::array<int32_t, 2> final_errors{
      accepted_errors[static_cast<size_t>(accepted)],
      EMEL_OK,
  };
  const bool succeeded = accepted && runtime_error == EMEL_OK;
  return final_errors[static_cast<size_t>(succeeded)];
}

template <size_t N>
inline std::string_view string_view_from_array(const std::array<char, N> &data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline size_t utf8_len(const char byte) {
  return emel::text::unicode_len_utf8(byte);
}

inline bool is_chinese_char(const uint32_t cpt) {
  return emel::text::unicode_cpt_is_han(cpt);
}

inline std::string cpt_to_utf8(const uint32_t cpt) {
  return emel::text::unicode_cpt_to_utf8(cpt);
}

inline std::string_view token_text(const emel::model::data::vocab &vocab,
                                   const int32_t id) {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(vocab.token_storage.data() + static_cast<size_t>(offset),
                          static_cast<size_t>(length));
}

inline bool is_token_type(const emel::model::data::vocab &vocab,
                          const int32_t id,
                          const int32_t type) {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  return valid_id && vocab.entries[idx].type == type;
}

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t hash_bytes(const uint32_t seed, const std::string_view data) {
  uint32_t hash = seed;
  for (const unsigned char byte : data) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash == 0u, 1u, hash);
}

inline uint32_t hash_sv(const std::string_view data) {
  return hash_bytes(k_fnv_offset, data);
}

inline uint32_t hash_concat(const std::string_view left,
                            const std::string_view right) {
  return hash_bytes(hash_bytes(k_fnv_offset, left), right);
}

inline uint32_t hash_pair(const std::string_view left,
                          const std::string_view right) {
  const uint32_t h1 = hash_sv(left);
  const uint32_t h2 = hash_sv(right);
  const uint32_t combined = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  return select_u32(combined == 0u, 1u, combined);
}

inline std::string_view merge_text(const emel::model::data::vocab &vocab,
                                   const int32_t idx) {
  const bool valid_idx = idx >= 0 && static_cast<uint32_t>(idx) < vocab.n_merges;
  const uint32_t merge_idx = select_u32(valid_idx, static_cast<uint32_t>(idx), 0u);
  const uint32_t offset = vocab.merge_offsets[merge_idx];
  const uint32_t length = vocab.merge_lengths[merge_idx];
  const size_t end = static_cast<size_t>(offset) + static_cast<size_t>(length);
  const bool range_ok = valid_idx && end <= vocab.merge_storage.size();
  const uint32_t safe_offset = select_u32(range_ok, offset, 0u);
  const uint32_t safe_length = select_u32(range_ok, length, 0u);
  return std::string_view(vocab.merge_storage.data() + static_cast<size_t>(safe_offset),
                          static_cast<size_t>(safe_length));
}

inline bool merge_match(const std::string_view merge,
                        const std::string_view left,
                        const std::string_view right) {
  const bool non_empty = !merge.empty();
  const size_t raw_pos = merge.find(' ');
  const bool has_separator = raw_pos != std::string_view::npos;
  const size_t pos = select_size(has_separator, raw_pos, 0u);
  const size_t expected_size = left.size() + right.size() + 1u;
  const bool size_match = merge.size() == expected_size;
  const bool left_match = merge.substr(0, pos) == left;
  const size_t right_start = select_size(has_separator, pos + 1u, 0u);
  const bool right_match = merge.substr(right_start) == right;
  return non_empty && has_separator && size_match && left_match && right_match;
}

inline bool insert_token_map(token_map &map,
                             const emel::model::data::vocab &vocab,
                             const std::string_view text,
                             const int32_t id) {
  const bool active = !text.empty();
  bool success = !active;
  bool loop_active = active;

  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const bool step_active = loop_active;
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing = map.values[slot];
    const std::string_view existing_text = token_text(vocab, existing);
    const bool same_text = step_active && hash_match && existing_text == text;
    const bool claim_slot = step_active && (empty_slot || same_text);
    const bool collision = step_active && hash_match && !same_text;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, id, existing);
    map.count += static_cast<uint32_t>(claim_slot && empty_slot);

    success = success || claim_slot;
    const bool step_done = claim_slot || collision;
    loop_active = loop_active && !step_done;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline bool insert_merge_map(merge_map &map,
                             const std::string_view left,
                             const std::string_view right,
                             const int32_t rank,
                             const emel::model::data::vocab &vocab) {
  const bool active = !left.empty() && !right.empty();
  bool loop_active = active;
  bool success = false;

  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0; probes < k_merge_hash_size; ++probes) {
    const bool step_active = loop_active;
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_rank = map.values[slot];
    const std::string_view merge = merge_text(vocab, existing_rank);
    const bool same_merge = step_active && hash_match && merge_match(merge, left, right);
    const bool claim_slot = step_active && empty_slot;
    const bool collision = step_active && hash_match && !same_merge;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, rank, existing_rank);
    map.count += static_cast<uint32_t>(claim_slot);

    success = success || claim_slot || same_merge;
    const bool step_done = claim_slot || same_merge || collision;
    loop_active = loop_active && !step_done;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline int32_t lookup_token(const action::context &ctx,
                            const std::string_view text) {
  const bool active = !text.empty();
  bool loop_active = active;
  int32_t resolved = k_token_null;

  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const bool step_active = loop_active;
    const uint32_t entry = ctx.token_to_id.hashes[slot];
    const bool empty_slot = entry == 0u;
    const bool hash_match = entry == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact_match = step_active && hash_match && token_text(*ctx.vocab, id) == text;
    const bool collision = step_active && hash_match && !exact_match;

    resolved = select_i32(exact_match, id, resolved);
    const bool step_done = step_active && (empty_slot || exact_match || collision);
    loop_active = loop_active && !step_done;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline int32_t lookup_token_concat(const action::context &ctx,
                                   const std::string_view left,
                                   const std::string_view right) {
  const uint32_t hash = hash_concat(left, right);
  const uint32_t mask = k_token_hash_size - 1u;
  const size_t combined_len = left.size() + right.size();
  uint32_t slot = hash & mask;
  int32_t resolved = k_token_null;
  bool loop_active = true;

  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const bool step_active = loop_active;
    const uint32_t entry = ctx.token_to_id.hashes[slot];
    const bool empty_slot = entry == 0u;
    const bool hash_match = entry == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const std::string_view token = token_text(*ctx.vocab, id);
    const bool size_match = token.size() == combined_len;

    const char empty_byte = '\0';
    const std::array<const char *, 2> token_ptrs = {&empty_byte, token.data()};
    const std::array<const char *, 2> left_ptrs = {&empty_byte, left.data()};
    const std::array<const char *, 2> right_ptrs = {&empty_byte, right.data()};

    const size_t left_len = select_size(size_match, left.size(), 0u);
    const size_t right_len = select_size(size_match, right.size(), 0u);
    const size_t right_offset = left_len;

    const char *token_ptr = token_ptrs[static_cast<size_t>(size_match)];
    const char *left_ptr = left_ptrs[static_cast<size_t>(!left.empty())];
    const char *right_ptr = right_ptrs[static_cast<size_t>(!right.empty())];

    const bool left_match = std::memcmp(token_ptr, left_ptr, left_len) == 0;
    const bool right_match = std::memcmp(token_ptr + right_offset, right_ptr, right_len) == 0;
    const bool exact_match =
        step_active && hash_match && size_match && left_match && right_match;

    resolved = select_i32(exact_match, id, resolved);
    const bool step_done = step_active && (empty_slot || exact_match);
    loop_active = loop_active && !step_done;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline int32_t lookup_merge_rank(const action::context &ctx,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view left,
                                 const std::string_view right) {
  const bool active = !left.empty() && !right.empty();
  bool done = !active;
  int32_t resolved = k_token_null;

  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0; probes < k_merge_hash_size && !done; ++probes) {
    const uint32_t entry = ctx.bpe_ranks.hashes[slot];
    const bool empty_slot = entry == 0u;
    const bool hash_match = entry == hash;
    const int32_t rank = ctx.bpe_ranks.values[slot];
    const std::string_view merge = merge_text(vocab, rank);
    const bool exact_match = hash_match && merge_match(merge, left, right);
    const bool collision = hash_match && !exact_match;

    resolved = select_i32(exact_match, rank, resolved);
    done = done || empty_slot || exact_match || collision;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool push_token(const event::encode &ev, const int32_t token, int32_t &count) {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t *base_ptr = pick_ptr(has_buffer, ev.token_ids.data(), &sink);

  const bool non_negative_count = count >= 0;
  const int32_t safe_count = select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space = has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;

  const size_t write_index = count_index * static_cast<size_t>(write);
  int32_t *write_ptr = base_ptr + write_index;
  *write_ptr = select_i32(write, token, *write_ptr);
  count += static_cast<int32_t>(write);
  return write;
}

inline const std::array<uint32_t, 256> &byte_to_codepoint_table() {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> map = {};
    std::array<bool, 256> used = {};

    for (size_t idx = 0; idx < 256; ++idx) {
      used[idx] = false;
      map[idx] = 0u;
    }

    for (uint32_t c = 33; c <= 126; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }

    for (uint32_t c = 161; c <= 172; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }

    for (uint32_t c = 174; c <= 255; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }

    uint32_t n = 0u;
    for (size_t idx = 0; idx < 256; ++idx) {
      const bool assign_extra = !used[idx];
      const uint32_t extra_value = 256u + n;
      map[idx] = select_u32(assign_extra, extra_value, map[idx]);
      n += static_cast<uint32_t>(assign_extra);
    }

    return map;
  }();
  return table;
}

inline uint8_t encode_cpt_utf8(const uint32_t cpt, char out[4]) {
  const uint8_t len = select_u8(
      cpt <= 0x7Fu,
      1u,
      select_u8(cpt <= 0x7FFu,
                2u,
                select_u8(cpt <= 0xFFFFu, 3u, 4u)));

  const size_t idx = static_cast<size_t>(len - 1u);

  const std::array<char, 4> first_bytes = {
      static_cast<char>(cpt),
      static_cast<char>(0xC0u | ((cpt >> 6u) & 0x1Fu)),
      static_cast<char>(0xE0u | ((cpt >> 12u) & 0x0Fu)),
      static_cast<char>(0xF0u | ((cpt >> 18u) & 0x07u)),
  };
  const std::array<char, 4> second_bytes = {
      0,
      static_cast<char>(0x80u | (cpt & 0x3Fu)),
      static_cast<char>(0x80u | ((cpt >> 6u) & 0x3Fu)),
      static_cast<char>(0x80u | ((cpt >> 12u) & 0x3Fu)),
  };
  const std::array<char, 4> third_bytes = {
      0,
      0,
      static_cast<char>(0x80u | (cpt & 0x3Fu)),
      static_cast<char>(0x80u | ((cpt >> 6u) & 0x3Fu)),
  };
  const std::array<char, 4> fourth_bytes = {
      0,
      0,
      0,
      static_cast<char>(0x80u | (cpt & 0x3Fu)),
  };

  out[0] = first_bytes[idx];
  out[1] = second_bytes[idx];
  out[2] = third_bytes[idx];
  out[3] = fourth_bytes[idx];
  return len;
}

inline const std::array<std::string, 256> &byte_to_utf8_table() {
  static const std::array<std::string, 256> table = [] {
    std::array<std::string, 256> map = {};
    const auto &codepoints = byte_to_codepoint_table();
    for (size_t idx = 0; idx < map.size(); ++idx) {
      map[idx] = cpt_to_utf8(codepoints[idx]);
    }
    return map;
  }();
  return table;
}

inline int32_t byte_to_token_raw(const action::context &ctx,
                                 const uint8_t byte) {
  const char raw = static_cast<char>(byte);
  return lookup_token(ctx, std::string_view(&raw, 1));
}

inline int32_t byte_to_token_piece(const action::context &ctx,
                                   const uint8_t byte) {
  char hex[7] = {};
  static const char *digits = "0123456789ABCDEF";
  hex[0] = '<';
  hex[1] = '0';
  hex[2] = 'x';
  hex[3] = digits[(byte >> 4u) & 0x0Fu];
  hex[4] = digits[byte & 0x0Fu];
  hex[5] = '>';
  hex[6] = '\0';

  const int32_t hex_token = lookup_token(ctx, std::string_view(hex, 6));
  const int32_t raw_token = byte_to_token_raw(ctx, byte);
  const bool has_hex = hex_token != k_token_null;
  return select_i32(has_hex, hex_token, raw_token);
}

inline int32_t byte_to_token_bpe(const action::context &ctx,
                                 const uint8_t byte) {
  const uint32_t cpt = byte_to_codepoint_table()[byte];
  char utf8[4] = {};
  const uint8_t len = encode_cpt_utf8(cpt, utf8);
  return lookup_token(ctx, std::string_view(utf8, len));
}

inline int32_t byte_to_token(const action::context &ctx,
                             const emel::model::data::vocab &vocab,
                             const uint8_t byte,
                             const emel::model::data::tokenizer_model model) {
  (void)vocab;

  const bool none_model = model == emel::model::data::tokenizer_model::NONE;
  const bool piece_model = model == emel::model::data::tokenizer_model::SPM ||
      model == emel::model::data::tokenizer_model::UGM ||
      model == emel::model::data::tokenizer_model::PLAMO2;
  const bool bpe_model = model == emel::model::data::tokenizer_model::BPE ||
      model == emel::model::data::tokenizer_model::WPM ||
      model == emel::model::data::tokenizer_model::RWKV;

  const int32_t piece_token = byte_to_token_piece(ctx, byte);
  const int32_t bpe_token = byte_to_token_bpe(ctx, byte);
  const int32_t raw_token = byte_to_token_raw(ctx, byte);

  const int32_t non_none_token = select_i32(piece_model,
                                            piece_token,
                                            select_i32(bpe_model, bpe_token, raw_token));
  return select_i32(none_model, k_token_null, non_none_token);
}

inline void ensure_tables_build_none(action::context &, bool &) noexcept {
}

inline void ensure_tables_insert_merge_none(action::context &,
                                            const std::string_view,
                                            const std::string_view,
                                            const int32_t,
                                            const emel::model::data::vocab &) noexcept {
}

inline bool ensure_tables_insert_token_none(action::context &,
                                            const emel::model::data::vocab &,
                                            const std::string_view,
                                            const int32_t) noexcept {
  return true;
}

inline bool ensure_tables_insert_token_some(action::context &ctx,
                                            const emel::model::data::vocab &vocab,
                                            const std::string_view text,
                                            const int32_t id) noexcept {
  return insert_token_map(ctx.token_to_id, vocab, text, id);
}

inline void ensure_tables_insert_merge_some(action::context &ctx,
                                            const std::string_view left,
                                            const std::string_view right,
                                            const int32_t idx,
                                            const emel::model::data::vocab &vocab) noexcept {
  insert_merge_map(ctx.bpe_ranks, left, right, idx, vocab);
}

inline void ensure_tables_build_some(action::context &ctx, bool &ok) noexcept {
  ctx.token_to_id.clear();
  ctx.bpe_ranks.clear();
  ctx.max_token_len = 0;

  const emel::model::data::vocab &vocab = *ctx.vocab;
  using insert_token_handler_t = bool (*)(action::context &,
                                          const emel::model::data::vocab &,
                                          std::string_view,
                                          int32_t) noexcept;
  const insert_token_handler_t insert_token_handlers[2] = {
      ensure_tables_insert_token_none,
      ensure_tables_insert_token_some,
  };

  bool loop_active = true;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const bool step_active = loop_active;
    const std::string_view text = token_text(vocab, static_cast<int32_t>(id));
    const bool inserted = insert_token_handlers[static_cast<size_t>(step_active)](
        ctx, vocab, text, static_cast<int32_t>(id));

    const int32_t text_len = static_cast<int32_t>(text.size());
    const bool longer = step_active && text_len > ctx.max_token_len;
    ctx.max_token_len = select_i32(longer, text_len, ctx.max_token_len);
    loop_active = loop_active && inserted;
  }
  const bool build_ok = loop_active;

  using insert_merge_handler_t = void (*)(action::context &,
                                          std::string_view,
                                          std::string_view,
                                          int32_t,
                                          const emel::model::data::vocab &) noexcept;
  const insert_merge_handler_t insert_merge_handlers[2] = {
      ensure_tables_insert_merge_none,
      ensure_tables_insert_merge_some,
  };

  for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
    const std::string_view merge = merge_text(vocab, static_cast<int32_t>(idx));
    const size_t pos_raw = merge.find(' ');
    const bool has_separator = pos_raw != std::string_view::npos;
    const size_t pos = select_size(has_separator, pos_raw, 0u);
    const std::string_view left = merge.substr(0, pos);
    const size_t right_start = select_size(has_separator, pos + 1u, 0u);
    const std::string_view right = merge.substr(right_start);
    const bool should_insert = !merge.empty() && has_separator;
    insert_merge_handlers[static_cast<size_t>(should_insert)](
        ctx,
        left,
        right,
        static_cast<int32_t>(idx),
        vocab);
  }

  ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
  ctx.tables_ready = build_ok;
  ok = build_ok;
}

inline void ensure_tables_rebuild_none(action::context &, bool &ok) noexcept {
  ok = false;
}

inline void ensure_tables_rebuild_some(action::context &ctx, bool &ok) noexcept {
  using build_handler_t = void (*)(action::context &, bool &) noexcept;
  const build_handler_t build_handlers[2] = {
      ensure_tables_build_some,
      ensure_tables_build_none,
  };

  bool build_ok = true;
  build_handlers[static_cast<size_t>(ctx.tables_ready)](ctx, build_ok);
  ok = ctx.tables_ready || build_ok;
}

inline bool ensure_tables(action::context &ctx) {
  bool ok = false;
  using rebuild_handler_t = void (*)(action::context &, bool &) noexcept;
  const rebuild_handler_t rebuild_handlers[2] = {
      ensure_tables_rebuild_none,
      ensure_tables_rebuild_some,
  };
  rebuild_handlers[static_cast<size_t>(ctx.vocab != nullptr)](ctx, ok);
  return ok;
}

inline void split_whitespace_noop(const std::string_view,
                                  std::vector<std::string_view> &,
                                  size_t &,
                                  const size_t) noexcept {
}

inline void split_whitespace_emit(const std::string_view text,
                                  std::vector<std::string_view> &parts,
                                  size_t &start,
                                  const size_t index) noexcept {
  parts.emplace_back(text.substr(start, index - start));
  start = index + 1u;
}

inline void split_whitespace(const std::string_view text,
                             std::vector<std::string_view> &parts) {
  parts.clear();
  size_t start = 0;

  using split_handler_t = void (*)(std::string_view,
                                   std::vector<std::string_view> &,
                                   size_t &,
                                   size_t) noexcept;
  const split_handler_t split_handlers[2] = {
      split_whitespace_noop,
      split_whitespace_emit,
  };

  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    const bool is_space = std::isspace(c) != 0;
    split_handlers[static_cast<size_t>(is_space)](text, parts, start, i);
  }

  parts.emplace_back(text.substr(start));
}

inline bool build_symbols(const std::string_view text,
                          encode_scratch &scratch,
                          encode_result &result) {
  scratch.symbol_count = 0;
  size_t offset = 0;

  while (offset < text.size() && scratch.symbol_count < scratch.offsets.size()) {
    const size_t len = std::min(text.size() - offset, utf8_len(text[offset]));
    const size_t symbol = scratch.symbol_count;

    scratch.offsets[symbol] = static_cast<uint32_t>(offset);
    scratch.lengths[symbol] = static_cast<uint32_t>(len);
    scratch.prev[symbol] = static_cast<int32_t>(symbol) - 1;

    const bool has_next = offset + len < text.size();
    scratch.next[symbol] = select_i32(has_next, static_cast<int32_t>(symbol) + 1, -1);

    scratch.symbol_count += 1;
    offset += len;
  }

  const bool success = offset == text.size();
  int32_t sink = 0;
  const bool set_prev_head = success && scratch.symbol_count > 0;
  int32_t *head_ptr = pick_ptr(set_prev_head, &scratch.prev[0], &sink);
  *head_ptr = -1;

  result.error = select_i32(success, result.error, EMEL_ERR_INVALID_ARGUMENT);
  return success;
}

inline void merge_symbols(encode_scratch &scratch,
                          const int32_t left,
                          const int32_t right) {
  scratch.lengths[static_cast<size_t>(left)] += scratch.lengths[static_cast<size_t>(right)];
  const int32_t right_next = scratch.next[static_cast<size_t>(right)];
  scratch.next[static_cast<size_t>(left)] = right_next;

  int32_t sink = 0;
  const bool has_right_next = right_next >= 0;
  int32_t *prev_ptr = pick_ptr(has_right_next,
                                 &scratch.prev[static_cast<size_t>(select_i32(has_right_next,
                                                                              right_next,
                                                                              0))],
                                 &sink);
  *prev_ptr = left;
  scratch.lengths[static_cast<size_t>(right)] = 0;
}

inline bool encode_bytes(const event::encode &ev,
                         action::context &ctx,
                         const emel::model::data::vocab &vocab,
                         const emel::model::data::tokenizer_model model,
                         encode_result &result) {
  int32_t count = 0;
  bool loop_active = true;

  for (size_t index = 0; index < ev.text.size(); ++index) {
    const bool step_active = loop_active;
    const unsigned char c = static_cast<unsigned char>(ev.text[index]);
    const int32_t token = byte_to_token(ctx, vocab, c, model);
    const int32_t gated_token = select_i32(step_active, token, k_token_null);
    const bool pushed = push_token(ev, gated_token, count);
    const bool step_ok = step_active && token != k_token_null && pushed;
    loop_active = loop_active && step_ok;
  }

  const bool success = loop_active;
  int32_t sink = result.token_count;
  int32_t *token_count_ptr = pick_ptr(success, &result.token_count, &sink);
  *token_count_ptr = count;
  result.error = select_i32(success, EMEL_OK, EMEL_ERR_BACKEND);
  return success;
}

}  // namespace emel::text::encoders::detail
