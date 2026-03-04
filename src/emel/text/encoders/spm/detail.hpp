#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "emel/model/data.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/spm/context.hpp"

namespace emel::text::encoders::spm::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline int32_t select_i32(const bool choose_true, const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true, const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask =
      static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true, const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline float select_f32(const bool choose_true, const float true_value,
                        const float false_value) noexcept {
  uint32_t true_bits = 0u;
  uint32_t false_bits = 0u;
  std::memcpy(&true_bits, &true_value, sizeof(true_bits));
  std::memcpy(&false_bits, &false_value, sizeof(false_bits));
  const uint32_t selected_bits = select_u32(choose_true, true_bits, false_bits);
  float selected = 0.0f;
  std::memcpy(&selected, &selected_bits, sizeof(selected));
  return selected;
}

inline std::string_view spm_token_text(const emel::model::data::vocab &vocab,
                                       const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(vocab.token_storage.data() +
                              static_cast<size_t>(offset),
                          static_cast<size_t>(length));
}

inline std::string_view spm_merge_text(const emel::model::data::vocab &vocab,
                                       const int32_t idx) noexcept {
  const bool valid_idx =
      idx >= 0 && static_cast<uint32_t>(idx) < vocab.n_merges;
  const uint32_t merge_idx =
      select_u32(valid_idx, static_cast<uint32_t>(idx), 0u);
  const uint32_t raw_offset = vocab.merge_offsets[merge_idx];
  const uint32_t raw_length = vocab.merge_lengths[merge_idx];
  const size_t merge_end =
      static_cast<size_t>(raw_offset) + static_cast<size_t>(raw_length);
  const bool bounded = valid_idx && merge_end <= vocab.merge_storage.size();
  const uint32_t offset = select_u32(bounded, raw_offset, 0u);
  const uint32_t length = select_u32(bounded, raw_length, 0u);
  return std::string_view(vocab.merge_storage.data() +
                              static_cast<size_t>(offset),
                          static_cast<size_t>(length));
}

inline bool spm_merge_match(const std::string_view merge,
                            const std::string_view left,
                            const std::string_view right) noexcept {
  const size_t pos = merge.find(' ');
  const bool has_space = pos != std::string_view::npos;
  const size_t left_len = select_size(has_space, pos, static_cast<size_t>(0));
  const size_t right_start =
      select_size(has_space, pos + static_cast<size_t>(1), merge.size());
  const size_t right_len = merge.size() - right_start;
  const std::string_view left_view(merge.data(), left_len);
  const std::string_view right_view(merge.data() + right_start, right_len);
  const size_t expected_size =
      left.size() + right.size() + static_cast<size_t>(1);
  const bool size_ok = merge.size() == expected_size;
  return has_space && size_ok && left_view == left && right_view == right;
}

inline uint32_t spm_hash_bytes(const uint32_t seed,
                               const std::string_view text) noexcept {
  uint32_t hash = seed;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash != 0u, hash, 1u);
}

inline uint32_t spm_hash_sv(const std::string_view text) noexcept {
  return spm_hash_bytes(k_fnv_offset, text);
}

inline uint32_t spm_hash_concat(const std::string_view left,
                                const std::string_view right) noexcept {
  return spm_hash_bytes(spm_hash_bytes(k_fnv_offset, left), right);
}

inline uint32_t spm_hash_pair(const std::string_view left,
                              const std::string_view right) noexcept {
  const uint32_t h1 = spm_hash_sv(left);
  const uint32_t h2 = spm_hash_sv(right);
  const uint32_t mixed = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  return select_u32(mixed != 0u, mixed, 1u);
}

inline bool spm_insert_token_map(emel::text::encoders::detail::token_map &map,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view text,
                                 const int32_t id) noexcept {
  const bool text_empty = text.empty();
  bool done = text_empty;
  bool success = text_empty;
  const uint32_t hash = spm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_id = map.values[slot];
    const std::string_view existing_text = spm_token_text(vocab, existing_id);
    const bool same_text = hash_match && existing_text == text;
    const bool claim_slot = empty_slot || same_text;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, id, existing_id);
    map.count += static_cast<uint32_t>(claim_slot && empty_slot);

    success = success || claim_slot;
    done = done || claim_slot;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline bool
spm_insert_merge_map(emel::text::encoders::detail::merge_map &map,
                     const std::string_view left, const std::string_view right,
                     const int32_t rank,
                     const emel::model::data::vocab &vocab) noexcept {
  const bool active = !left.empty() && !right.empty();
  bool done = !active;
  bool success = false;
  const uint32_t hash = spm_hash_pair(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_merge_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_rank = map.values[slot];
    const std::string_view merge = spm_merge_text(vocab, existing_rank);
    const bool same_pair = hash_match && spm_merge_match(merge, left, right);
    const bool claim_slot = empty_slot || same_pair;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, rank, existing_rank);
    map.count += static_cast<uint32_t>(claim_slot && empty_slot);

    success = success || claim_slot;
    done = done || claim_slot;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline int32_t
spm_lookup_token(const emel::text::encoders::spm::action::context &ctx,
                 const std::string_view text) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool active = has_vocab && !text.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = spm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact = hash_match && spm_token_text(*ctx.vocab, id) == text;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline int32_t
spm_lookup_token_concat(const emel::text::encoders::spm::action::context &ctx,
                        const std::string_view left,
                        const std::string_view right) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool active = has_vocab && (!left.empty() || !right.empty());
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = spm_hash_concat(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  const size_t combined_len = left.size() + right.size();
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const std::string_view token = spm_token_text(*ctx.vocab, id);
    const bool size_match = token.size() == combined_len;
    const bool left_match = size_match && token.substr(0, left.size()) == left;
    const bool right_match = size_match && token.substr(left.size()) == right;
    const bool exact = hash_match && left_match && right_match;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool spm_push_token(const event::encode &ev, const int32_t token,
                           int32_t &count) noexcept {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t *base_ptrs[2] = {&sink, ev.token_ids.data()};
  int32_t *base = base_ptrs[static_cast<size_t>(has_buffer)];
  const bool non_negative_count = count >= 0;
  const int32_t safe_count = select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space =
      has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;
  const size_t target_index = count_index * static_cast<size_t>(write);
  int32_t *target = base + target_index;
  *target = select_i32(write, token, *target);
  count += static_cast<int32_t>(write);
  return write;
}

inline bool spm_push_token_if(const bool emit_token, const event::encode &ev,
                              const int32_t token, int32_t &count) noexcept {
  const bool pushed = spm_push_token(ev, token, count);
  return emit_token && pushed;
}

inline bool
spm_build_symbols(const std::string_view text,
                  emel::text::encoders::detail::encode_scratch &scratch,
                  encode_result &result) noexcept {
  scratch.symbol_count = 0;
  size_t offset = 0;
  bool ok = true;

  for (; ok && offset < text.size();) {
    const bool has_capacity = scratch.symbol_count < scratch.offsets.size();
    const size_t len_raw = emel::text::encoders::detail::utf8_len(text[offset]);
    const size_t remaining = text.size() - offset;
    const size_t len = select_size(len_raw <= remaining, len_raw, remaining);
    const size_t idx = select_size(
        has_capacity, static_cast<size_t>(scratch.symbol_count), 0u);
    scratch.offsets[idx] = select_u32(
        has_capacity, static_cast<uint32_t>(offset), scratch.offsets[idx]);
    scratch.lengths[idx] = select_u32(has_capacity, static_cast<uint32_t>(len),
                                      scratch.lengths[idx]);
    scratch.prev[idx] =
        select_i32(has_capacity, static_cast<int32_t>(scratch.symbol_count) - 1,
                   scratch.prev[idx]);
    const bool has_next = offset + len < text.size();
    const int32_t next_value = select_i32(
        has_next, static_cast<int32_t>(scratch.symbol_count) + 1, -1);
    scratch.next[idx] = select_i32(has_capacity, next_value, scratch.next[idx]);
    scratch.symbol_count += static_cast<uint32_t>(has_capacity);
    offset += len * static_cast<size_t>(has_capacity);

    ok = ok && has_capacity;
  }

  const bool patch_head = scratch.symbol_count > 0;
  scratch.prev[0] = select_i32(patch_head, -1, scratch.prev[0]);

  const std::array<int32_t, 2> errors{EMEL_ERR_INVALID_ARGUMENT, EMEL_OK};
  result.error = errors[static_cast<size_t>(ok)];
  return ok;
}

inline void
spm_merge_symbols(emel::text::encoders::detail::encode_scratch &scratch,
                  const int32_t left, const int32_t right) noexcept {
  scratch.lengths[static_cast<size_t>(left)] +=
      scratch.lengths[static_cast<size_t>(right)];
  const int32_t right_next = scratch.next[static_cast<size_t>(right)];
  scratch.next[static_cast<size_t>(left)] = right_next;
  const bool patch_next = right_next >= 0;
  const size_t safe_right_next =
      static_cast<size_t>(select_i32(patch_next, right_next, 0));
  scratch.prev[safe_right_next] =
      select_i32(patch_next, left, scratch.prev[safe_right_next]);
  scratch.lengths[static_cast<size_t>(right)] = 0;
}

inline void
spm_merge_symbols_noop(emel::text::encoders::detail::encode_scratch &,
                       const int32_t, const int32_t) noexcept {}

inline void
spm_merge_symbols_if(emel::text::encoders::detail::encode_scratch &scratch,
                     const bool has_merge, const int32_t left,
                     const int32_t right) noexcept {
  using merge_fn = void (*)(emel::text::encoders::detail::encode_scratch &,
                            int32_t, int32_t) noexcept;
  static constexpr std::array<merge_fn, 2> merge_table{
      &spm_merge_symbols_noop,
      &spm_merge_symbols,
  };
  merge_table[static_cast<size_t>(has_merge)](scratch, left, right);
}

inline bool
spm_tables_ready(const emel::text::encoders::spm::action::context &ctx,
                 const emel::model::data::vocab &vocab) noexcept {
  return ctx.tables_ready && ctx.vocab == &vocab;
}

inline bool
rebuild_spm_tables(emel::text::encoders::spm::action::context &ctx) noexcept {
  bool ok = true;
  ctx.token_to_id.clear();
  ctx.bpe_ranks.clear();
  ctx.max_token_len = 0;

  const emel::model::data::vocab &vocab = *ctx.vocab;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text =
        spm_token_text(vocab, static_cast<int32_t>(id));
    const bool inserted = spm_insert_token_map(ctx.token_to_id, vocab, text,
                                               static_cast<int32_t>(id));
    ok = ok && inserted;
    const int32_t text_len = static_cast<int32_t>(text.size());
    const bool longer = text_len > ctx.max_token_len;
    ctx.max_token_len = select_i32(longer, text_len, ctx.max_token_len);
  }

  for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
    const std::string_view merge =
        spm_merge_text(vocab, static_cast<int32_t>(idx));
    const size_t split = merge.find(' ');
    const bool has_pair = !merge.empty() && split != std::string_view::npos;
    const size_t left_len = select_size(has_pair, split, 0u);
    const size_t right_start = left_len + static_cast<size_t>(has_pair);
    const size_t right_len =
        (merge.size() - right_start) * static_cast<size_t>(has_pair);
    const std::string_view left(merge.data(), left_len);
    const std::string_view right(merge.data() + right_start, right_len);
    spm_insert_merge_map(ctx.bpe_ranks, left, right, static_cast<int32_t>(idx),
                         vocab);
  }

  ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
  ctx.tables_ready = ok;
  return ok;
}

inline bool
keep_spm_tables(emel::text::encoders::spm::action::context &ctx) noexcept {
  return ctx.tables_ready;
}

inline bool
ensure_spm_tables(emel::text::encoders::spm::action::context &ctx) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool already_ready = has_vocab && ctx.tables_ready;
  const bool needs_rebuild = has_vocab && !ctx.tables_ready;
  using rebuild_fn =
      bool (*)(emel::text::encoders::spm::action::context &) noexcept;
  static constexpr std::array<rebuild_fn, 2> rebuild_table{
      &keep_spm_tables,
      &rebuild_spm_tables,
  };
  const bool rebuild_ready =
      rebuild_table[static_cast<size_t>(needs_rebuild)](ctx);
  const bool ready = already_ready || (needs_rebuild && rebuild_ready);
  return has_vocab && ready;
}

inline bool
spm_emit_space_marker(emel::text::encoders::detail::encode_scratch &scratch,
                      size_t &out_len, const bool escape_spaces,
                      const bool emit) noexcept {
  constexpr std::array<char, 3> marker = {'\xE2', '\x96', '\x81'};
  const size_t marker_len_raw = select_size(
      escape_spaces, static_cast<size_t>(3), static_cast<size_t>(1));
  const size_t marker_len = marker_len_raw * static_cast<size_t>(emit);
  const bool has_capacity = out_len + marker_len <= scratch.buffer.size();

  for (size_t i = 0; i < marker_len_raw; ++i) {
    const bool write = emit && has_capacity && i < marker_len;
    const size_t write_index = select_size(write, out_len + i, 0u);
    const char plain = ' ';
    const int32_t escaped_i32 = static_cast<int32_t>(marker[i]);
    const int32_t plain_i32 = static_cast<int32_t>(plain);
    const int32_t value_i32 = select_i32(escape_spaces, escaped_i32, plain_i32);
    scratch.buffer[write_index] = static_cast<char>(select_i32(
        write, value_i32, static_cast<int32_t>(scratch.buffer[write_index])));
  }
  out_len += marker_len * static_cast<size_t>(has_capacity);

  return !emit || has_capacity;
}

inline bool spm_emit_char(emel::text::encoders::detail::encode_scratch &scratch,
                          size_t &out_len, const char value,
                          const bool emit) noexcept {
  const size_t write_len = static_cast<size_t>(emit);
  const bool has_capacity = out_len + write_len <= scratch.buffer.size();
  const bool write = emit && has_capacity;
  const size_t write_index = select_size(write, out_len, 0u);
  scratch.buffer[write_index] = static_cast<char>(
      select_i32(write, static_cast<int32_t>(value),
                 static_cast<int32_t>(scratch.buffer[write_index])));
  out_len += write_len * static_cast<size_t>(has_capacity);
  return !emit || has_capacity;
}

inline int32_t
spm_byte_to_token(const emel::text::encoders::spm::action::context &ctx,
                  const uint8_t byte) noexcept {
  constexpr std::array<char, 16> digits = {
      '0', '1', '2', '3', '4', '5', '6', '7',
      '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
  };

  std::array<char, 6> hex = {};
  hex[0] = '<';
  hex[1] = '0';
  hex[2] = 'x';
  hex[3] = digits[(byte >> 4u) & 0x0Fu];
  hex[4] = digits[byte & 0x0Fu];
  hex[5] = '>';

  const int32_t hex_token =
      spm_lookup_token(ctx, std::string_view(hex.data(), hex.size()));
  const char raw = static_cast<char>(byte);
  const int32_t raw_token = spm_lookup_token(ctx, std::string_view(&raw, 1));
  return select_i32(hex_token != k_token_null, hex_token, raw_token);
}

inline int32_t prepare_spm(const event::encode &ev,
                           emel::text::encoders::spm::action::context &ctx,
                           const emel::model::data::vocab &vocab,
                           const bool active) noexcept {
  size_t out_len = 0;
  const bool add_prefix =
      vocab.add_space_prefix && !vocab.treat_whitespace_as_suffix;
  const bool add_suffix =
      vocab.add_space_prefix && vocab.treat_whitespace_as_suffix;
  const bool escape_spaces = vocab.escape_whitespaces;
  bool prefix_inserted = false;
  int32_t err = EMEL_OK;

  for (const char c : ev.text) {
    const bool step_active = active && err == EMEL_OK;
    const bool prefix_now =
        step_active && add_prefix && !prefix_inserted && c != ' ';
    const bool prefix_ok =
        spm_emit_space_marker(ctx.scratch, out_len, escape_spaces, prefix_now);
    const bool prefix_fail = prefix_now && !prefix_ok;
    err = select_i32(prefix_fail, EMEL_ERR_INVALID_ARGUMENT, err);
    prefix_inserted = prefix_inserted || prefix_now;

    const bool is_space = c == ' ';
    const bool emit_space = step_active && is_space;
    const bool space_ok =
        spm_emit_space_marker(ctx.scratch, out_len, escape_spaces, emit_space);
    const bool space_fail = emit_space && !space_ok;
    err = select_i32(space_fail, EMEL_ERR_INVALID_ARGUMENT, err);

    const bool emit_char = step_active && !is_space;
    const bool char_ok = spm_emit_char(ctx.scratch, out_len, c, emit_char);
    const bool char_fail = emit_char && !char_ok;
    err = select_i32(char_fail, EMEL_ERR_INVALID_ARGUMENT, err);
  }

  const bool suffix_active = active && err == EMEL_OK;
  const bool emit_suffix = suffix_active && add_suffix;
  const bool suffix_ok =
      spm_emit_space_marker(ctx.scratch, out_len, escape_spaces, emit_suffix);
  const bool suffix_fail = emit_suffix && !suffix_ok;
  err = select_i32(suffix_fail, EMEL_ERR_INVALID_ARGUMENT, err);

  encode_result result{};
  const bool can_build = active && err == EMEL_OK;
  const size_t escaped_len = out_len * static_cast<size_t>(can_build);
  const std::string_view escaped(ctx.scratch.buffer.data(), escaped_len);
  const bool symbols_ok = spm_build_symbols(escaped, ctx.scratch, result);
  err = select_i32(can_build && !symbols_ok, result.error, err);
  return err;
}

inline int32_t prepare_spm(const event::encode &ev,
                           emel::text::encoders::spm::action::context &ctx,
                           const emel::model::data::vocab &vocab) noexcept {
  return prepare_spm(ev, ctx, vocab, true);
}

inline int32_t merge_spm(emel::text::encoders::spm::action::context &ctx,
                         const emel::model::data::vocab &vocab,
                         const bool active) noexcept {
  const std::string_view escaped(ctx.scratch.buffer.data(),
                                 ctx.scratch.buffer.size());
  const bool can_merge = active && ctx.scratch.symbol_count > 1;
  const int32_t merge_pass_limit =
      select_i32(can_merge, ctx.scratch.symbol_count - 1, 0);
  bool merge_active = can_merge;

  for (int32_t merge_pass = 0; merge_pass < merge_pass_limit; ++merge_pass) {
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_left = -1;
    int32_t best_right = -1;

    for (int32_t left = select_i32(merge_active, 0, -1); left != -1;
         left = ctx.scratch.next[static_cast<size_t>(left)]) {
      const int32_t right = ctx.scratch.next[static_cast<size_t>(left)];
      const bool has_right = right >= 0;
      const int32_t safe_right = select_i32(has_right, right, 0);
      const size_t left_off = ctx.scratch.offsets[static_cast<size_t>(left)];
      const size_t left_len = ctx.scratch.lengths[static_cast<size_t>(left)];
      const size_t right_off =
          ctx.scratch.offsets[static_cast<size_t>(safe_right)];
      const size_t right_len =
          ctx.scratch.lengths[static_cast<size_t>(safe_right)] *
          static_cast<size_t>(has_right);
      const std::string_view left_view = escaped.substr(left_off, left_len);
      const std::string_view right_view = escaped.substr(right_off, right_len);
      const int32_t token = spm_lookup_token_concat(ctx, left_view, right_view);
      const bool has_token = has_right && token != k_token_null;
      const uint32_t token_index =
          select_u32(has_token, static_cast<uint32_t>(token), 0u);
      const float score = vocab.entries[token_index].score;
      const bool better = has_token && score > best_score;
      const bool tie = has_token && score == best_score;
      const bool left_pref = best_left < 0 || left < best_left;
      const bool choose = better || (tie && left_pref);
      best_score = select_f32(choose, score, best_score);
      best_left = select_i32(choose, left, best_left);
      best_right = select_i32(choose, right, best_right);
    }

    const bool has_best = merge_active && best_left >= 0 && best_right >= 0;
    spm_merge_symbols_if(ctx.scratch, has_best, best_left, best_right);
    merge_active = merge_active && has_best;
  }

  (void)vocab;
  return EMEL_OK;
}

inline int32_t merge_spm(emel::text::encoders::spm::action::context &ctx,
                         const emel::model::data::vocab &vocab) noexcept {
  return merge_spm(ctx, vocab, true);
}

inline encode_result emit_spm(const event::encode &ev,
                              emel::text::encoders::spm::action::context &ctx,
                              const emel::model::data::vocab &vocab,
                              const bool active) noexcept {
  (void)vocab;
  const std::string_view escaped(ctx.scratch.buffer.data(),
                                 ctx.scratch.buffer.size());
  encode_result result{};
  int32_t count = 0;
  int32_t err = EMEL_OK;
  int32_t idx = select_i32(active, 0, -1);
  for (; idx != -1;) {
    const bool step_active = err == EMEL_OK;
    const bool has_symbol = step_active && ctx.scratch.lengths[static_cast<size_t>(idx)] != 0u;
    const size_t offset = ctx.scratch.offsets[static_cast<size_t>(idx)];
    const size_t length = ctx.scratch.lengths[static_cast<size_t>(idx)] *
                          static_cast<size_t>(has_symbol);
    const std::string_view symbol = escaped.substr(offset, length);
    const int32_t token = spm_lookup_token(ctx, symbol);

    const bool emit_direct = has_symbol && token != k_token_null;
    const bool direct_ok = spm_push_token_if(emit_direct, ev, token, count);
    const bool direct_fail = emit_direct && !direct_ok;
    err = select_i32(step_active && direct_fail, EMEL_ERR_INVALID_ARGUMENT, err);

    const bool emit_bytes = has_symbol && token == k_token_null;
    const size_t byte_len = symbol.size() * static_cast<size_t>(emit_bytes);
    size_t byte_limit = byte_len;
    for (size_t byte_offset = 0; byte_offset < byte_limit; ++byte_offset) {
      const unsigned char c = static_cast<unsigned char>(symbol[byte_offset]);
      const int32_t byte_token = spm_byte_to_token(ctx, c);
      const bool byte_valid = byte_token != k_token_null;
      const bool byte_ok = spm_push_token_if(byte_valid, ev, byte_token, count);
      const bool byte_fail = !byte_valid || !byte_ok;
      err = select_i32(err == EMEL_OK && byte_fail, EMEL_ERR_BACKEND, err);
      byte_limit = select_size(err == EMEL_OK, byte_limit, byte_offset + 1u);
    }

    const int32_t next_idx = ctx.scratch.next[static_cast<size_t>(idx)];
    idx = select_i32(err == EMEL_OK, next_idx, -1);
  }

  result.token_count = count * static_cast<int32_t>(err == EMEL_OK);
  result.error = err;
  return result;
}

inline encode_result emit_spm(const event::encode &ev,
                              emel::text::encoders::spm::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  return emit_spm(ev, ctx, vocab, true);
}

inline encode_result encode_spm(const event::encode &ev,
                                emel::text::encoders::spm::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  const bool non_empty = !ev.text.empty();
  const bool tables_ready = spm_tables_ready(ctx, vocab);
  const bool table_missing = non_empty && !tables_ready;
  const int32_t table_error =
      select_i32(table_missing, EMEL_ERR_INVALID_ARGUMENT, EMEL_OK);
  const bool prepare_active = non_empty && table_error == EMEL_OK;
  const int32_t prepare_error = prepare_spm(ev, ctx, vocab, prepare_active);
  const bool merge_active = prepare_active && prepare_error == EMEL_OK;
  const int32_t merge_error = merge_spm(ctx, vocab, merge_active);
  const bool emit_active = merge_active && merge_error == EMEL_OK;
  const encode_result emitted = emit_spm(ev, ctx, vocab, emit_active);

  const int32_t step_error =
      select_i32(table_error == EMEL_OK, prepare_error, table_error);
  const int32_t merge_step_error =
      select_i32(step_error == EMEL_OK, merge_error, step_error);
  const int32_t final_error =
      select_i32(merge_step_error == EMEL_OK, emitted.error, merge_step_error);

  encode_result result{};
  result.error = select_i32(non_empty, final_error, EMEL_OK);
  result.token_count = emitted.token_count *
                       static_cast<int32_t>(result.error == EMEL_OK) *
                       static_cast<int32_t>(non_empty);
  return result;
}

} // namespace emel::text::encoders::spm::detail
