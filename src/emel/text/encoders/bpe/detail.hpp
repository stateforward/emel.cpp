#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "emel/model/data.hpp"
#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::bpe::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

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

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t bpe_hash_bytes(const uint32_t seed,
                               const std::string_view text) noexcept {
  uint32_t hash = seed;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash != 0u, hash, 1u);
}

inline uint32_t bpe_hash_sv(const std::string_view text) noexcept {
  return bpe_hash_bytes(k_fnv_offset, text);
}

inline uint32_t bpe_hash_pair(const std::string_view left,
                              const std::string_view right) noexcept {
  const uint32_t h1 = bpe_hash_sv(left);
  const uint32_t h2 = bpe_hash_sv(right);
  const uint32_t mixed = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  return select_u32(mixed != 0u, mixed, 1u);
}

inline std::string_view bpe_token_text(const emel::model::data::vocab &vocab,
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

inline std::string_view bpe_merge_text(const emel::model::data::vocab &vocab,
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

inline bool bpe_merge_match(const std::string_view merge,
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

inline bool bpe_insert_token_map(emel::text::encoders::detail::token_map &map,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view text,
                                 const int32_t id) noexcept {
  const bool text_empty = text.empty();
  bool done = text_empty;
  bool success = text_empty;
  const uint32_t hash = bpe_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_id = map.values[slot];
    const std::string_view existing_text = bpe_token_text(vocab, existing_id);
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
bpe_insert_merge_map(emel::text::encoders::detail::merge_map &map,
                     const std::string_view left, const std::string_view right,
                     const int32_t rank,
                     const emel::model::data::vocab &vocab) noexcept {
  const bool active = !left.empty() && !right.empty();
  bool done = !active;
  bool success = false;
  const uint32_t hash = bpe_hash_pair(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_merge_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_rank = map.values[slot];
    const std::string_view merge = bpe_merge_text(vocab, existing_rank);
    const bool same_pair = hash_match && bpe_merge_match(merge, left, right);
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
bpe_lookup_token(const emel::text::encoders::bpe::action::context &ctx,
                 const std::string_view text) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const emel::model::data::vocab * vocab = ctx.vocab;
  const bool active = has_vocab && !text.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = bpe_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact = hash_match && bpe_token_text(*vocab, id) == text;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline int32_t
bpe_lookup_merge_rank(const emel::text::encoders::bpe::action::context &ctx,
                      const emel::model::data::vocab &vocab,
                      const std::string_view left,
                      const std::string_view right) noexcept {
  const bool active = !left.empty() && !right.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = bpe_hash_pair(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_merge_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.bpe_ranks.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t rank = ctx.bpe_ranks.values[slot];
    const std::string_view merge = bpe_merge_text(vocab, rank);
    const bool exact = hash_match && bpe_merge_match(merge, left, right);

    resolved = select_i32(exact, rank, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool bpe_push_token(const event::encode &ev, const int32_t token,
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

inline bool
bpe_build_symbols(const std::string_view text,
                  emel::text::encoders::detail::encode_scratch &scratch,
                  encode_result &result) noexcept {
  scratch.symbol_count = 0;
  size_t offset = 0;
  bool ok = true;

  for (; offset < text.size();) {
    const bool has_capacity = scratch.symbol_count < scratch.offsets.size();
    const size_t len_raw = emel::text::unicode_len_utf8(text[offset]);
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
    scratch.symbol_count += static_cast<int32_t>(has_capacity);
    offset += len;

    ok = ok && has_capacity;
  }

  const bool patch_head = scratch.symbol_count > 0;
  scratch.prev[0] = select_i32(patch_head, -1, scratch.prev[0]);

  const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
  result.error = errors[static_cast<size_t>(ok)];
  return ok;
}

inline void
bpe_merge_symbols(emel::text::encoders::detail::encode_scratch &scratch,
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
bpe_merge_symbols_noop(emel::text::encoders::detail::encode_scratch &,
                       const int32_t, const int32_t) noexcept {}

inline void
bpe_merge_symbols_if(emel::text::encoders::detail::encode_scratch &scratch,
                     const bool has_merge, const int32_t left,
                     const int32_t right) noexcept {
  using merge_fn = void (*)(emel::text::encoders::detail::encode_scratch &,
                            int32_t, int32_t) noexcept;
  static constexpr std::array<merge_fn, 2> merge_table{
      &bpe_merge_symbols_noop,
      &bpe_merge_symbols,
  };
  merge_table[static_cast<size_t>(has_merge)](scratch, left, right);
}

inline bool bpe_push_token_if(const bool emit_token, const event::encode &ev,
                              const int32_t token, int32_t &count) noexcept {
  const bool pushed = bpe_push_token(ev, token, count);
  return emit_token && pushed;
}

inline bool
rebuild_bpe_tables(emel::text::encoders::bpe::action::context &ctx) noexcept {
  bool ok = true;
  ctx.token_to_id.clear();
  ctx.bpe_ranks.clear();
  ctx.max_token_len = 0;

  const emel::model::data::vocab &vocab = *ctx.vocab;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text =
        bpe_token_text(vocab, static_cast<int32_t>(id));
    const bool inserted = bpe_insert_token_map(ctx.token_to_id, vocab, text,
                                               static_cast<int32_t>(id));
    ok = ok && inserted;
    const int32_t text_len = static_cast<int32_t>(text.size());
    const bool longer = text_len > ctx.max_token_len;
    ctx.max_token_len = select_i32(longer, text_len, ctx.max_token_len);
  }

  for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
    const std::string_view merge =
        bpe_merge_text(vocab, static_cast<int32_t>(idx));
    const size_t split = merge.find(' ');
    const bool has_pair = !merge.empty() && split != std::string_view::npos;
    const size_t left_len = select_size(has_pair, split, 0u);
    const size_t right_start = left_len + static_cast<size_t>(has_pair);
    const size_t right_len =
        (merge.size() - right_start) * static_cast<size_t>(has_pair);
    const std::string_view left(merge.data(), left_len);
    const std::string_view right(merge.data() + right_start, right_len);
    bpe_insert_merge_map(ctx.bpe_ranks, left, right, static_cast<int32_t>(idx),
                         vocab);
  }

  ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
  ctx.tables_ready = ok;
  return ok;
}

inline bool
keep_bpe_tables(emel::text::encoders::bpe::action::context &ctx) noexcept {
  return ctx.tables_ready;
}

inline bool
ensure_bpe_tables(emel::text::encoders::bpe::action::context &ctx) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool already_ready = has_vocab && ctx.tables_ready;
  const bool needs_rebuild = has_vocab && !ctx.tables_ready;
  using rebuild_fn =
      bool (*)(emel::text::encoders::bpe::action::context &) noexcept;
  static constexpr std::array<rebuild_fn, 2> rebuild_table{
      &keep_bpe_tables,
      &rebuild_bpe_tables,
  };
  const bool rebuild_ready =
      rebuild_table[static_cast<size_t>(needs_rebuild)](ctx);
  const bool ready = already_ready || (needs_rebuild && rebuild_ready);
  return has_vocab && ready;
}

inline bool encode_bpe_word_merge_path(
    const event::encode &ev, emel::text::encoders::bpe::action::context &ctx,
    const emel::model::data::vocab &vocab, const std::string_view word,
    int32_t &count, encode_result &result) {
  bool ok = bpe_build_symbols(word, ctx.scratch, result);

  const bool can_merge = ok && ctx.scratch.symbol_count > 1;
  const int32_t merge_pass_limit =
      select_i32(can_merge, ctx.scratch.symbol_count - 1, 0);
  bool merge_active = can_merge;

  for (int32_t merge_pass = 0; merge_pass < merge_pass_limit; ++merge_pass) {
    int32_t best_left = -1;
    int32_t best_right = -1;
    int32_t best_rank = std::numeric_limits<int32_t>::max();

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
      const std::string_view left_view(word.data() + left_off, left_len);
      const std::string_view right_view(word.data() + right_off, right_len);
      const int32_t rank =
          bpe_lookup_merge_rank(ctx, vocab, left_view, right_view);
      const bool has_rank = has_right && rank != k_token_null;
      const bool better = has_rank && (rank < best_rank ||
                                       (rank == best_rank && left < best_left));
      best_rank = select_i32(better, rank, best_rank);
      best_left = select_i32(better, left, best_left);
      best_right = select_i32(better, right, best_right);
    }

    const bool has_merge = merge_active && best_left >= 0 && best_right >= 0;
    bpe_merge_symbols_if(ctx.scratch, has_merge, best_left, best_right);
    merge_active = merge_active && has_merge;
  }

  const bool has_symbol_chain = ok && ctx.scratch.symbol_count > 0;
  const int32_t first_symbol = select_i32(has_symbol_chain, 0, -1);
  int32_t idx = first_symbol;
  for (; idx != -1;) {
    const bool step_active = ok;
    const bool has_symbol = step_active && ctx.scratch.lengths[static_cast<size_t>(idx)] > 0;
    const size_t sym_off = ctx.scratch.offsets[static_cast<size_t>(idx)];
    const size_t sym_len = ctx.scratch.lengths[static_cast<size_t>(idx)];
    const std::string_view symbol(word.data() + sym_off,
                                  sym_len * static_cast<size_t>(has_symbol));
    const int32_t token = bpe_lookup_token(ctx, symbol);
    const bool direct_hit = has_symbol && token != k_token_null;
    const bool direct_pushed = bpe_push_token_if(direct_hit, ev, token, count);
    ok = ok && (!direct_hit || direct_pushed);

    const size_t byte_limit = select_size(step_active && !direct_hit, symbol.size(), 0u);
    for (size_t byte_offset = 0; byte_offset < byte_limit;) {
      size_t len = emel::text::unicode_len_utf8(symbol[byte_offset]);
      const size_t remaining = symbol.size() - byte_offset;
      len = select_size(len <= remaining, len, static_cast<size_t>(1));
      const std::string_view unit(symbol.data() + byte_offset, len);
      const int32_t byte_token = bpe_lookup_token(ctx, unit);
      const bool emit_byte = byte_token != k_token_null;
      const bool byte_pushed =
          bpe_push_token_if(emit_byte, ev, byte_token, count);
      const bool step_ok = !emit_byte || byte_pushed;
      ok = ok && step_ok;
      byte_offset += len;
    }

    const int32_t next_idx = ctx.scratch.next[static_cast<size_t>(idx)];
    idx = next_idx;
  }

  const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
  result.error = errors[static_cast<size_t>(ok)];
  return ok;
}

inline encode_result
encode_bpe_ignore_merges(const event::encode &ev,
                         emel::text::encoders::bpe::action::context &ctx) {
  encode_result result{};
  int32_t count = 0;
  const int32_t token = bpe_lookup_token(ctx, ev.text);
  const bool token_found = token != k_token_null;
  const bool token_pushed = bpe_push_token_if(token_found, ev, token, count);

  const size_t error_index = (static_cast<size_t>(token_found) << 1u) |
                             static_cast<size_t>(token_pushed);
  const std::array<int32_t, 4> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend),
                                      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
  result.error = errors[error_index];
  result.token_count = count * static_cast<int32_t>(result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok));
  return result;
}

inline encode_result
encode_bpe_merge_path(const event::encode &ev,
                      emel::text::encoders::bpe::action::context &ctx,
                      const emel::model::data::vocab &vocab) {
  encode_result result{};
  int32_t count = 0;
  const bool ok =
      encode_bpe_word_merge_path(ev, ctx, vocab, ev.text, count, result);
  const std::array<int32_t, 2> errors{result.error, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
  result.error = errors[static_cast<size_t>(ok)];
  result.token_count = count * static_cast<int32_t>(ok);
  return result;
}

inline encode_result
encode_bpe_ignore_or_merge(const event::encode &ev,
                           emel::text::encoders::bpe::action::context &ctx,
                           const emel::model::data::vocab &vocab) {
  using fallback_fn = encode_result (*)(
      const event::encode &, emel::text::encoders::bpe::action::context &,
      const emel::model::data::vocab &, const encode_result &);
  static constexpr std::array<fallback_fn, 2> fallback_table{
      +[](const event::encode &, emel::text::encoders::bpe::action::context &,
          const emel::model::data::vocab &,
          const encode_result &current) { return current; },
      +[](const event::encode &value_ev,
          emel::text::encoders::bpe::action::context &value_ctx,
          const emel::model::data::vocab &value_vocab, const encode_result &) {
        return encode_bpe_merge_path(value_ev, value_ctx, value_vocab);
      },
  };
  encode_result result = encode_bpe_ignore_merges(ev, ctx);
  const bool fallback = result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
  return fallback_table[static_cast<size_t>(fallback)](ev, ctx, vocab, result);
}

inline encode_result encode_bpe_backend_error() {
  encode_result result{};
  result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend);
  return result;
}

inline encode_result encode_bpe_invalid_preprocessed() {
  encode_result result{};
  result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
  return result;
}

inline encode_result encode_bpe_dispatch_preprocessed(
    const event::encode &ev, emel::text::encoders::bpe::action::context &ctx,
    const emel::model::data::vocab &vocab) {
  using path_fn = encode_result (*)(
      const event::encode &, emel::text::encoders::bpe::action::context &,
      const emel::model::data::vocab &);
  static constexpr std::array<path_fn, 2> path_table{
      encode_bpe_merge_path,
      encode_bpe_ignore_or_merge,
  };
  return path_table[static_cast<size_t>(vocab.ignore_merges)](ev, ctx, vocab);
}

inline encode_result
encode_bpe_dispatch_table_state(const event::encode &ev,
                                emel::text::encoders::bpe::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  using preprocessed_fn = encode_result (*)(
      const event::encode &, emel::text::encoders::bpe::action::context &,
      const emel::model::data::vocab &);
  static constexpr std::array<preprocessed_fn, 2> preprocessed_table{
      +[](const event::encode &, emel::text::encoders::bpe::action::context &,
          const emel::model::data::vocab &) {
        return encode_bpe_invalid_preprocessed();
      },
      +[](const event::encode &value_ev,
          emel::text::encoders::bpe::action::context &value_ctx,
          const emel::model::data::vocab &value_vocab) {
        return encode_bpe_dispatch_preprocessed(value_ev, value_ctx,
                                                value_vocab);
      },
  };
  return preprocessed_table[static_cast<size_t>(ev.preprocessed)](ev, ctx,
                                                                  vocab);
}

inline encode_result
encode_bpe_non_empty(const event::encode &ev,
                     emel::text::encoders::bpe::action::context &ctx,
                     const emel::model::data::vocab &vocab) {
  const bool tables_ready = ensure_bpe_tables(ctx);
  using table_fn = encode_result (*)(
      const event::encode &, emel::text::encoders::bpe::action::context &,
      const emel::model::data::vocab &);
  static constexpr std::array<table_fn, 2> table_state_paths{
      +[](const event::encode &, emel::text::encoders::bpe::action::context &,
          const emel::model::data::vocab &) {
        return encode_bpe_backend_error();
      },
      +[](const event::encode &value_ev,
          emel::text::encoders::bpe::action::context &value_ctx,
          const emel::model::data::vocab &value_vocab) {
        return encode_bpe_dispatch_table_state(value_ev, value_ctx,
                                               value_vocab);
      },
  };
  return table_state_paths[static_cast<size_t>(tables_ready)](ev, ctx, vocab);
}

inline encode_result
encode_bpe_empty(const event::encode &,
                 emel::text::encoders::bpe::action::context &,
                 const emel::model::data::vocab &) {
  return encode_result{};
}

inline encode_result encode_bpe(const event::encode &ev,
                                emel::text::encoders::bpe::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  using encode_fn = encode_result (*)(
      const event::encode &, emel::text::encoders::bpe::action::context &,
      const emel::model::data::vocab &);
  static constexpr std::array<encode_fn, 2> encode_paths{
      &encode_bpe_empty,
      &encode_bpe_non_empty,
  };
  return encode_paths[static_cast<size_t>(!ev.text.empty())](ev, ctx, vocab);
}

} // namespace emel::text::encoders::bpe::detail
