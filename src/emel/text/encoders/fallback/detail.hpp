#pragma once

#include <cstdint>
#include <string_view>

#include "emel/text/encoders/fallback/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::fallback::detail {

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

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t fallback_hash_bytes(const uint32_t seed,
                                    const std::string_view text) noexcept {
  uint32_t hash = seed;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash != 0u, hash, 1u);
}

inline uint32_t fallback_hash_sv(const std::string_view text) noexcept {
  return fallback_hash_bytes(k_fnv_offset, text);
}

inline std::string_view fallback_token_text(const emel::model::data::vocab &vocab,
                                            const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(
    vocab.token_storage.data() + static_cast<size_t>(offset),
    static_cast<size_t>(length));
}

inline bool fallback_insert_token_map(emel::text::encoders::detail::token_map &map,
                                      const emel::model::data::vocab &vocab,
                                      const std::string_view text,
                                      const int32_t id) noexcept {
  const bool text_empty = text.empty();
  bool done = text_empty;
  bool success = text_empty;
  const uint32_t hash = fallback_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_id = map.values[slot];
    const std::string_view existing_text = fallback_token_text(vocab, existing_id);
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

inline bool ensure_fallback_tables(emel::text::encoders::action::context &ctx,
                                   const emel::model::data::vocab &vocab) noexcept {
  auto rebuild_none = [](emel::text::encoders::action::context &,
                         const emel::model::data::vocab &,
                         bool &) noexcept {};
  auto rebuild_some = [](emel::text::encoders::action::context &ctx_value,
                         const emel::model::data::vocab &vocab_value,
                         bool &ok_value) noexcept {
    ctx_value.vocab = &vocab_value;
    ctx_value.tables_ready = false;
    ctx_value.token_to_id.clear();
    ctx_value.bpe_ranks.clear();
    ctx_value.max_token_len = 0;

    for (uint32_t id = 0; id < vocab_value.n_tokens; ++id) {
      const std::string_view text =
          fallback_token_text(vocab_value, static_cast<int32_t>(id));
      const bool inserted = fallback_insert_token_map(
          ctx_value.token_to_id, vocab_value, text, static_cast<int32_t>(id));
      ok_value = ok_value && inserted;
      const int32_t text_len = static_cast<int32_t>(text.size());
      const bool longer = text_len > ctx_value.max_token_len;
      ctx_value.max_token_len = select_i32(longer, text_len, ctx_value.max_token_len);
    }

    ctx_value.tables_ready = ok_value;
  };

  const bool already_ready = ctx.tables_ready && ctx.vocab == &vocab;
  bool ok = true;
  using rebuild_handler_t = void (*)(emel::text::encoders::action::context &,
                                     const emel::model::data::vocab &,
                                     bool &) noexcept;
  const rebuild_handler_t rebuild_handlers[2] = {rebuild_none, rebuild_some};
  rebuild_handlers[static_cast<size_t>(!already_ready)](ctx, vocab, ok);

  return already_ready || ctx.tables_ready;
}

inline int32_t fallback_lookup_token(const emel::text::encoders::action::context &ctx,
                                     const emel::model::data::vocab &vocab,
                                     const std::string_view text) noexcept {
  const bool active = !text.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = fallback_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact = hash_match && fallback_token_text(vocab, id) == text;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool fallback_push_token(const event::encode &ev,
                                const int32_t token,
                                int32_t &count) noexcept {
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

inline encode_result encode_fallback_exec(const event::encode &ev,
                                          emel::text::encoders::action::context &ctx,
                                          const emel::model::data::vocab &vocab) {
  auto finalize_success = [](encode_result &result_value, const int32_t count_value) {
    result_value.token_count = count_value;
    result_value.error = EMEL_OK;
  };
  auto finalize_failure = [](encode_result &result_value, const int32_t) {
    result_value.token_count = 0;
    result_value.error = EMEL_ERR_BACKEND;
  };

  encode_result result{};
  result.token_count = 0;

  int32_t count = 0;
  bool failed = false;
  for (size_t i = 0; i < ev.text.size() && !failed; ++i) {
    const unsigned char byte = static_cast<unsigned char>(ev.text[i]);
    const char raw = static_cast<char>(byte);
    const int32_t token = fallback_lookup_token(ctx, vocab, std::string_view(&raw, 1));
    const bool pushed = fallback_push_token(ev, token, count);
    const bool ok = token != k_token_null && pushed;
    failed = failed || !ok;
  }

  using finalize_handler_t = void (*)(encode_result &, int32_t);
  const finalize_handler_t finalize_handlers[2] = {finalize_success, finalize_failure};
  finalize_handlers[static_cast<size_t>(failed)](result, count);
  return result;
}

inline encode_result encode_fallback_empty_text(
    const event::encode &,
    emel::text::encoders::action::context &,
    const emel::model::data::vocab &) {
  encode_result result{};
  result.token_count = 0;
  result.error = EMEL_OK;
  return result;
}

inline encode_result encode_fallback_missing_tables(
    const event::encode &,
    emel::text::encoders::action::context &,
    const emel::model::data::vocab &) {
  encode_result result{};
  result.token_count = 0;
  result.error = EMEL_ERR_INVALID_ARGUMENT;
  return result;
}

inline encode_result encode_fallback_ready_tables(
    const event::encode &ev,
    emel::text::encoders::action::context &ctx,
    const emel::model::data::vocab &vocab) {
  return encode_fallback_exec(ev, ctx, vocab);
}

inline encode_result encode_fallback_non_empty_text(
    const event::encode &ev,
    emel::text::encoders::action::context &ctx,
    const emel::model::data::vocab &vocab) {
  const bool tables_ready = ctx.tables_ready && ctx.vocab == &vocab;
  using table_handler_t = encode_result (*)(const event::encode &,
                                            emel::text::encoders::action::context &,
                                            const emel::model::data::vocab &);
  const table_handler_t table_handlers[2] = {
      encode_fallback_ready_tables,
      encode_fallback_missing_tables,
  };
  return table_handlers[static_cast<size_t>(!tables_ready)](ev, ctx, vocab);
}

inline encode_result encode_fallback(const event::encode &ev,
                                     emel::text::encoders::action::context &ctx,
                                     const emel::model::data::vocab &vocab) {
  using empty_handler_t = encode_result (*)(const event::encode &,
                                            emel::text::encoders::action::context &,
                                            const emel::model::data::vocab &);
  const empty_handler_t empty_handlers[2] = {
      encode_fallback_non_empty_text,
      encode_fallback_empty_text,
  };
  return empty_handlers[static_cast<size_t>(ev.text.empty())](ev, ctx, vocab);
}

}  // namespace emel::text::encoders::fallback::detail
