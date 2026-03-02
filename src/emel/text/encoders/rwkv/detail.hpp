#pragma once

#include <cstdint>
#include <string>

#include "emel/text/encoders/rwkv/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::rwkv::detail {

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

inline uint8_t select_u8(const bool choose_true,
                         const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) | (true_value & mask));
}

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

template <class pointer_type>
inline pointer_type *select_ptr(const bool choose_true,
                                pointer_type *true_value,
                                pointer_type *false_value) noexcept {
  const uintptr_t mask = static_cast<uintptr_t>(0) - static_cast<uintptr_t>(choose_true);
  const uintptr_t t = reinterpret_cast<uintptr_t>(true_value);
  const uintptr_t f = reinterpret_cast<uintptr_t>(false_value);
  return reinterpret_cast<pointer_type *>((f & ~mask) | (t & mask));
}

inline std::string_view rwkv_token_text(const emel::model::data::vocab &vocab,
                                        const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(
    vocab.token_storage.data() + static_cast<size_t>(offset), static_cast<size_t>(length));
}

inline bool rwkv_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
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

inline bool unescape_rwkv_token(const std::string_view escaped,
                                std::string &out) {
  out.clear();
  out.reserve(escaped.size());
  bool escaping = false;
  uint8_t hex_remaining = 0;
  uint8_t hex_acc = 0;

  for (const char c : escaped) {
    bool consumed = false;

    for (bool in_hex = hex_remaining != 0; in_hex; in_hex = false) {
      const uint8_t byte = static_cast<uint8_t>(c);
      const bool alpha = byte >= static_cast<uint8_t>('a');
      const uint8_t alpha_value = static_cast<uint8_t>(byte - static_cast<uint8_t>('a') + 10u);
      const uint8_t digit_value = static_cast<uint8_t>(byte - static_cast<uint8_t>('0'));
      const uint8_t nibble = select_u8(alpha, alpha_value, digit_value);
      hex_acc = static_cast<uint8_t>((hex_acc << 4u) + nibble);
      hex_remaining = static_cast<uint8_t>(hex_remaining - 1u);
      for (bool emit_hex = hex_remaining == 0; emit_hex; emit_hex = false) {
        out.push_back(static_cast<char>(hex_acc));
        hex_acc = 0;
      }
      consumed = true;
    }

    for (bool escaped_mode = !consumed && escaping; escaped_mode; escaped_mode = false) {
      const bool esc_t = c == 't';
      const bool esc_n = c == 'n';
      const bool esc_r = c == 'r';
      const bool esc_x = c == 'x';
      char mapped = c;
      mapped = static_cast<char>(
        select_i32(esc_r, static_cast<int32_t>('\r'), static_cast<int32_t>(mapped)));
      mapped = static_cast<char>(
        select_i32(esc_n, static_cast<int32_t>('\n'), static_cast<int32_t>(mapped)));
      mapped = static_cast<char>(
        select_i32(esc_t, static_cast<int32_t>('\t'), static_cast<int32_t>(mapped)));
      for (bool emit_char = !esc_x; emit_char; emit_char = false) {
        out.push_back(mapped);
      }
      hex_remaining = select_u8(esc_x, static_cast<uint8_t>(2), hex_remaining);
      escaping = false;
      consumed = true;
    }

    for (bool begin_escape = !consumed && c == '\\'; begin_escape; begin_escape = false) {
      escaping = true;
      consumed = true;
    }

    for (bool emit_plain = !consumed; emit_plain; emit_plain = false) {
      out.push_back(c);
    }
  }
  return hex_remaining == 0;
}

inline bool rwkv_tables_ready(const emel::text::encoders::rwkv::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  return ctx.rwkv_tables_ready && ctx.rwkv_vocab == &vocab;
}

inline bool ensure_rwkv_tables(emel::text::encoders::rwkv::action::context &ctx,
                               const emel::model::data::vocab &vocab) {
  for (bool already_ready = rwkv_tables_ready(ctx, vocab);
       already_ready;
       already_ready = false) {
    return true;
  }
  ctx.rwkv_vocab = &vocab;
  ctx.rwkv_tables_ready = false;
  ctx.token_matcher = emel::text::encoders::detail::naive_trie{};

  std::string unescaped;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text = rwkv_token_text(vocab, static_cast<int32_t>(id));
    for (bool has_text = !text.empty(); has_text; has_text = false) {
      const bool unescaped_ok = unescape_rwkv_token(text, unescaped);
      for (bool unescape_fail = !unescaped_ok; unescape_fail; unescape_fail = false) {
        return false;
      }
      for (bool insert_token = !unescaped.empty(); insert_token; insert_token = false) {
        ctx.token_matcher.insert(unescaped.data(), unescaped.size(), static_cast<int32_t>(id));
      }
    }
  }
  ctx.rwkv_tables_ready = true;
  return true;
}

inline int32_t rwkv_lookup_unescaped_token(const emel::model::data::vocab &vocab,
                                           const std::string_view target) {
  int32_t resolved = k_token_null;
  std::string unescaped;
  bool done = false;
  for (uint32_t id = 0; id < vocab.n_tokens && !done; ++id) {
    const std::string_view text = rwkv_token_text(vocab, static_cast<int32_t>(id));
    for (bool has_text = !text.empty(); has_text; has_text = false) {
      const bool ok = unescape_rwkv_token(text, unescaped);
      const bool match = ok && unescaped == target;
      resolved = select_i32(match, static_cast<int32_t>(id), resolved);
      done = done || match;
    }
  }
  return resolved;
}

inline int32_t rwkv_resolve_unk_id(const emel::model::data::vocab &vocab) {
  int32_t unk_id = vocab.unk_id;
  for (bool lookup = unk_id == k_token_null; lookup; lookup = false) {
    unk_id = rwkv_lookup_unescaped_token(vocab, "<unk>");
  }
  return unk_id;
}

inline encode_result encode_rwkv(const event::encode &ev,
                                 emel::text::encoders::rwkv::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  encode_result result{};
  result.token_count = 0;
  const bool has_text = !ev.text.empty();
  const bool tables_ready = rwkv_tables_ready(ctx, vocab);
  result.error = select_i32(has_text && !tables_ready, EMEL_ERR_INVALID_ARGUMENT, EMEL_OK);

  int32_t count = 0;
  const int32_t unk_id = rwkv_resolve_unk_id(vocab);
  size_t position = 0;
  bool active = has_text && tables_ready;

  while (active && position < ev.text.size()) {
    const auto *node = ctx.token_matcher.traverse(ev.text[position]);
    int32_t token_id = unk_id;
    size_t token_end = position + 1;
    size_t offset = position + 1;
    const auto *walk = node;

    while (walk != nullptr) {
      token_id = select_i32(walk->has_value, walk->value, token_id);
      token_end = select_size(walk->has_value, offset, token_end);
      const bool can_advance = offset < ev.text.size();
      const size_t safe_index = select_size(can_advance, offset, position);
      const char next_char = ev.text[safe_index];
      const auto *next_walk = walk->traverse(next_char);
      walk = select_ptr(can_advance, next_walk, static_cast<decltype(next_walk)>(nullptr));
      offset += static_cast<size_t>(can_advance);
    }

    const bool emit_token = token_id != k_token_null;
    const bool token_push_ok = rwkv_push_token(ev, token_id, count);
    const bool push_failed = emit_token && !token_push_ok;
    result.error = select_i32(push_failed, EMEL_ERR_INVALID_ARGUMENT, result.error);
    active = active && !push_failed;
    position = token_end;
  }

  result.token_count = select_i32(result.error == EMEL_OK, count, 0);
  return result;
}

}  // namespace emel::text::encoders::rwkv::detail
