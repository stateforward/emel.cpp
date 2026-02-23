#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "emel/emel.h"
#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::action {

namespace detail {

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;

inline bool is_special_token_type(const int32_t type) noexcept {
  return type == k_token_type_control || type == k_token_type_user_defined ||
         type == k_token_type_unknown;
}

inline bool parse_hex_nibble(const char c, uint8_t & value) noexcept {
  if (c >= '0' && c <= '9') {
    value = static_cast<uint8_t>(c - '0');
    return true;
  }
  if (c >= 'a' && c <= 'f') {
    value = static_cast<uint8_t>(10 + (c - 'a'));
    return true;
  }
  if (c >= 'A' && c <= 'F') {
    value = static_cast<uint8_t>(10 + (c - 'A'));
    return true;
  }
  return false;
}

inline bool parse_plamo2_byte_token(const std::string_view piece,
                                    uint8_t & value) noexcept {
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
      piece[2] != 'x' || piece[5] != '>') {
    return false;
  }
  uint8_t hi = 0;
  uint8_t lo = 0;
  if (!parse_hex_nibble(piece[3], hi) || !parse_hex_nibble(piece[4], lo)) {
    return false;
  }
  value = static_cast<uint8_t>((hi << 4) | lo);
  return true;
}

inline size_t utf8_sequence_length(const uint8_t lead) noexcept {
  if ((lead & 0x80u) == 0u) {
    return 1;
  }
  if ((lead & 0xE0u) == 0xC0u) {
    return 2;
  }
  if ((lead & 0xF0u) == 0xE0u) {
    return 3;
  }
  if ((lead & 0xF8u) == 0xF0u) {
    return 4;
  }
  return 0;
}

inline bool is_utf8_continuation(const uint8_t value) noexcept {
  return (value & 0xC0u) == 0x80u;
}

}  // namespace detail

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void clear_request(context & ctx) noexcept {
  ctx.token_id = -1;
  ctx.emit_special = false;
  ctx.pending_bytes = nullptr;
  ctx.pending_length = 0;
  ctx.pending_capacity = 0;
  ctx.output = nullptr;
  ctx.output_capacity = 0;
  ctx.output_length = 0;
}

inline bool write_bytes(context & ctx, const char * bytes,
                        const size_t len) noexcept {
  if (len == 0) {
    return true;
  }
  if (ctx.output == nullptr || ctx.output_length + len > ctx.output_capacity) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    return false;
  }
  std::memcpy(ctx.output + ctx.output_length, bytes, len);
  ctx.output_length += len;
  return true;
}

inline bool flush_pending_complete_sequences(context & ctx) noexcept {
  while (ctx.pending_length > 0) {
    const uint8_t lead = ctx.pending_bytes[0];
    const size_t needed = detail::utf8_sequence_length(lead);
    if (needed == 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return false;
    }
    if (ctx.pending_length < needed) {
      return true;
    }
    for (size_t idx = 1; idx < needed; ++idx) {
      if (!detail::is_utf8_continuation(ctx.pending_bytes[idx])) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return false;
      }
    }

    if (!write_bytes(ctx, reinterpret_cast<const char *>(ctx.pending_bytes),
                     needed)) {
      return false;
    }

    const size_t remaining = ctx.pending_length - needed;
    if (remaining > 0) {
      std::memmove(ctx.pending_bytes, ctx.pending_bytes + needed, remaining);
    }
    ctx.pending_length = remaining;
  }

  return true;
}

struct begin_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.vocab = ev.vocab;
    ctx.is_bound = false;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    clear_request(ctx);
  }
};

struct reject_bind {
  void operator()(const event::bind &, context & ctx) const noexcept {
    ctx.is_bound = false;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct commit_bind {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.vocab == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    ctx.is_bound = true;
    ctx.last_error = EMEL_OK;
  }
};

struct begin_detokenize {
  void operator()(const event::detokenize & ev, context & ctx) const noexcept {
    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = 0;
    }
    if (ev.pending_length_out != nullptr) {
      *ev.pending_length_out = ev.pending_length;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }

    ctx.token_id = ev.token_id;
    ctx.emit_special = ev.emit_special;
    ctx.pending_bytes = ev.pending_bytes;
    ctx.pending_length = ev.pending_length;
    ctx.pending_capacity = ev.pending_capacity;
    ctx.output = ev.output;
    ctx.output_capacity = ev.output_capacity;
    ctx.output_length = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_detokenize {
  void operator()(const event::detokenize &, context & ctx) const noexcept {
    ctx.output_length = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct decode_token {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.output_length = 0;

    if (ctx.vocab == nullptr || !ctx.is_bound || ctx.pending_bytes == nullptr ||
        ctx.pending_capacity == 0 || ctx.pending_length > ctx.pending_capacity ||
        (ctx.output == nullptr && ctx.output_capacity > 0)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    if (ctx.token_id < 0 ||
        static_cast<uint32_t>(ctx.token_id) >= ctx.vocab->n_tokens) {
      set_error(ctx, EMEL_ERR_MODEL_INVALID);
      return;
    }

    const auto & entry = ctx.vocab->entries[static_cast<uint32_t>(ctx.token_id)];
    if (!ctx.emit_special && detail::is_special_token_type(entry.type)) {
      return;
    }

    const std::string_view piece(
        ctx.vocab->token_storage.data() + entry.text_offset,
        entry.text_length);

    uint8_t byte_value = 0;
    if (detail::parse_plamo2_byte_token(piece, byte_value)) {
      if (ctx.pending_length >= ctx.pending_capacity) {
        set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
      ctx.pending_bytes[ctx.pending_length] = byte_value;
      ctx.pending_length += 1;
      flush_pending_complete_sequences(ctx);
      return;
    }

    if (!flush_pending_complete_sequences(ctx)) {
      return;
    }
    if (ctx.pending_length != 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    write_bytes(ctx, piece.data(), piece.size());
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context & ctx) const noexcept {
    ctx.output_length = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr commit_bind commit_bind{};
inline constexpr begin_detokenize begin_detokenize{};
inline constexpr reject_detokenize reject_detokenize{};
inline constexpr decode_token decode_token{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::detokenizer::action
