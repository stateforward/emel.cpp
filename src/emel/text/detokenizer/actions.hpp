#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/errors.hpp"
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
  const bool is_digit = c >= '0' && c <= '9';
  const bool is_lower = c >= 'a' && c <= 'f';
  const bool is_upper = c >= 'A' && c <= 'F';
  const uint8_t digit_value = static_cast<uint8_t>(c - '0');
  const uint8_t lower_value = static_cast<uint8_t>(10 + (c - 'a'));
  const uint8_t upper_value = static_cast<uint8_t>(10 + (c - 'A'));
  value = static_cast<uint8_t>(static_cast<uint8_t>(is_digit) * digit_value +
                               static_cast<uint8_t>(is_lower) * lower_value +
                               static_cast<uint8_t>(is_upper) * upper_value);
  return is_digit || is_lower || is_upper;
}

inline bool parse_plamo2_byte_token(const std::string_view piece,
                                    uint8_t & value) noexcept {
  const bool format_ok = piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
                         piece[2] == 'x' && piece[5] == '>';
  uint8_t hi = 0;
  uint8_t lo = 0;
  const bool nibbles_ok =
      format_ok && parse_hex_nibble(piece[3], hi) && parse_hex_nibble(piece[4], lo);
  value = static_cast<uint8_t>((hi << 4) | lo);
  return nibbles_ok;
}

inline size_t utf8_sequence_length(const uint8_t lead) noexcept {
  const bool one = (lead & 0x80u) == 0u;
  const bool two = (lead & 0xE0u) == 0xC0u;
  const bool three = (lead & 0xF0u) == 0xE0u;
  const bool four = (lead & 0xF8u) == 0xF0u;
  return static_cast<size_t>(one) +
         static_cast<size_t>(two) * 2u +
         static_cast<size_t>(three) * 3u +
         static_cast<size_t>(four) * 4u;
}

inline bool is_utf8_continuation(const uint8_t value) noexcept {
  return (value & 0xC0u) == 0x80u;
}

inline void clear_request(context &) noexcept {}

inline size_t read_output_length(const event::detokenize & ev) noexcept {
  return ev.output_length_out;
}

inline size_t read_pending_length(const event::detokenize & ev) noexcept {
  return ev.pending_length_out;
}

inline void set_bind_error(const event::bind & ev, const int32_t err) noexcept {
  ev.error_out = err;
}

inline void set_detokenize_error(const event::detokenize & ev,
                                 const int32_t err,
                                 const size_t output_length,
                                 const size_t pending_length) noexcept {
  ev.output_length_out = output_length;
  ev.pending_length_out = pending_length;
  ev.error_out = err;
}

inline bool write_bytes(const event::detokenize & ev,
                        size_t & output_length,
                        const size_t pending_length,
                        const char * bytes,
                        const size_t len) noexcept {
  const bool has_payload = len != 0;
  const bool writable = !has_payload || (ev.output != nullptr && output_length + len <= ev.output_capacity);
  while (!writable) {
    set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
    break;
  }
  while (writable && has_payload) {
    std::memcpy(ev.output + output_length, bytes, len);
    break;
  }
  output_length += len * static_cast<size_t>(writable && has_payload);
  return writable;
}

inline bool flush_pending_complete_sequences(const event::detokenize & ev,
                                             size_t & pending_length,
                                             size_t & output_length) noexcept {
  bool ok = true;
  bool write_failed = false;
  bool needs_more_bytes = false;

  while (pending_length > 0 && ok && !needs_more_bytes) {
    const uint8_t lead = ev.pending_bytes[0];
    const size_t needed = utf8_sequence_length(lead);
    const bool lead_ok = needed != 0;
    ok = ok && lead_ok;

    const bool sequence_ready = ok && pending_length >= needed;
    needs_more_bytes = ok && !sequence_ready;

    bool continuation_ok = true;
    size_t idx = 1;
    while (idx < needed && sequence_ready && continuation_ok) {
      continuation_ok = continuation_ok && is_utf8_continuation(ev.pending_bytes[idx]);
      ++idx;
    }
    ok = ok && (!sequence_ready || continuation_ok);

    bool wrote = true;
    const bool write_candidate = sequence_ready && continuation_ok;
    while (write_candidate) {
      wrote = write_bytes(
          ev, output_length, pending_length, reinterpret_cast<const char *>(ev.pending_bytes), needed);
      break;
    }
    write_failed = write_failed || (write_candidate && !wrote);
    ok = ok && (!write_candidate || wrote);

    const size_t consumed = needed * static_cast<size_t>(write_candidate && wrote);
    const size_t remaining = pending_length - consumed;
    while (consumed != 0 && remaining > 0) {
      std::memmove(ev.pending_bytes, ev.pending_bytes + consumed, remaining);
      break;
    }
    pending_length = remaining;
  }

  while (!ok && !write_failed) {
    set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
    break;
  }

  return ok;
}

inline void begin_bind(const event::bind & ev, context & ctx) noexcept {
  set_bind_error(ev, error_code(error::none));
  ctx.vocab = &ev.vocab;
  ctx.is_bound = false;
}

inline void reject_bind(const event::bind & ev, context & ctx) noexcept {
  ctx.is_bound = false;
  set_bind_error(ev, error_code(error::invalid_request));
}

inline void commit_bind(const event::bind & ev, context & ctx) noexcept {
  ctx.is_bound = true;
  set_bind_error(ev, error_code(error::none));
}

inline void notify_bind_done(const event::bind & ev) noexcept {
  (void)ev.dispatch_done(ev.owner_sm, events::binding_done{ev});
}

inline void notify_bind_error(const event::bind & ev) noexcept {
  (void)ev.dispatch_error(ev.owner_sm, events::binding_error{ev, ev.error_out});
}

inline void begin_detokenize(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev, error_code(error::none), 0, ev.pending_length);
}

inline void reject_detokenize(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev, error_code(error::invalid_request), 0, ev.pending_length);
}

inline void decode_token(const event::detokenize & ev,
                         const context & ctx) noexcept {
  size_t pending_length = ev.pending_length;
  size_t output_length = 0;
  set_detokenize_error(ev, error_code(error::none), output_length, pending_length);

  const bool request_ok =
      ctx.vocab != nullptr && ctx.is_bound && ev.pending_bytes != nullptr &&
      ev.pending_capacity > 0 && pending_length <= ev.pending_capacity &&
      (ev.output != nullptr || ev.output_capacity == 0);
  while (!request_ok) {
    set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
    break;
  }

  const bool token_ok =
      request_ok && ev.token_id >= 0 && static_cast<uint32_t>(ev.token_id) < ctx.vocab->n_tokens;
  while (request_ok && !token_ok) {
    set_detokenize_error(ev, error_code(error::model_invalid), output_length, pending_length);
    break;
  }

  while (token_ok) {
    const auto & entry = ctx.vocab->entries[static_cast<uint32_t>(ev.token_id)];
    const bool skip_special = !ev.emit_special && is_special_token_type(entry.type);
    while (skip_special) {
      set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
      break;
    }

    const bool decode_piece = !skip_special;
    while (decode_piece) {
      const std::string_view piece(ctx.vocab->token_storage.data() + entry.text_offset,
                                   entry.text_length);

      uint8_t byte_value = 0;
      const bool byte_piece = parse_plamo2_byte_token(piece, byte_value);

      const bool byte_capacity_ok = !byte_piece || pending_length < ev.pending_capacity;
      while (byte_piece && !byte_capacity_ok) {
        set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
        break;
      }

      const bool byte_path = byte_piece && byte_capacity_ok;
      while (byte_path) {
        ev.pending_bytes[pending_length] = byte_value;
        break;
      }
      pending_length += static_cast<size_t>(byte_path);

      bool byte_flush_ok = true;
      while (byte_path) {
        byte_flush_ok = flush_pending_complete_sequences(ev, pending_length, output_length);
        break;
      }
      const bool byte_done = byte_path && byte_flush_ok;
      while (byte_done) {
        set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
        break;
      }

      const bool text_path = !byte_piece;
      bool text_flush_ok = true;
      while (text_path) {
        text_flush_ok = flush_pending_complete_sequences(ev, pending_length, output_length);
        break;
      }

      const bool text_ready = text_path && text_flush_ok;
      const bool pending_empty = text_ready && pending_length == 0;
      while (text_ready && !pending_empty) {
        set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
        break;
      }

      bool wrote_text = true;
      while (pending_empty) {
        wrote_text = write_bytes(ev, output_length, pending_length, piece.data(), piece.size());
        break;
      }
      while (pending_empty && wrote_text) {
        set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
        break;
      }
      break;
    }
    break;
  }
}

inline void mark_done(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::none),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void notify_detokenize_done(const event::detokenize & ev) noexcept {
  (void)ev.dispatch_done(
      ev.owner_sm,
      events::detokenize_done{ev, ev.output_length_out, ev.pending_length_out});
}

inline void notify_detokenize_error(const event::detokenize & ev) noexcept {
  (void)ev.dispatch_error(ev.owner_sm, events::detokenize_error{ev, ev.error_out});
}

template <class event_type>
inline void on_unexpected(const event_type & ev) noexcept {
  (void)ev;
}

}  // namespace detail

inline void clear_request(context & ctx) noexcept {
  detail::clear_request(ctx);
}

inline size_t read_output_length(const event::detokenize & ev) noexcept {
  return detail::read_output_length(ev);
}

inline size_t read_pending_length(const event::detokenize & ev) noexcept {
  return detail::read_pending_length(ev);
}

inline bool write_bytes(const event::detokenize & ev,
                        size_t & output_length,
                        const size_t pending_length,
                        const char * bytes,
                        const size_t len) noexcept {
  return detail::write_bytes(ev, output_length, pending_length, bytes, len);
}

inline bool flush_pending_complete_sequences(const event::detokenize & ev,
                                             size_t & pending_length,
                                             size_t & output_length) noexcept {
  return detail::flush_pending_complete_sequences(ev, pending_length, output_length);
}

struct begin_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    detail::begin_bind(ev, ctx);
  }
};

struct reject_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    detail::reject_bind(ev, ctx);
  }
};

struct commit_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    detail::commit_bind(ev, ctx);
  }
};

struct notify_bind_done {
  void operator()(const event::bind & ev) const noexcept {
    detail::notify_bind_done(ev);
  }
};

struct notify_bind_error {
  void operator()(const event::bind & ev) const noexcept {
    detail::notify_bind_error(ev);
  }
};

struct begin_detokenize {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::begin_detokenize(ev);
  }
};

struct reject_detokenize {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::reject_detokenize(ev);
  }
};

struct decode_token {
  void operator()(const event::detokenize & ev, const context & ctx) const noexcept {
    detail::decode_token(ev, ctx);
  }
};

struct mark_done {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_done(ev);
  }
};

struct notify_detokenize_done {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::notify_detokenize_done(ev);
  }
};

struct notify_detokenize_error {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::notify_detokenize_error(ev);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    detail::on_unexpected(ev);
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr commit_bind commit_bind{};
inline constexpr begin_detokenize begin_detokenize{};
inline constexpr reject_detokenize reject_detokenize{};
inline constexpr decode_token decode_token{};
inline constexpr mark_done mark_done{};
inline constexpr notify_bind_done notify_bind_done{};
inline constexpr notify_bind_error notify_bind_error{};
inline constexpr notify_detokenize_done notify_detokenize_done{};
inline constexpr notify_detokenize_error notify_detokenize_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::detokenizer::action
