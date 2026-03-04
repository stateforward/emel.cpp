#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::action {

namespace detail {

constexpr int32_t k_token_type_unknown = 2;
constexpr int32_t k_token_type_control = 3;
constexpr int32_t k_token_type_user_defined = 4;
constexpr size_t k_utf8_max_sequence_length = 4;

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
  return static_cast<size_t>(one) + static_cast<size_t>(two) * 2u +
         static_cast<size_t>(three) * 3u + static_cast<size_t>(four) * 4u;
}

inline bool is_utf8_continuation(const uint8_t value) noexcept {
  return (value & 0xC0u) == 0x80u;
}

inline std::string_view token_piece(const event::detokenize & ev,
                                    const context & ctx) noexcept {
  const auto & entry = ctx.vocab->entries[static_cast<uint32_t>(ev.token_id)];
  return std::string_view(ctx.vocab->token_storage.data() + entry.text_offset,
                          entry.text_length);
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
  const bool writable = !has_payload ||
                        (ev.output != nullptr && output_length + len <= ev.output_capacity);
  const int32_t error_value =
      static_cast<int32_t>(!writable) * error_code(error::invalid_request) +
      static_cast<int32_t>(writable) * ev.error_out;
  set_detokenize_error(ev, error_value, output_length, pending_length);

  char scratch = 0;
  const std::array<char *, 2> output_candidates = {&scratch, ev.output};
  char * output_ptr = output_candidates[static_cast<size_t>(ev.output != nullptr)];
  const size_t write_len = len * static_cast<size_t>(writable && has_payload);
  for (size_t i = 0; i < write_len; ++i) {
    output_ptr[output_length + i] = bytes[i];
  }
  output_length += write_len;
  return writable;
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
  set_detokenize_error(ev, error_code(error::none), 0u, ev.pending_length);
}

inline void reject_detokenize(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev, error_code(error::invalid_request), 0u, ev.pending_length);
}

inline void mark_model_invalid(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::model_invalid),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void mark_invalid_pending_full(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::invalid_request),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void mark_invalid_pending_not_empty(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::invalid_request),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void mark_invalid_pending_sequence(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::invalid_request),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void mark_internal_error(const event::detokenize & ev) noexcept {
  set_detokenize_error(ev,
                       error_code(error::internal_error),
                       read_output_length(ev),
                       read_pending_length(ev));
}

inline void append_byte_piece(const event::detokenize & ev,
                              const context & ctx) noexcept {
  const std::string_view piece = token_piece(ev, ctx);
  uint8_t byte_value = 0;
  const bool parsed = parse_plamo2_byte_token(piece, byte_value);

  const size_t append_mask = static_cast<size_t>(parsed);
  const size_t keep_mask = static_cast<size_t>(!parsed);
  const size_t pending_index = ev.pending_length_out * append_mask;
  ev.pending_bytes[pending_index] = static_cast<uint8_t>(
      append_mask * static_cast<size_t>(byte_value) +
      keep_mask * static_cast<size_t>(ev.pending_bytes[pending_index]));

  const size_t next_pending_length = ev.pending_length_out + append_mask;
  const int32_t error_value = static_cast<int32_t>(!parsed) * error_code(error::internal_error) +
                              static_cast<int32_t>(parsed) * ev.error_out;
  set_detokenize_error(ev, error_value, ev.output_length_out, next_pending_length);
}

inline void write_pending_head_sequence(const event::detokenize & ev) noexcept {
  size_t & pending_length = ev.pending_length_out;
  size_t & output_length = ev.output_length_out;
  const size_t needed = utf8_sequence_length(ev.pending_bytes[0]);
  const bool wrote = write_bytes(ev,
                                 output_length,
                                 pending_length,
                                 reinterpret_cast<const char *>(ev.pending_bytes),
                                 needed);
  const size_t consumed = needed * static_cast<size_t>(wrote);
  const size_t remaining = pending_length - consumed;
  const size_t shift = consumed * static_cast<size_t>(remaining > 0);
  for (size_t i = 0; i < remaining; ++i) {
    ev.pending_bytes[i] = ev.pending_bytes[i + shift];
  }
  set_detokenize_error(ev, ev.error_out, output_length, remaining);
}

inline void write_text_piece(const event::detokenize & ev,
                             const context & ctx) noexcept {
  const std::string_view piece = token_piece(ev, ctx);
  size_t & output_length = ev.output_length_out;
  const size_t pending_length = ev.pending_length_out;
  (void)write_bytes(ev, output_length, pending_length, piece.data(), piece.size());
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

struct mark_model_invalid {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_model_invalid(ev);
  }
};

struct mark_invalid_pending_full {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_invalid_pending_full(ev);
  }
};

struct mark_invalid_pending_not_empty {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_invalid_pending_not_empty(ev);
  }
};

struct mark_invalid_pending_sequence {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_invalid_pending_sequence(ev);
  }
};

struct mark_internal_error {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::mark_internal_error(ev);
  }
};

struct append_byte_piece {
  void operator()(const event::detokenize & ev, const context & ctx) const noexcept {
    detail::append_byte_piece(ev, ctx);
  }
};

struct write_pending_head_sequence {
  void operator()(const event::detokenize & ev) const noexcept {
    detail::write_pending_head_sequence(ev);
  }
};

struct write_text_piece {
  void operator()(const event::detokenize & ev, const context & ctx) const noexcept {
    detail::write_text_piece(ev, ctx);
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
inline constexpr mark_model_invalid mark_model_invalid{};
inline constexpr mark_invalid_pending_full mark_invalid_pending_full{};
inline constexpr mark_invalid_pending_not_empty mark_invalid_pending_not_empty{};
inline constexpr mark_invalid_pending_sequence mark_invalid_pending_sequence{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr append_byte_piece append_byte_piece{};
inline constexpr write_pending_head_sequence write_pending_head_sequence{};
inline constexpr write_text_piece write_text_piece{};
inline constexpr mark_done mark_done{};
inline constexpr notify_bind_done notify_bind_done{};
inline constexpr notify_bind_error notify_bind_error{};
inline constexpr notify_detokenize_done notify_detokenize_done{};
inline constexpr notify_detokenize_error notify_detokenize_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::detokenizer::action
