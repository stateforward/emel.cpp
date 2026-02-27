#pragma once

#include <algorithm>
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

inline void write_error(int32_t * error_out, const int32_t err) noexcept {
  if (error_out != nullptr) {
    *error_out = err;
  }
}

inline void clear_request(context &) noexcept {}

inline size_t read_output_length(const event::detokenize & ev) noexcept {
  return ev.output_length_out != nullptr ? *ev.output_length_out : 0;
}

inline size_t read_pending_length(const event::detokenize & ev) noexcept {
  return ev.pending_length_out != nullptr ? *ev.pending_length_out : ev.pending_length;
}

inline void write_lengths(const event::detokenize & ev,
                          const size_t output_length,
                          const size_t pending_length) noexcept {
  if (ev.output_length_out != nullptr) {
    *ev.output_length_out = output_length;
  }
  if (ev.pending_length_out != nullptr) {
    *ev.pending_length_out = pending_length;
  }
}

inline void set_bind_error(const event::bind & ev, const int32_t err) noexcept {
  write_error(ev.error_out, err);
}

inline void set_detokenize_error(const event::detokenize & ev,
                                 const int32_t err,
                                 const size_t output_length,
                                 const size_t pending_length) noexcept {
  write_lengths(ev, output_length, pending_length);
  write_error(ev.error_out, err);
}

inline bool write_bytes(const event::detokenize & ev,
                        size_t & output_length,
                        const size_t pending_length,
                        const char * bytes,
                        const size_t len) noexcept {
  if (len == 0) {
    return true;
  }
  if (ev.output == nullptr || output_length + len > ev.output_capacity) {
    set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
    return false;
  }
  std::memcpy(ev.output + output_length, bytes, len);
  output_length += len;
  return true;
}

inline bool flush_pending_complete_sequences(const event::detokenize & ev,
                                             size_t & pending_length,
                                             size_t & output_length) noexcept {
  while (pending_length > 0) {
    const uint8_t lead = ev.pending_bytes[0];
    const size_t needed = detail::utf8_sequence_length(lead);
    if (needed == 0) {
      set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
      return false;
    }
    if (pending_length < needed) {
      return true;
    }
    for (size_t idx = 1; idx < needed; ++idx) {
      if (!detail::is_utf8_continuation(ev.pending_bytes[idx])) {
        set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
        return false;
      }
    }

    if (!write_bytes(ev, output_length, pending_length,
                     reinterpret_cast<const char *>(ev.pending_bytes), needed)) {
      return false;
    }

    const size_t remaining = pending_length - needed;
    if (remaining > 0) {
      std::memmove(ev.pending_bytes, ev.pending_bytes + needed, remaining);
    }
    pending_length = remaining;
  }

  return true;
}

struct begin_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    set_bind_error(ev, error_code(error::none));
    ctx.vocab = ev.vocab;
    ctx.is_bound = false;
  }
};

struct reject_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    ctx.is_bound = false;
    set_bind_error(ev, error_code(error::invalid_request));
  }
};

struct commit_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    if (ctx.vocab == nullptr) {
      ctx.is_bound = false;
      set_bind_error(ev, error_code(error::invalid_request));
      return;
    }
    ctx.is_bound = true;
    set_bind_error(ev, error_code(error::none));
  }
};

struct ensure_bind_error {
  void operator()(const event::bind & ev) const noexcept {
    if (ev.error_out == nullptr || *ev.error_out != error_code(error::none)) {
      return;
    }
    *ev.error_out = error_code(error::backend_error);
  }
};

struct notify_bind_done {
  void operator()(const event::bind & ev) const noexcept {
    (void)ev.dispatch_done(ev.owner_sm, events::binding_done{&ev});
  }
};

struct notify_bind_error {
  void operator()(const event::bind & ev) const noexcept {
    (void)ev.dispatch_error(ev.owner_sm, events::binding_error{&ev, *ev.error_out});
  }
};

struct begin_detokenize {
  void operator()(const event::detokenize & ev) const noexcept {
    set_detokenize_error(ev, error_code(error::none), 0, ev.pending_length);
  }
};

struct reject_detokenize {
  void operator()(const event::detokenize & ev) const noexcept {
    set_detokenize_error(ev, error_code(error::invalid_request), 0, ev.pending_length);
  }
};

struct decode_token {
  void operator()(const event::detokenize & ev, const context & ctx) const noexcept {
    size_t pending_length = ev.pending_length;
    size_t output_length = 0;
    set_detokenize_error(ev, error_code(error::none), output_length, pending_length);

    if (ctx.vocab == nullptr || !ctx.is_bound || ev.pending_bytes == nullptr ||
        ev.pending_capacity == 0 || pending_length > ev.pending_capacity ||
        (ev.output == nullptr && ev.output_capacity > 0)) {
      set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
      return;
    }

    if (ev.token_id < 0 ||
        static_cast<uint32_t>(ev.token_id) >= ctx.vocab->n_tokens) {
      set_detokenize_error(ev, error_code(error::model_invalid), output_length, pending_length);
      return;
    }

    const auto & entry = ctx.vocab->entries[static_cast<uint32_t>(ev.token_id)];
    if (!ev.emit_special && detail::is_special_token_type(entry.type)) {
      set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
      return;
    }

    const std::string_view piece(ctx.vocab->token_storage.data() + entry.text_offset,
                                 entry.text_length);

    uint8_t byte_value = 0;
    if (detail::parse_plamo2_byte_token(piece, byte_value)) {
      if (pending_length >= ev.pending_capacity) {
        set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
        return;
      }
      ev.pending_bytes[pending_length] = byte_value;
      pending_length += 1;
      if (!flush_pending_complete_sequences(ev, pending_length, output_length)) {
        return;
      }
      set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
      return;
    }

    if (!flush_pending_complete_sequences(ev, pending_length, output_length)) {
      return;
    }
    if (pending_length != 0) {
      set_detokenize_error(ev, error_code(error::invalid_request), output_length, pending_length);
      return;
    }

    if (!write_bytes(ev, output_length, pending_length, piece.data(), piece.size())) {
      return;
    }

    set_detokenize_error(ev, error_code(error::none), output_length, pending_length);
  }
};

struct mark_done {
  void operator()(const event::detokenize & ev) const noexcept {
    set_detokenize_error(ev, error_code(error::none), read_output_length(ev), read_pending_length(ev));
  }
};

struct ensure_detokenize_error {
  void operator()(const event::detokenize & ev) const noexcept {
    if (ev.error_out == nullptr || *ev.error_out != error_code(error::none)) {
      return;
    }
    set_detokenize_error(ev, error_code(error::backend_error), read_output_length(ev),
                         read_pending_length(ev));
  }
};

struct notify_detokenize_done {
  void operator()(const event::detokenize & ev) const noexcept {
    (void)ev.dispatch_done(ev.owner_sm,
                           events::detokenize_done{&ev, *ev.output_length_out, *ev.pending_length_out});
  }
};

struct notify_detokenize_error {
  void operator()(const event::detokenize & ev) const noexcept {
    (void)ev.dispatch_error(ev.owner_sm, events::detokenize_error{&ev, *ev.error_out});
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.output_length_out; }) {
      if (ev.output_length_out != nullptr) {
        *ev.output_length_out = 0;
      }
    }
    if constexpr (requires { ev.pending_length_out; ev.pending_length; }) {
      if (ev.pending_length_out != nullptr) {
        *ev.pending_length_out = ev.pending_length;
      }
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = error_code(error::invalid_request);
      }
    }
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr commit_bind commit_bind{};
inline constexpr ensure_bind_error ensure_bind_error{};
inline constexpr begin_detokenize begin_detokenize{};
inline constexpr reject_detokenize reject_detokenize{};
inline constexpr decode_token decode_token{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_detokenize_error ensure_detokenize_error{};
inline constexpr notify_bind_done notify_bind_done{};
inline constexpr notify_bind_error notify_bind_error{};
inline constexpr notify_detokenize_done notify_detokenize_done{};
inline constexpr notify_detokenize_error notify_detokenize_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::detokenizer::action
