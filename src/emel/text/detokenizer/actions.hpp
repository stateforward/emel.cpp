#pragma once

#include "emel/text/detokenizer/context.hpp"
#include "emel/text/detokenizer/detail.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::action {

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
