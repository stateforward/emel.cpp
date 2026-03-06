#pragma once

#include <array>
#include <cstddef>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"

namespace emel::text::encoders::action {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

inline void signal_unexpected_request(const event::encode & request) noexcept {
  int32_t token_count_sink = 0;
  int32_t error_sink = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  emel::text::encoders::detail::write_optional(request.token_count_out, token_count_sink, 0);
  emel::text::encoders::detail::write_optional(
    request.error_out, error_sink, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));

  event::encode_ctx runtime_ctx{};
  runtime_ctx.token_count = 0;
  runtime_ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
  emel::text::encoders::detail::publish_result(request, runtime_ctx);
}

}  // namespace detail

struct begin_encode {
  void operator()(const event::encode_runtime & ev, context &) const noexcept {
    ev.ctx.token_count = 0;
    ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct sync_vocab {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    ctx.vocab = &ev.request.vocab;
    ctx.tables_ready = false;
    ctx.ugm_ready = false;
  }
};

struct reject_invalid_encode {
  void operator()(const event::encode_runtime & ev, context &) const noexcept {
    ev.ctx.token_count = 0;
    ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
  }
};

struct run_encode {
  void operator()(const event::encode_runtime &, context &) const noexcept {
  }
};

struct mark_done {
  void operator()(const event::encode_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct ensure_last_error {
  void operator()(const event::encode_runtime & ev, context &) const noexcept {
    const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend), ev.ctx.err};
    ev.ctx.err = errors[static_cast<size_t>(ev.ctx.err != emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok))];
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    if constexpr (requires { runtime_ev.ctx.err; runtime_ev.ctx.token_count; }) {
      runtime_ev.ctx.token_count = 0;
      runtime_ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
    } else if constexpr (requires { runtime_ev.request; }) {
      detail::signal_unexpected_request(runtime_ev.request);
    }
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr sync_vocab sync_vocab{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr run_encode run_encode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::action
