#pragma once

#include <type_traits>

#include "emel/encoder/context.hpp"
#include "emel/encoder/events.hpp"

namespace emel::encoder::action {

namespace detail {

inline void dispatch_done(const event::encode & ev, const int32_t token_count) {
  if (ev.dispatch_done == nullptr || ev.owner_sm == nullptr) {
    return;
  }
  ev.dispatch_done(ev.owner_sm, events::encoding_done{&ev, token_count});
}

inline void dispatch_error(const event::encode & ev, const int32_t err) {
  if (ev.dispatch_error == nullptr || ev.owner_sm == nullptr) {
    return;
  }
  ev.dispatch_error(ev.owner_sm, events::encoding_error{&ev, err});
}

inline int32_t normalize_error(const int32_t phase_error, const int32_t last_error) noexcept {
  if (last_error != EMEL_OK) {
    return last_error;
  }
  if (phase_error != EMEL_OK) {
    return phase_error;
  }
  return EMEL_ERR_BACKEND;
}

}  // namespace detail

struct reject_invalid_encode {
  void operator()(const event::encode & ev, context & ctx) const {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    detail::dispatch_error(ev, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct run_encode {
  void operator()(context &) const noexcept {
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
    ctx.last_error = detail::normalize_error(ctx.phase_error, ctx.last_error);
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & ctx) const noexcept {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    using decayed_event = std::decay_t<Event>;
    if constexpr (std::is_same_v<decayed_event, event::encode>) {
      if (ev.token_count_out != nullptr) {
        *ev.token_count_out = 0;
      }
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      detail::dispatch_error(ev, EMEL_ERR_INVALID_ARGUMENT);
    } else if constexpr (std::is_same_v<decayed_event, events::encoding_done> ||
                         std::is_same_v<decayed_event, events::encoding_error>) {
      if (ev.request == nullptr) {
        return;
      }
      if (ev.request->token_count_out != nullptr) {
        *ev.request->token_count_out = 0;
      }
      if (ev.request->error_out != nullptr) {
        *ev.request->error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      detail::dispatch_error(*ev.request, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr run_encode run_encode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::encoder::action
