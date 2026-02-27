#pragma once

#include <cstdint>

#include "emel/text/conditioner/errors.hpp"

namespace emel::text::conditioner::detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type &ev) noexcept {
  if constexpr (requires {
                  ev.request;
                  ev.ctx;
                }) {
    return (ev);
  } else if constexpr (requires { ev.event_; }) {
    // Boost.SML may wrap events in internal back:: wrappers exposing event_.
    return (ev.event_);
  } else {
    return (ev);
  }
}

constexpr int32_t to_local_error_code(const error err) noexcept {
  return static_cast<int32_t>(err);
}

} // namespace emel::text::conditioner::detail
