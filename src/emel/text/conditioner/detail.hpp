#pragma once

#include <cstdint>

#include "emel/text/conditioner/errors.hpp"

namespace emel::text::conditioner::detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.request; ev.ctx; }) {
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

constexpr error from_external_error_code(const int32_t err) noexcept {
  switch (err) {
    case to_local_error_code(error::none):
      return error::none;
    case to_local_error_code(error::invalid_argument):
      return error::invalid_argument;
    case to_local_error_code(error::model_invalid):
      return error::model_invalid;
    case to_local_error_code(error::capacity):
      return error::capacity;
    case to_local_error_code(error::backend):
      return error::backend;
    case to_local_error_code(error::untracked):
      return error::untracked;
    case 5:  // legacy EMEL_ERR_MODEL_INVALID
      return error::model_invalid;
    case 6:  // legacy EMEL_ERR_BACKEND
      return error::backend;
    default:
      return error::untracked;
  }
}

}  // namespace emel::text::conditioner::detail
