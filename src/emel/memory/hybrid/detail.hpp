#pragma once

#include "emel/emel.h"
#include "emel/error/error.hpp"
#include "emel/memory/hybrid/errors.hpp"

namespace emel::memory::hybrid::detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline emel::error::type cast_api_error(const int32_t err) noexcept {
  return static_cast<emel::error::type>(err);
}

inline bool is_backend_family_error(const int32_t err) noexcept {
  return err == static_cast<int32_t>(emel::error::cast(hybrid::error::backend_error)) ||
         err == static_cast<int32_t>(emel::error::cast(hybrid::error::out_of_memory)) ||
         err == static_cast<int32_t>(emel::error::cast(hybrid::error::none));
}

inline bool is_out_of_memory_error(const int32_t err) noexcept {
  return err == static_cast<int32_t>(emel::error::cast(hybrid::error::out_of_memory));
}

inline bool is_backend_or_none_error(const int32_t err) noexcept {
  return err == static_cast<int32_t>(emel::error::cast(hybrid::error::backend_error)) ||
         err == static_cast<int32_t>(emel::error::cast(hybrid::error::none));
}

}  // namespace emel::memory::hybrid::detail
