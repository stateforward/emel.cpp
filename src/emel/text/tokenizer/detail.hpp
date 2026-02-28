#pragma once

#include <array>
#include <cstddef>

#include "emel/text/tokenizer/errors.hpp"

namespace emel::text::tokenizer::detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type &ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

template <class value_type>
inline void write_optional(value_type *destination, value_type &sink,
                           const value_type value) noexcept {
  value_type *destinations[2] = {&sink, destination};
  value_type *const target =
      destinations[static_cast<size_t>(destination != nullptr)];
  *target = value;
}

template <class event_type>
inline bool ignore_callback(void *, const event_type &) noexcept {
  return true;
}

template <class event_type>
inline void dispatch_optional_callback(void *owner,
                                       bool (*callback)(void *owner,
                                                        const event_type &),
                                       const event_type &payload) noexcept {
  const size_t callback_ready = static_cast<size_t>(callback != nullptr);
  const size_t owner_ready = static_cast<size_t>(owner != nullptr);
  const size_t valid = callback_ready & owner_ready;
  bool (*callbacks[2])(void *, const event_type &) = {
      ignore_callback<event_type>, callback};
  void *owners[2] = {nullptr, owner};
  callbacks[valid](owners[valid], payload);
}

inline int32_t select_error_code(const bool ok,
                                 const int32_t runtime_error) noexcept {
  const std::array<int32_t, 2> fallback_errors = {
      error_code(error::backend_error), runtime_error};
  const int32_t failure_error = fallback_errors[static_cast<size_t>(
      runtime_error != error_code(error::none))];
  const std::array<int32_t, 2> final_errors = {failure_error,
                                               error_code(error::none)};
  return final_errors[static_cast<size_t>(ok)];
}

template <class request_type, class done_event_type, class error_event_type>
inline void dispatch_result_callback(
    const bool ok, const request_type &request, const done_event_type &done_ev,
    const error_event_type &error_ev,
    void (*on_done)(const request_type &, const done_event_type &,
                    const error_event_type &) noexcept,
    void (*on_error)(const request_type &, const done_event_type &,
                     const error_event_type &) noexcept) noexcept {
  using dispatch_fn_type =
      void (*)(const request_type &, const done_event_type &,
               const error_event_type &) noexcept;
  const std::array<dispatch_fn_type, 2> dispatchers = {on_error, on_done};
  dispatchers[static_cast<size_t>(ok)](request, done_ev, error_ev);
}

} // namespace emel::text::tokenizer::detail
