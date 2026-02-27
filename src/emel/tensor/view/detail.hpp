#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/tensor/view/errors.hpp"
#include "emel/tensor/view/events.hpp"

namespace emel::tensor::view::detail {

template <class value_type>
value_type & bind_or_sink(value_type * ptr, value_type & sink) noexcept {
  return ptr != nullptr ? *ptr : sink;
}

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  bool accepted = false;
};

struct capture_tensor_view_runtime {
  const event::capture_tensor_view & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

}  // namespace emel::tensor::view::detail
