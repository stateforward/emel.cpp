#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "emel/error/error.hpp"
#include "emel/tensor/view/errors.hpp"
#include "emel/tensor/view/events.hpp"

namespace emel::tensor::view::detail {

template <class value_type>
value_type & bind_or_sink(value_type * ptr, value_type & sink) noexcept {
  value_type * choices[2] = {&sink, ptr};
  return *choices[static_cast<size_t>(ptr != nullptr)];
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

template <class policy>
struct capture_tensor_view_runtime {
  using policy_type = policy;
  const event::capture_tensor_view<policy> & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

}  // namespace emel::tensor::view::detail
