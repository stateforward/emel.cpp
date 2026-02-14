#pragma once

#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"

namespace emel::tensor::allocator::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err == EMEL_OK;
    }
    return true;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err != EMEL_OK;
    }
    return false;
  }
};

}  // namespace emel::tensor::allocator::guard
