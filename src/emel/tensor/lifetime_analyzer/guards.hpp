#pragma once

#include "emel/tensor/lifetime_analyzer/actions.hpp"

namespace emel::tensor::lifetime_analyzer::guard {

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

}  // namespace emel::tensor::lifetime_analyzer::guard
