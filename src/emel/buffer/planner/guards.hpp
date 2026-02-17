#pragma once

#include "emel/buffer/planner/actions.hpp"

namespace emel::buffer::planner::guard {

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

struct valid_plan {
  bool operator()(const event::plan & ev, const action::context &) const noexcept {
    return action::detail::valid_plan_event(ev) && action::detail::valid_strategy(ev.strategy);
  }
};

struct has_request {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    return ev.request != nullptr;
  }
};

struct missing_request {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    return ev.request == nullptr;
  }
};

}  // namespace emel::buffer::planner::guard
