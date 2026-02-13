#pragma once

#include "emel/buffer_planner/actions.hpp"

namespace emel::buffer_planner::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event &, const action::context & ctx) const noexcept {
    return ctx.pending_error == EMEL_OK;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event &, const action::context & ctx) const noexcept {
    return ctx.pending_error != EMEL_OK;
  }
};

}  // namespace emel::buffer_planner::guard
