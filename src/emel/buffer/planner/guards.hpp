#pragma once

#include "emel/buffer/planner/actions.hpp"

namespace emel::buffer::planner::guard {

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

}  // namespace emel::buffer::planner::guard
