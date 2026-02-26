#pragma once

#include "emel/graph/processor/bind_step/context.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"

namespace emel::graph::processor::bind_step::guard {

struct phase_prefailed {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(processor::error::none);
  }
};

struct phase_request_callback {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(processor::error::none) &&
           ev.request.bind_inputs != nullptr;
  }
};

struct phase_missing_callback {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(processor::error::none) &&
           ev.request.bind_inputs == nullptr;
  }
};

struct callback_ok {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.phase_callback_ok && ev.ctx.phase_callback_err == 0;
  }
};

struct callback_error {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return ev.ctx.phase_callback_err != 0;
  }
};

struct callback_failed_without_error {
  bool operator()(const processor::event::execute_step & ev, const action::context &) const noexcept {
    return !ev.ctx.phase_callback_ok && ev.ctx.phase_callback_err == 0;
  }
};

}  // namespace emel::graph::processor::bind_step::guard
