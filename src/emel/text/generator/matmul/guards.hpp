#pragma once

#include "emel/text/generator/matmul/context.hpp"
#include "emel/text/generator/matmul/events.hpp"

namespace emel::text::generator::matmul::guard {

struct guard_parallel_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return ctx.parallel_matmul_lanes.valid() &&
           action::lane_storage_ready(ctx) && ev.request.src0.ne[1] > 0u;
  }
};

struct guard_parallel_unavailable {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return !ctx.parallel_matmul_lanes.valid() ||
           !action::lane_storage_ready(ctx) || ev.request.src0.ne[1] == 0u;
  }
};

struct guard_serial_accepted {
  bool operator()(const event::execute_serial & ev,
                  const action::context &) const noexcept {
    return ev.result.all_lanes_accepted;
  }
};

struct guard_serial_rejected {
  bool operator()(const event::execute_serial & ev,
                  const action::context &) const noexcept {
    return !ev.result.all_lanes_accepted;
  }
};

struct guard_parallel_submission_failed {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    return !ev.result.all_submitted;
  }
};

struct guard_parallel_join_failed {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    return ev.result.all_submitted && !ev.result.joined;
  }
};

struct guard_parallel_lane_rejected {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    return ev.result.all_submitted && ev.result.joined &&
           !ev.result.all_lanes_accepted;
  }
};

struct guard_parallel_all_lanes_accepted {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    return ev.result.all_submitted && ev.result.joined &&
           ev.result.all_lanes_accepted;
  }
};

}  // namespace emel::text::generator::matmul::guard
