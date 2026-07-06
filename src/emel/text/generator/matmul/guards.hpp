#pragma once

#include "emel/text/generator/matmul/context.hpp"
#include "emel/text/generator/matmul/events.hpp"

namespace emel::text::generator::matmul::guard {

struct guard_parallel_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return ctx.parallel_matmul_lanes != nullptr && ev.request.src0.ne[1] > 0u;
  }
};

struct guard_parallel_unavailable {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return ctx.parallel_matmul_lanes == nullptr || ev.request.src0.ne[1] == 0u;
  }
};

struct guard_serial_accepted {
  bool operator()(const event::execute_serial & ev,
                  const action::context &) const noexcept {
    return ev.result.lane_accepted[0];
  }
};

struct guard_serial_rejected {
  bool operator()(const event::execute_serial & ev,
                  const action::context &) const noexcept {
    return !ev.result.lane_accepted[0];
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

template <size_t lane_index>
struct guard_parallel_lane_rejected {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    if (ev.result.lane_count <= lane_index) {
      return false;
    }
    for (size_t index = 0u; index < lane_index; ++index) {
      if (!ev.result.lane_accepted[index]) {
        return false;
      }
    }
    return ev.result.all_submitted && ev.result.joined &&
           !ev.result.lane_accepted[lane_index];
  }
};

struct guard_parallel_all_lanes_accepted {
  bool operator()(const event::execute_parallel & ev,
                  const action::context &) const noexcept {
    if (!ev.result.all_submitted || !ev.result.joined) {
      return false;
    }
    for (size_t lane = 0u; lane < ev.result.lane_count; ++lane) {
      if (!ev.result.lane_accepted[lane]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace emel::text::generator::matmul::guard
