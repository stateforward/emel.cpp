#pragma once

#include "emel/text/generator/matmul/context.hpp"
#include "emel/text/generator/matmul/events.hpp"

namespace emel::text::generator::matmul::guard {

inline bool guard_parallel_storage_ready(
    const event::execute_parallel & ev,
    const action::context & ctx) noexcept {
  return ctx.parallel_matmul_lanes.valid() &&
         action::lane_storage_ready(ctx) && ev.request.src0.ne[1] > 0u;
}

inline bool guard_uses_x8_row_groups(
    const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
         code == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
         code == emel::kernel::detail::dtype_q6_k_x8 ||
         code == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
         code == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool guard_uses_x4_row_groups(
    const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
         code == emel::kernel::detail::dtype_q8_0_x4_bl8;
}

struct guard_parallel_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return guard_parallel_storage_ready(ev, ctx);
  }
};

struct guard_parallel_unavailable {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return !ctx.parallel_matmul_lanes.valid() ||
           !action::lane_storage_ready(ctx) || ev.request.src0.ne[1] == 0u;
  }
};

struct guard_parallel_x8_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return guard_parallel_storage_ready(ev, ctx) &&
           guard_uses_x8_row_groups(ev.request.src0.type);
  }
};

struct guard_parallel_x4_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return guard_parallel_storage_ready(ev, ctx) &&
           guard_uses_x4_row_groups(ev.request.src0.type);
  }
};

struct guard_parallel_unit_ready {
  bool operator()(const event::execute_parallel & ev,
                  const action::context & ctx) const noexcept {
    return guard_parallel_storage_ready(ev, ctx) &&
           !guard_uses_x8_row_groups(ev.request.src0.type) &&
           !guard_uses_x4_row_groups(ev.request.src0.type);
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
