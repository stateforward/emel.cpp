#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

#include "emel/text/generator/matmul/context.hpp"
#include "emel/text/generator/matmul/events.hpp"

namespace emel::text::generator::matmul::action {

struct effect_configure_kernel_kind {
  void operator()(const event::configure_kernel_kind & ev, context & ctx) const noexcept {
    ctx.kernel_kind = ev.kind;
    ctx.kernel.set_kind(ctx.kernel_kind);
    for (size_t lane = 0u;
         lane < ctx.lane_capacity && ctx.lane_kernels != nullptr; ++lane) {
      ctx.lane_kernels[lane].set_kind(ctx.kernel_kind);
    }
  }
};

struct effect_execute_serial {
  void operator()(const event::execute_serial & ev, context & ctx) const noexcept {
    ev.result = {};
    ev.result.lane_count = 1u;
    ev.result.all_submitted = true;
    ev.result.joined = true;
    ctx.kernel.set_kind(ctx.kernel_kind);
    ev.result.all_lanes_accepted = ctx.kernel.process_event(ev.request);
  }
};

inline void execute_parallel_with_group_rows(
    const event::execute_parallel & ev,
    context & ctx,
    const uint64_t group_rows) noexcept {
  ev.result = {};
  const size_t lane_count = detail::compute_matmul_row_slices(
      ev.request.src0.ne[1], group_rows, ctx.active_lanes,
      std::span<detail::matmul_row_slice>{ctx.row_slices.get(), ctx.lane_capacity});
  ev.result.lane_count = lane_count;

  lane_join_group group{};
  bool all_submitted = true;
  for (size_t lane = 1u; lane < lane_count; ++lane) {
    ctx.lane_events[lane] =
        detail::compute_sliced_mul_mat_event(ev.request, group_rows, ctx.row_slices[lane]);
    auto & lane_kernel = ctx.lane_kernels[lane];
    lane_kernel.set_kind(ctx.kernel_kind);
    ctx.lane_dispatches[lane] = lane_dispatch{
        .kernel = &lane_kernel,
        .request = &ctx.lane_events[lane],
    };
    const bool submitted = ctx.parallel_matmul_lanes.try_submit(
        group, lane_task{
                   .run = run_lane_dispatch,
                   .ctx = &ctx.lane_dispatches[lane],
               });
    all_submitted = all_submitted && submitted;
  }

  ctx.lane_events[0] =
      detail::compute_sliced_mul_mat_event(ev.request, group_rows, ctx.row_slices[0]);
  ctx.lane_kernels[0].set_kind(ctx.kernel_kind);
  bool all_accepted = ctx.lane_kernels[0].process_event(ctx.lane_events[0]);
  ev.result.all_submitted = all_submitted;
  ev.result.joined = group.wait();
  for (size_t lane = 1u; lane < lane_count; ++lane) {
    all_accepted = all_accepted && ctx.lane_dispatches[lane].accepted;
  }
  ev.result.all_lanes_accepted = all_accepted;
}

struct effect_execute_parallel_x8 {
  void operator()(const event::execute_parallel & ev,
                  context & ctx) const noexcept {
    execute_parallel_with_group_rows(
        ev, ctx, emel::kernel::detail::quant::Q4_K_X8_ROWS);
  }
};

struct effect_execute_parallel_x4 {
  void operator()(const event::execute_parallel & ev,
                  context & ctx) const noexcept {
    execute_parallel_with_group_rows(
        ev, ctx, emel::kernel::detail::quant::Q8_0_X4_ROWS);
  }
};

struct effect_execute_parallel_unit {
  void operator()(const event::execute_parallel & ev,
                  context & ctx) const noexcept {
    execute_parallel_with_group_rows(ev, ctx, 1u);
  }
};

struct effect_accept_serial_execution {
  void operator()(const event::execute_serial & ev, context &) const noexcept {
    ev.accepted = true;
  }
};

struct effect_accept_parallel_execution {
  void operator()(const event::execute_parallel & ev, context &) const noexcept {
    ev.accepted = true;
  }
};

struct effect_reject_serial_execution {
  void operator()(const event::execute_serial & ev, context &) const noexcept {
    ev.accepted = false;
  }
};

struct effect_reject_parallel_execution {
  void operator()(const event::execute_parallel & ev, context &) const noexcept {
    ev.accepted = false;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.accepted; }) {
      ev.accepted = false;
    }
  }
};

template <class counter_fn>
uint64_t compute_kernel_counter_total(const context & ctx, counter_fn && counter) noexcept {
  uint64_t total = std::invoke(counter, ctx.kernel);
  for (size_t lane = 0u;
       lane < ctx.lane_capacity && ctx.lane_kernels != nullptr; ++lane) {
    total += std::invoke(counter, ctx.lane_kernels[lane]);
  }
  return total;
}

inline constexpr effect_configure_kernel_kind effect_configure_kernel_kind{};
inline constexpr effect_execute_serial effect_execute_serial{};
inline constexpr effect_execute_parallel_x8 effect_execute_parallel_x8{};
inline constexpr effect_execute_parallel_x4 effect_execute_parallel_x4{};
inline constexpr effect_execute_parallel_unit effect_execute_parallel_unit{};
inline constexpr effect_accept_serial_execution effect_accept_serial_execution{};
inline constexpr effect_accept_parallel_execution effect_accept_parallel_execution{};
inline constexpr effect_reject_serial_execution effect_reject_serial_execution{};
inline constexpr effect_reject_parallel_execution effect_reject_parallel_execution{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

}  // namespace emel::text::generator::matmul::action
