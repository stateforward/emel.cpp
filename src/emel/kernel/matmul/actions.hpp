#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "emel/kernel/matmul/context.hpp"
#include "emel/kernel/matmul/events.hpp"

namespace emel::kernel::matmul::action {

struct lane_dispatch {
  emel::kernel::sm *kernel = nullptr;
  const emel::kernel::event::op_mul_mat *request = nullptr;
  bool accepted = false;
};

template <size_t... lanes>
inline void configure_lane_kinds(context &ctx,
                                 std::index_sequence<lanes...>) noexcept {
  const emel::kernel::event::configure_kind configure{ctx.kernel_kind};
  (ctx.lanes->kernels[lanes].process_event(configure), ...);
}

struct effect_configure_kernel_kind {
  void operator()(const event::configure_kernel_kind &ev,
                  context &ctx) const noexcept {
    ctx.kernel_kind = ev.kind;
    ctx.kernel.process_event(
        emel::kernel::event::configure_kind{ctx.kernel_kind});
    configure_lane_kinds(ctx, std::make_index_sequence<MAX_PARALLEL_LANES>{});
  }
};

struct effect_execute_serial {
  void operator()(const event::execute_serial &ev,
                  context &ctx) const noexcept {
    ev.result = {};
    ev.result.lane_count = 1u;
    ev.result.all_submitted = true;
    ev.result.all_lanes_accepted = ctx.kernel.process_event(ev.request);
  }
};

template <size_t lane, size_t lane_count, uint64_t group_rows>
inline detail::matmul_row_slice
compute_fixed_row_slice(const uint64_t rows) noexcept {
  const uint64_t groups = (rows + group_rows - 1u) / group_rows;
  const uint64_t groups_per_lane = groups / lane_count;
  const uint64_t extra_groups = groups % lane_count;
  const uint64_t begin_group =
      lane * groups_per_lane + std::min<uint64_t>(lane, extra_groups);
  const uint64_t lane_groups =
      groups_per_lane + static_cast<uint64_t>(lane < extra_groups);
  const uint64_t begin_row = begin_group * group_rows;
  const uint64_t end_row =
      std::min(rows, (begin_group + lane_groups) * group_rows);
  return detail::matmul_row_slice{
      .row_begin = static_cast<int32_t>(begin_row),
      .row_count = static_cast<int32_t>(end_row - begin_row),
  };
}

template <size_t lane_count, uint64_t group_rows, size_t... lanes>
inline void prepare_fixed_lanes(
    const event::execute_parallel &ev, context &ctx,
    std::array<detail::matmul_row_slice, lane_count> &row_slices,
    std::array<emel::kernel::event::op_mul_mat, lane_count> &lane_events,
    std::array<lane_dispatch, MAX_PARALLEL_LANES> &lane_dispatches,
    std::index_sequence<lanes...>) noexcept {
  ((row_slices[lanes] = compute_fixed_row_slice<lanes, lane_count, group_rows>(
        ev.request.src0.ne[1]),
    lane_events[lanes] = detail::compute_sliced_mul_mat_event(
        ev.request, group_rows, row_slices[lanes]),
    lane_dispatches[lanes] =
        lane_dispatch{
            .kernel = &ctx.lanes->kernels[lanes],
            .request = &lane_events[lanes],
        }),
   ...);
}

template <size_t lane_count, size_t... lane_offsets>
inline size_t submit_fixed_worker_lanes(
    context &ctx, lane_pool::join_group &group,
    std::array<lane_dispatch, MAX_PARALLEL_LANES> &lane_dispatches,
    std::index_sequence<lane_offsets...>) noexcept {
  ((lane_dispatches[lane_offsets + 1u].accepted = false), ...);
  return ctx.parallel_matmul_lanes->try_submit_batch(
      group, ([&dispatch = lane_dispatches[lane_offsets + 1u]]() noexcept {
        dispatch.accepted = dispatch.kernel->process_event(*dispatch.request);
      })...);
}

template <size_t... lane_offsets>
inline bool fixed_worker_lanes_accepted(
    const context &ctx,
    const std::array<lane_dispatch, MAX_PARALLEL_LANES> &lane_dispatches,
    std::index_sequence<lane_offsets...>) noexcept {
  (void)ctx;
  return (lane_dispatches[lane_offsets + 1u].accepted && ...);
}

template <size_t lane_count, uint64_t group_rows>
struct effect_execute_parallel {
  void operator()(const event::execute_parallel &ev,
                  context &ctx) const noexcept {
    static_assert(lane_count == 2u || lane_count == 4u || lane_count == 8u);
    ev.result = {};
    ev.result.lane_count = lane_count;
    std::array<detail::matmul_row_slice, lane_count> row_slices = {};
    std::array<emel::kernel::event::op_mul_mat, lane_count> lane_events = {};
    std::array<lane_dispatch, MAX_PARALLEL_LANES> lane_dispatches = {};
    prepare_fixed_lanes<lane_count, group_rows>(
        ev, ctx, row_slices, lane_events, lane_dispatches,
        std::make_index_sequence<lane_count>{});

    lane_pool::join_group group{};
    ev.result.submitted_worker_lanes = submit_fixed_worker_lanes<lane_count>(
        ctx, group, lane_dispatches,
        std::make_index_sequence<lane_count - 1u>{});
    ev.result.all_submitted =
        ev.result.submitted_worker_lanes == lane_count - 1u;
    const bool owner_accepted =
        ctx.lanes->kernels[0].process_event(lane_events[0]);
    (void)group.wait();
    ev.result.drained_worker_lanes = ev.result.submitted_worker_lanes;
    ev.result.all_lanes_accepted =
        owner_accepted &&
        fixed_worker_lanes_accepted(
            ctx, lane_dispatches, std::make_index_sequence<lane_count - 1u>{});
  }
};

struct effect_accept_serial_execution {
  void operator()(const event::execute_serial &ev, context &) const noexcept {
    ev.accepted = true;
  }
};

struct effect_accept_parallel_execution {
  void operator()(const event::execute_parallel &ev, context &) const noexcept {
    ev.accepted = true;
  }
};

struct effect_reject_serial_execution {
  void operator()(const event::execute_serial &ev, context &) const noexcept {
    ev.accepted = false;
  }
};

struct effect_reject_parallel_execution {
  void operator()(const event::execute_parallel &ev, context &) const noexcept {
    ev.accepted = false;
  }
};

template <class event_type> struct effect_emit_execute_done {
  void operator()(const event_type &ev, context &) const noexcept {
    ev.on_done(events::execute_done<event_type>{.request = ev});
  }
};

template <class event_type> struct effect_emit_execute_error {
  void operator()(const event_type &ev, context &) const noexcept {
    ev.on_error(events::execute_error<event_type>{.request = ev});
  }
};

struct effect_emit_serial_done
    : effect_emit_execute_done<event::execute_serial> {};
struct effect_emit_serial_error
    : effect_emit_execute_error<event::execute_serial> {};
struct effect_emit_parallel_done
    : effect_emit_execute_done<event::execute_parallel> {};
struct effect_emit_parallel_error
    : effect_emit_execute_error<event::execute_parallel> {};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.accepted; }) {
      ev.accepted = false;
    }
  }
};

template <size_t... lanes>
inline bool capture_lane_diagnostics(
    context &ctx,
    std::array<emel::kernel::event::diagnostics, MAX_PARALLEL_LANES> &out,
    std::index_sequence<lanes...>) noexcept {
  return (ctx.lanes->kernels[lanes].process_event(
              emel::kernel::event::capture_diagnostics{out[lanes]}) &&
          ...);
}

template <class member_type, size_t... lanes>
inline uint64_t
compute_diagnostics_total(const emel::kernel::event::diagnostics &serial,
                          const std::array<emel::kernel::event::diagnostics,
                                           MAX_PARALLEL_LANES> &parallel,
                          member_type emel::kernel::event::diagnostics::*member,
                          std::index_sequence<lanes...>) noexcept {
  return serial.*member + ((parallel[lanes].*member) + ... + 0u);
}

struct effect_capture_diagnostics {
  void operator()(const event::capture_diagnostics &ev,
                  context &ctx) const noexcept {
    ev.out = {};
    emel::kernel::event::diagnostics serial = {};
    std::array<emel::kernel::event::diagnostics, MAX_PARALLEL_LANES> parallel =
        {};
    const bool serial_accepted = ctx.kernel.process_event(
        emel::kernel::event::capture_diagnostics{serial});
    const bool lanes_accepted = capture_lane_diagnostics(
        ctx, parallel, std::make_index_sequence<MAX_PARALLEL_LANES>{});
    const auto total = [&serial, &parallel](auto member) noexcept {
      return compute_diagnostics_total(
          serial, parallel, member,
          std::make_index_sequence<MAX_PARALLEL_LANES>{});
    };
    ev.out.optimized_flash_dispatch_calls = total(
        &emel::kernel::event::diagnostics::optimized_flash_dispatch_calls);
    ev.out.shared_flash_dispatch_calls =
        total(&emel::kernel::event::diagnostics::shared_flash_dispatch_calls);
    ev.out.optimized_q2_dispatch_calls =
        total(&emel::kernel::event::diagnostics::optimized_q2_dispatch_calls);
    ev.out.shared_q2_dispatch_calls =
        total(&emel::kernel::event::diagnostics::shared_q2_dispatch_calls);
    ev.out.optimized_q3_dispatch_calls =
        total(&emel::kernel::event::diagnostics::optimized_q3_dispatch_calls);
    ev.out.shared_q3_dispatch_calls =
        total(&emel::kernel::event::diagnostics::shared_q3_dispatch_calls);
    ev.out.optimized_q4_dispatch_calls =
        total(&emel::kernel::event::diagnostics::optimized_q4_dispatch_calls);
    ev.out.optimized_q4_vector_dispatch_calls = total(
        &emel::kernel::event::diagnostics::optimized_q4_vector_dispatch_calls);
    ev.out.optimized_q4_vector_packed_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q4_vector_packed_dispatch_calls);
    ev.out.optimized_q4_vector_packed_q8_rhs_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q4_vector_packed_q8_rhs_dispatch_calls);
    ev.out.shared_q4_dispatch_calls =
        total(&emel::kernel::event::diagnostics::shared_q4_dispatch_calls);
    ev.out.optimized_q6_dispatch_calls =
        total(&emel::kernel::event::diagnostics::optimized_q6_dispatch_calls);
    ev.out.optimized_q6_vector_dispatch_calls = total(
        &emel::kernel::event::diagnostics::optimized_q6_vector_dispatch_calls);
    ev.out.optimized_q6_vector_argmax_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_argmax_dispatch_calls);
    ev.out.optimized_q6_vector_packed_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_packed_dispatch_calls);
    ev.out.optimized_q6_vector_packed_q8_rhs_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_packed_q8_rhs_dispatch_calls);
    ev.out.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls);
    ev.out.optimized_q6_vector_prepared_q8_rhs_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_prepared_q8_rhs_dispatch_calls);
    ev.out.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls);
    ev.out.optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls =
        total(
            &emel::kernel::event::diagnostics::
                optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls);
    ev.out.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls =
        total(&emel::kernel::event::diagnostics::
                  optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls);
    ev.out.shared_q6_dispatch_calls =
        total(&emel::kernel::event::diagnostics::shared_q6_dispatch_calls);
    ev.out.serial_optimized_q4_dispatch_calls =
        serial.optimized_q4_dispatch_calls;
    ev.out.parallel_optimized_q4_dispatch_calls =
        total(&emel::kernel::event::diagnostics::optimized_q4_dispatch_calls) -
        serial.optimized_q4_dispatch_calls;
    ev.accepted = serial_accepted && lanes_accepted;
  }
};

inline constexpr effect_configure_kernel_kind effect_configure_kernel_kind{};
inline constexpr effect_execute_serial effect_execute_serial{};
inline constexpr effect_accept_serial_execution
    effect_accept_serial_execution{};
inline constexpr effect_accept_parallel_execution
    effect_accept_parallel_execution{};
inline constexpr effect_reject_serial_execution
    effect_reject_serial_execution{};
inline constexpr effect_reject_parallel_execution
    effect_reject_parallel_execution{};
inline constexpr effect_capture_diagnostics effect_capture_diagnostics{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::kernel::matmul::action
